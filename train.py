import parameters as params
import os
import torch
import glob
import numpy as np
import optuna
import joblib
from torch.utils.data import Dataset, DataLoader
from dataset import train_Dataset, val_Dataset, test_Dataset
from network import VATMAN
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_num_threads(1) 

def make_five_fold():
    data_root_dir = params.data_root_dir
    feature_extractor = params.feature_extractor
    anomaly_class = params.anomaly_class
    
    data_path = data_root_dir + '/' + feature_extractor + '/' + anomaly_class

    videos = os.listdir(data_path)
    videos.sort()

    fold_size = len(videos) // 6

    five_fold = []
    five_fold_label = []
    for i in range(6):
        fold_videos = videos[i * fold_size : i * fold_size + fold_size]
        
        one_fold = np.array([])
        one_fold_label = []
        for fold_video in fold_videos:
            fold_video_path = data_path + '/' + fold_video

            segments = glob.glob(fold_video_path + '/*.npy')
            segments.sort()

            for segment in segments:
                segment_npy = np.load(segment)

                if 'anomaly' in segment:
                    one_fold_label.append(1)
                else:
                    one_fold_label.append(0)
                
                if len(one_fold) == 0:
                    one_fold = segment_npy
                else:
                    one_fold = np.append(one_fold, segment_npy, axis=0)
                
                # print(one_fold.shape)

        five_fold.append(one_fold)
        five_fold_label.append(one_fold_label)

    print('every fold shape : ')    
    for i in range(len(five_fold)):
        print('fold', str(i), ':', five_fold[i].shape)
        # print('label', len(five_fold_label[i]))
    print('----------------------------')

    return five_fold, five_fold_label


T_AUROC = [0, 0, 0, 0, 0]
T_AUPRC = [0, 0, 0, 0, 0] 
G_AUROC = [0, 0, 0, 0, 0]
G_AUPRC = [0, 0, 0, 0, 0]

def objective(trial, X_train, X_test, X_val, y_test, y_val, fold_num):
    target_path = params.save_root_dir + '/' + params.exp_name + '_' + params.feature_extractor + '_' + params.anomaly_class

    # dynamic hyperparameter tuned by optuna 
    cfg = {
        'batch_size' : trial.suggest_categorical('batch_size', [4, 8, 16, 32]),
        'lr' : trial.suggest_loguniform('lr', 1e-7, 1e-3)
    }

    trainset = train_Dataset(X_train)
    dataloader = DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VATMAN(params.embedding_dim, params.heads, params.embedding_dim // params.heads, params.dropout)
    model.to(device)
    model_parameter = list(model.parameters())

    optimizer = optim.Adam(model_parameter, lr=cfg['lr'])

    max_val_auroc = 0
    max_val_auprc = 0
    for epoch in range(params.epochs):
        model.train()
        print('--------------------------------------------')
        print('Fold : ', fold_num)
        print('Trial : ', trial.number)
        print('Epoch : ', epoch)
        print('class : ', params.anomaly_class)
        print()
        
        running_loss = 0
        for idx, data in enumerate(dataloader):
            data = data.to(device)
            output, _ = model(data)

            loss = F.mse_loss(output, data) 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print('Loss : ', running_loss / len(dataloader))

        with torch.no_grad():
            model.eval()

            val_prediction = []
            val_ground_truth = []

            validationset = val_Dataset(X_val, y_val)
            validationloader = DataLoader(validationset, batch_size=1, shuffle=False)

            for idx, (data, label) in enumerate(validationloader):
                data = data.to(device)
                data = data.squeeze()

                att_tensor = torch.zeros(params.infer_sample_num)
                for jdx in range(params.infer_sample_num):
                    x, att = model(data[jdx].unsqueeze(dim=0))
                    att_tensor[jdx] = att

                if params.infer_attention_mode == 'avg':
                    prediction = att_tensor.mean()
                elif params.infer_attention_mode == 'sum':
                    prediction = att_tensor.sum()
                else:
                    prediction, _ = att_tensor.max()

                att = att.cpu()

                val_prediction.append(np.array(prediction))
                val_ground_truth.append(np.array(label))

        val_prediction = np.array(val_prediction)
        val_prediction = np.squeeze(val_prediction)

        print(val_prediction.shape)
        val_ground_truth = np.array(val_ground_truth)
        val_ground_truth = np.squeeze(val_ground_truth)
        print(val_ground_truth.shape)

        val_fpr, val_tpr, _ = roc_curve(val_ground_truth, val_prediction)

        val_auroc = auc(val_fpr, val_tpr)        
        val_precision, val_recall, _ = precision_recall_curve(val_ground_truth, val_prediction)
        val_auprc = auc(val_recall, val_precision)

        print('Val auroc : ', val_auroc)
        print('Val auprc : ', val_auprc)

        if max_val_auroc < val_auroc:
            max_val_auroc = val_auroc

            if T_AUROC[fold_num] < max_val_auroc:
                T_AUROC[fold_num] = max_val_auroc

                testset = test_Dataset(X_test, y_test)
                testloader = DataLoader(testset, batch_size=1, shuffle=False)

                with torch.no_grad():
                    model.eval()

                    test_prediction = []
                    test_ground_truth = []

                    for idx, (data, label) in enumerate(testloader):
                        data = data.to(device)

                        data = data.squeeze()

                        att_tensor = torch.zeros(params.infer_sample_num)
                        for jdx in range(params.infer_sample_num):
                            x, att = model(data[jdx].unsqueeze(dim=0))
                            att_tensor[jdx] = att

                        if params.infer_attention_mode == 'avg':
                            prediction = att_tensor.mean()
                        elif params.infer_attention_mode == 'sum':
                            prediction = att_tensor.sum()
                        else:
                            prediction, _ = att_tensor.max()

                        att = att.cpu()

                        test_prediction.append(np.array(prediction))
                        test_ground_truth.append(np.array(label))

                
                test_prediction = np.array(test_prediction)
                test_prediction = np.squeeze(test_prediction)
                test_ground_truth = np.array(test_ground_truth)
                test_ground_truth = np.squeeze(test_ground_truth)

                test_fpr, test_tpr, _ = roc_curve(test_ground_truth, test_prediction)

                test_auroc = auc(test_fpr, test_tpr)        
                test_precision, test_recall, _ = precision_recall_curve(test_ground_truth, test_prediction)
                test_auprc = auc(test_recall, test_precision)

                print('Test auroc : ', test_auroc)
                print('Test auprc : ', test_auprc)
                print('True / Total : ', np.mean(test_ground_truth))

                G_AUROC[fold_num] = test_auroc
                G_AUPRC[fold_num] = test_auprc

                torch.save(model.state_dict(), target_path + '/fold' + str(fold_num) + '_saved_model_max_val_auroc.pt')
                print('Model Saved ...')

                f = open(target_path + '/fold' + str(fold_num) + '_raw_data.csv', 'a', newline='')

                wr = csv.writer(f)

                row = []
                row.append('trial:')
                row.append(trial.number)
                wr.writerow(row)

                row = []
                row.append('epoch:')
                row.append(epoch)
                wr.writerow(row)

                row = []
                row.append('prediction:')
                row.extend(test_prediction)
                wr.writerow(row)

                row = []
                row.append('ground_truth:')
                row.extend(test_ground_truth)
                wr.writerow(row)

                row = []
                row.append('true/total:')
                row.append(np.mean(test_ground_truth))
                wr.writerow(row)

                row = []
                row.append('val_auroc:')
                row.append(val_auroc)
                wr.writerow(row)

                row = []
                row.append('val_auprc:')
                row.append(val_auprc)
                wr.writerow(row)

                row = []
                row.append('test_auroc:')
                row.append(test_auroc)
                wr.writerow(row)

                row = []
                row.append('test_auprc:')
                row.append(test_auprc)
                wr.writerow(row)

                row = []
                row.append('-----------------------------------------------')
                wr.writerow(row)
                
                f.close()
                print('Raw data is saved ...')

    return max_val_auroc


def fold_train(X_train, X_test, X_val, y_test, y_val, fold_num):
    sampler = optuna.samplers.TPESampler()

    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(lambda trial : objective(trial, X_train, X_test, X_val, y_test, y_val, fold_num), n_trials=20)
    joblib.dump(study, './optuna_results/optuna_' + str(fold_num) + '.pkl')


def five_fold_cross_validation(): 
    five_fold, five_fold_label = make_five_fold()

    for fold_num in range(5):
        X_test = five_fold[fold_num]
        X_val = five_fold[fold_num + 1]

        y_test = five_fold_label[fold_num]
        y_val = five_fold_label[fold_num + 1]

        X_train = np.array([])
        for i in range(6):
            if i != fold_num and i != fold_num + 1:
                temp_data = five_fold[i]
                temp_label = five_fold_label[i]

                for jdx, data in enumerate(temp_data):
                    data = np.expand_dims(data, axis=0)
                    if temp_label[jdx] == 0:
                        if len(X_train) == 0:
                            X_train = data
                        else:
                            X_train = np.append(X_train, data, axis=0)
                    else:
                        continue

        
        print('Fold : ', fold_num)
        print('X_train : ', X_train.shape)
        print('X_val : ', X_val.shape)
        print('X_test : ', X_test.shape)
        print('---------------------------------------')

        fold_train(X_train, X_test, X_val, y_test, y_val, fold_num)
    
    print('----------------------------------------------------')
    print('<AUROC>')
    for idx in range(5):
        print(G_AUROC[idx])
    print()
    print('<AUPRC>')
    for idx in range(5):
        print(G_AUPRC[idx])
    print()
    print('AVG AUROC : ', sum(G_AUROC) / 5)
    print('AVG AUPRC : ', sum(G_AUPRC) / 5)

    target_path = params.save_root_dir + '/' + params.exp_name + '_' + params.feature_extractor + '_' + params.anomaly_class
    f = open(target_path + '/all_results.csv', 'a', newline='')

    wr = csv.writer(f)

    row = []
    row.append('AUROC')
    wr.writerow(row)

    for idx in range(5):
        row=[]
        row.append(G_AUROC[idx])
        wr.writerow(row)

    row = []
    row.append('AUPRC')
    wr.writerow(row)

    for idx in range(5):
        row=[]
        row.append(G_AUPRC[idx])
        wr.writerow(row)

    row = []
    row.append('AVG AUROC:') 
    row.append(sum(G_AUROC) / 5)
    wr.writerow(row)

    row = []
    row.append('AVG AUPRC:')
    row.append(sum(G_AUPRC) / 5)
    wr.writerow(row) 

    f.close()  





if __name__ == '__main__':
    target_path = params.save_root_dir + '/' + params.exp_name + '_' + params.feature_extractor + '_' + params.anomaly_class

    if os.path.isdir(target_path) == False:
        os.mkdir(target_path)

    five_fold_cross_validation()

