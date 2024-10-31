import torch.utils.data as data
import torch
import numpy as np
import parameters as params

class train_Dataset(data.Dataset):
    def __init__(self, X_train):
        self.X_train = X_train
        np.random.shuffle(self.X_train)

        self.sample_num = len(self.X_train) // params.token_sample_num

        self.tokens = np.array([])
        for idx in range(self.sample_num):
            if idx == 0:
                self.tokens = np.expand_dims(self.X_train[:params.token_sample_num], axis=0)
            else:
                self.tokens = np.append(self.tokens, np.expand_dims(self.X_train[params.token_sample_num * idx : params.token_sample_num * idx + params.token_sample_num], axis=0), axis=0)


    def __getitem__(self, index):
        tensor_segment = torch.FloatTensor(self.tokens[index])
        tensor_segment = torch.squeeze(tensor_segment)

        return tensor_segment
    
    def __len__(self):
        return self.sample_num


class val_Dataset(data.Dataset):
    def __init__(self, X_val, y_val):
        self.X_val = X_val
        self.y_val = y_val
    
    def __getitem__(self, index):
        output = torch.zeros(params.infer_sample_num, params.token_sample_num, params.embedding_dim)
        for jdx in range(params.infer_sample_num):
            X_val_not_x = np.delete(self.X_val, index, axis=0)
            np.random.shuffle(X_val_not_x)
            
            tokens_not_x = X_val_not_x[:params.token_sample_num - 1]
            tokens = np.concatenate((np.expand_dims(self.X_val[index], axis=0), tokens_not_x), axis=0)

            tensor_segment = torch.FloatTensor(tokens)
            tensor_segment = torch.squeeze(tensor_segment)
            output[jdx] = tensor_segment

        return output, self.y_val[index]
    
    def __len__(self):
        return len(self.X_val)

class test_Dataset(data.Dataset):
    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test
    
    def __getitem__(self, index):
        output = torch.zeros(params.infer_sample_num, params.token_sample_num, params.embedding_dim)
        for jdx in range(params.infer_sample_num):
            X_test_not_x = np.delete(self.X_test, index, axis=0)
            np.random.shuffle(X_test_not_x)
            
            tokens_not_x = X_test_not_x[:params.token_sample_num - 1]
            tokens = np.concatenate((np.expand_dims(self.X_test[index], axis=0), tokens_not_x), axis=0)

            tensor_segment = torch.FloatTensor(tokens)
            tensor_segment = torch.squeeze(tensor_segment)
            output[jdx] = tensor_segment

        return output, self.y_test[index]
    
    def __len__(self):
        return len(self.X_test)