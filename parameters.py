
data_root_dir = '/projects/vode/harim_AVSS_feature'

save_root_dir = '/projects/vode/harim_AVSS_results'

feature_extractor = 'X3D' #I3D, P3D, S3D, TimeSformer, X3D, TimeSformer_mean

anomaly_class = 'robbery' #fight, kidnap, robbery, swoon, swoon_re, trespass, vandalism

exp_name = 'lastVAT-9-(2)'

embedding_dim = 2048

eps = 1e-7

epochs = 200


# static hyperparameters
#==============================
infer_sample_num = 10

token_sample_num = 256

heads = 32

dropout = 0.1

# output embedding dimension of the each encoder/decoder module
encoder_dim = [2048, 512, 256] 
decoder_dim = [256, 512, 2048]

# pooling option for anomaly score
infer_attention_mode = 'avg'
layer_attention_mode = 'sum'  #avg, max, sum ...
head_attention_mode = 'sum' #avg, max, sum ...
#==============================

