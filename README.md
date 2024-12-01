<p align="center">
  <img src="images/logo.jpg" alt="Figure 1" width="100%">
  <br>
</p>

# VATMAN: Video Anomaly Transformer for Monitoring Accidents and Nefariousness (author implementation)

This repository contains the official author implementation of **"VATMAN: Video Anomaly Transformer for Monitoring Accidents and Nefariousness"** (AVSS IEEE 2024).


## Experimental Setup

We provide the network architecture of the proposed VATMAN model, along with the pipeline code to enable users to train and test the network on the video feature dataset.
The Transformer component used in the asymmetric Autoencoder is based on the implementation from [lucidrains's repository](https://github.com/lucidrains/vit-pytorch). 
All experiments were conducted in an environment with `Python 3.8.12`, `PyTorch 1.12.1`, and `CUDA Toolkit 12.3`.

### Dataset and Hyperparameter Configuration

User-configurable parameters, such as paths to the training datasets and hyperparameters, are primarily defined in `parameters.py`.

- **Dataset Configuration**

To specify the dataset path, you need to edit the following parameters in `parameters.py`.
These parameters are used solely to access the dataset and name the result file.
```python
#parameters.py
data_root_dir = '[ROOT PATH OF THE DATASETS]'

feature_extractor = '[THE NAME OF THE FEATURE EXTRACTOR USED TO EXTRACT FEATURE FROM VIDEO SEGMENT]'

anomaly_class = '[ANOMALY CLASS OF THE DATASET]'
```

The directory structure of the folder containing the daaset is as follows:
```plaintext
data_root_dir/
├── feature_extractor_1/
│   ├── anomaly_class_1/
│   │   ├── video1/
│   │   │   ├── segment_feature1.npy
│   │   │   ├── segment_feature2.npy
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── ...
```
The `segment_feature` must have a vector shape of [1, `embedding_dim`]. 
The `embedding_dim` depends on the feature extractor being used. 
You can adjust it by modifying the following parameter.
```python
#parameters.py

embedding_dim = '[THE DIMENSION OF SEGMENT_FEATURE]'
```

The remaining parameters in `parameters.py` are static hyperparameters optimized for our dataset.
You can adjust them as needed for your experiments.
The batch size and learning rate are automatically tuned across all models used in the experiments using the hyperparameter tuning tool Optuna ([link](https://optuna.org/)).

- **Setting the Experimental Results Path**

After configuring the dataset and hyperparameters, you need to set the path for saving the experimental results.. 
This can be done in `parameters.py` by modifying the variables `save_root_dir` and `exp_name`. 
`results_path` specifies the default directory where the experimental results will be saved, and `exp_name` defines the name of the current experiment.

- **Run the Experiment**
  
Once you have completed the setup in `parameters.py`, you can start the experiment by running the following command:
```python
python train.py
```

### Citation
If you find this code useful for your work, please cite the following and consider starring this repository:
```bibtex
@inproceedings{kim2024vatman,
  title={VATMAN: Video Anomaly Transformer for Monitoring Accidents and Nefariousness},
  author={Kim, Harim and Lee, Chang Ha and Hong, Charmgil},
  booktitle={2024 IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS)},
  pages={1--7},
  year={2024},
  organization={IEEE}
}
```
