<p align="center">
  <img src="images/logo.jpg" alt="Figure 1" width="100%">
  <br>
</p>

# VATMAN: Video Anomaly Transformer for Monitoring Accidents and Nefariousness (author implementation)

This repository contains our implementation of **"VATMAN: Video Anomaly Transformer for Monitoring Accidents and Nefariousness"** (AVSS IEEE 2024).


## Experimental Setup

We provide the network architecture of the proposed VATMAN model and share the pipeline code that enables users to train and test the network on the video feature dataset. 
The Transformer part used in asymmetric Autoencoder references the implementation provided by [lucidrains's repository](https://github.com/lucidrains/vit-pytorch). 
All experiments were conducted in an environment with `Python 3.8.12`, `PyTorch 1.12.1`, and `CUDA Toolkit 12.3`.

### Dataset and Hyperparameter Configuration

The parameters that users can modify, such as the paths to the training datasets and the hyperparameters, are primarily defined in `parameters.py`.

- **Dataset Configuration**

To locate the path of the dataset, you need to edit the following parameters in `parameters.py`.
They are only used to access the dataset and name the result file.
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

To 

As an example, we share the Lung dataset [^4] used in the experiments described in the paper (datasets/lung-1vs5.csv). 
All datasets used in the experiments must be min-max normalized per feature, with the last feature serving as a binary label distinguishing between normal and abnormal cases. 
The path to the dataset, saved as a `.csv` file, can be specified as follows:

```python
# parameters.py
dataset_root = '[PATH OF dataset.csv FILE]'
```

- **Hyperparameter Configuration**

The default hyperparameters for the model have been optimized based on the experiments conducted with the Lung dataset as described in the paper. 
These hyperparameters can also be adjusted in `parameters.py` as shown below:

```python
# parameters.py
hp = {
    'batch_size' : [BATCH SIZE],
    'lr' : [LEARNING RATE],
    'sequence_len' : [SEQUENCE LENGTH],
    'heads' : [NUMBER OF HEADS],
    'dim' : [ENCODER'S INPUT DIMENSION],
    'num_layers' : [NUMBER OF LAYERS],
    'layer_conf' : [LAYER CONFIGURATION: {'same', 'smaller', 'hybrid'} OPTIONS ARE AVAILABLE] 
}
```

- **Setting the Experimental Results Path**

After configuring the dataset and hyperparameters, you will need to set the path for saving the experimental results. 
This can also be done in `parameters.py` using the variables `results_path` and `exp_name`. 
`results_path` specifies the default directory where the experimental results will be saved, and `exp_name` defines the name of the current experiment.

For example, with the following configuration:
```python
# parameters.py
results_path = './results'

exp_name = 'test'
```
The best-performing model trained during the experiment will be saved as `results/test/best_auroc_model.pt`.

- **Run the Experiment**
  
If you have completely finished setting up `parameters.py`, you can start the experiment by running the following command:
```
python main.py
```

### Citation
If you utilize this code, please cite the following paper and star this repository:
```bibtex
@inproceedings{kim2024transpad,
  title={Transformer for Point Anomaly Detection},
  author={Kim, Harim and Lee, Chang Ha and Hong, Charmgil},
  booktitle={Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
  year={2024},
}
```

### References

[^1]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.
[^2]: Hu Wang, Guansong Pang, Chunhua Shen, and Congbo Ma. 2019. Unsupervised representation learning by predicting random distances. arXiv preprint arXiv:1912.12186.
[^3]: Leland McInnes, John Healy, and James Melville. 1802. Umap: uniform manifold approximation and projection for dimension reduction. arxiv 2018. arXiv preprint arXiv:1802.03426.
[^4]: Z.Q. Hong and J.Y. Yang. 1992. Lung cancer. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C57596. (1992).
