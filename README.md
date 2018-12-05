# README

# Requirement

## general environment

* python 3.5
* CUDA 9.2 (optional, if set USE_GPU in config.py equal to True, you need a set up a complete cuda environment for tensorflow)


## python package requirement
* tensorflow 1.10.1
* xgboost 0.80
* scikit-learn 0.19.1
* pandas  0.23.4
* numpy  1.15.1

# Preprocess & Feature Enginnering
The codes of data pre-process and feature enginnering are in preprocess floder.
The result of feature engineering is also in Project5002.ipynb. Please move it to the floder containing original csv data files.

Before running main.py, you should run preprocess.py in preprocess firstly. Please copy original csv data files to the current floder (same as where the main.py is).




# Run Steps
1. Edit the `DATA_DIR` and `GEN_DATA_DIR` in `config.py`. 

2. (optional) Edit the other parameters in `config.py`

3. `python main.py`


# Notice
Detailed code explanation please refer to the code comments. 

# Github Repo
https://github.com/fan31415/Attention-LSTM-on-Air-Quality-Prediction

