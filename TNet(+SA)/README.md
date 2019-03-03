# TNet-Att(+SA)

## Requirements
* Python 3.6
* Theano 0.9.0
* numpy 1.13.1
* pygpu 0.6.9
* GloVe.840B.300d

## Running
```
THEANO_FLAGS="device=gpu0" python main_total.py -ds_name [YOUR_DATASET_NAME] -log_name [YOUR_LOG_NAME]
```
