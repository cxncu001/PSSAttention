# MN(+AS)

## Requirements
* Python 2.7.0 or higher
* TensorFlow 1.6.0 or higher
* GloVe.840B.300d

## Set envirnment variables to enable the GPU support
```
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/PATH/TO/MN(+AS):$PYTHONPATH
```

## Data
'''
Use MN(+AS)/scripts/build_vocab.py to generate Vocab
```
python MN(+AS)/scripts/build_vocab.py /PATH/TO/DATA  /PATH/TO/Vocab 
```

## Running

1. Use MN(+AS)/bin/train.py to train baseline models for futher erasing and mining.
```
python MN(+AS)/bin/train.py \
       --input /PATH/TO/TRAIN_SET_i/context /PATH/TO/TRAIN_SET_i/aspect /PATH/TO/TRAIN_SET_i/polarity  \
       --validation /PATH/TO/VAL_SET|TEST_SET/context /PATH/TO/VAL_SET|TEST_SET/aspect /PATH/TO/VAL_SET|TEST_SET/polarity  \
       --vocabulary /PATH/TO/Vocab \
       --model BL_MN \
       --parameters=device_list=[0],train_steps=20000,hops=1 \
       --pretrained_embedding /PATH/TO/PRETRAINED_EMBEDDING \
       --output /PATH/TO/TRAINED_MODEL_i
```
2. Use MN(+AS)/bin/predicter.py to get the attention weights of training instances.
```
python MN(+AS)/bin/predicter.py \
    --input /PATH/TO/TRAIN_SET_i/context /PATH/TO/TRAIN_SET_i/aspect /PATH/TO/TRAIN_SET_i/polarity \
    --vocabulary /PATH/TO/Vocab \
    --models BL_MN \
    --checkpoints /PATH/TO/TRAINED_MODEL_i \
    --parameters=predict_batch_size=32,device_list=[0],hops=1 \
    --output /PATH/TO/LOG
```
3. Use MN(+AS)/scripts/erasing_data.py to erase data to get TRAIN_SET_(i+1).
```
python MN(+AS)/scripts/erasing_data.py
```
4. Back to step 1, until i==5.

5. Use MN(+AS)/bin/final_train.py to train final model.
```
python SEMEVAL/thumt/bin/final_train.py \
    --input /PATH/TO/TRAIN_SET/context /PATH/TO/TRAIN_SET/aspect /PATH/TO/TRAIN_SET/polarity /PATH/TO/TRAIN_SET/AS
    _value /PATH/TO/TRAIN_SET/polarity/AS_mask \
    --validation /PATH/TO/VAL_SET|TEST_SET/context /PATH/TO/VAL_SET|TEST_SET/aspect /PATH/TO/VAL_SET|TEST_SET/polarity \
    --vocabulary /PATH/TO/Vocab \
    --model FINAL_BL_MN \
    --parameters=device_list=[0],train_steps=10000,hops=1 \
    --pretrained_embedding /PATH/TO/PRETRAINED_EMBEDDING \
    --output /PATH/TO/TRAINED_MODEL_i
```
