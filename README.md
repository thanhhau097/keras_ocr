# What do we need to do to train a model now?
- Change `data_root` in `configs/config.json` file
- In `train.py`, we need to change 2 lines:
    - config.vocab_type = 'xxx' (xxx in ['ctc', 'attention', 'joint'])
    - model = XxxModel(config)
   
### Data sample
- Please take a look at `data/val.json` file, the data in this file is a dictionary with following structure:

```
{    
    path1: label1, 
    path2: label2,
    ...
    pathn: labeln
}
```


# How to use SageMaker?
- Run `run_sagemaker.ipynb` file with jupyter notebook on SageMaker.  