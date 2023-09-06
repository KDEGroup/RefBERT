# RefBERT: **A Two-Stage Pre-trained Framework for Automatic Rename Refactoring**

The source code and datasets for ISSTA 2023 paper: RefBERT: A Two-Stage Pre-trained Framework for Automatic Rename Refactoring [[arXiv preprint]](https://arxiv.org/abs/2305.17708).


## Folder
- `model` folder contains `pretrain` and `finetune` two folders, which include source code for the two stages described in our paper.
- `result` folder contains several figures reported in the paper (submission version).

## Environment

The required environment is included in `environments.txt`.

## Data
The datasets used in experiments can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1aw2yiUTXwB3gJrDcFWeDpYvGgJNYjt51) or [阿里云盘](https://www.aliyundrive.com/s/5em3yznNQgT). 

## How to run

Our model has two stages: pretraining and finetuning.

First,  pre-train the model:
```
python ./model/pretrain/train.py
```

Then, fine-tune the model on the LP task and the TG task:
```
python ./model/finetune/LP/train.py
python ./model/finetune/TG/train.py
```

Note that, since the number of masked tokens in each function differs in finetuning, we set batch size to 1 in the finetuning stage.
The parameters `args.do_train` and `args.do_eval` control whether the program is training the model or evaluating the model. 
More details can be found in the paper.




