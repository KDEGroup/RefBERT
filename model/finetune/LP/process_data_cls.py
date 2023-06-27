import json
from tqdm import tqdm
import re
import numpy as np
import pickle

from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import os
import random
import time
from datetime import datetime
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel, RobertaForMaskedLM, AutoTokenizer
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling,AutoModel
from transformers import Trainer, TrainingArguments, pipeline, Trainer, TrainingArguments

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import pipeline, RobertaTokenizerFast
from transformers import DataCollatorForLanguageModeling
import  gc
import torch.utils.data as data


from nltk.tokenize import word_tokenize, MWETokenizer

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")

max_code_length = 512
def read_file(data_path):
    with open(data_path, 'r',encoding='utf-8') as f:
        data = json.load(f)
    return data
    
class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, code_ids, labels_ids, attention_mask, variable_ids):
        self.labels_ids = labels_ids
        self.code_ids = code_ids        
        self.attention_mask = attention_mask
        self.variable_ids = variable_ids

def convert_examples_to_features_cls(temp):
    # try:
    mask_variable = temp['mask_variable']
    variable_tokens = tokenizer.tokenize(mask_variable)
    variable_ids = tokenizer.convert_tokens_to_ids(variable_tokens)
    masked_len = len(variable_ids)
    
    mask_code = temp['mask_code']
    code_tokens = tokenizer.tokenize(mask_code)
    code_tokens = [tokenizer.cls_token]+code_tokens[:max_code_length-2]+[tokenizer.sep_token]
    
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    
    if tokenizer.mask_token_id not in code_ids:
        return None
        
    
    labels_ids = torch.tensor(masked_len-1)
    
    variable_ids += [tokenizer.pad_token_id]*(max_code_length-masked_len)
    variable_ids = torch.tensor(variable_ids)

    padding_length = max_code_length - len(code_ids)
    attention_mask = [1]*len(code_ids)
    attention_mask += [0]*padding_length
    attention_mask = torch.tensor(attention_mask)
    
    code_ids += [tokenizer.pad_token_id]*padding_length
    code_ids = torch.tensor(code_ids)
    
    assert code_ids.size()==attention_mask.size(), "=====1 and 3 Unequal！！======="
    return InputFeatures(code_ids, labels_ids, attention_mask, variable_ids)


 
def process_data(path, output_path):
    with open(path, 'r', encoding='utf-8') as f:
        res = []
        data = json.load(f)
        for temp in tqdm(data):
            exa = convert_examples_to_features_cls(temp)
            if exa != None:
                res.append(exa)
        print("len of data:", len(res))
        with open(output_path, 'wb') as f:
            pickle.dump(res, f)

if __name__ == '__main__':
    
    process_data('./data/javaref/final_train.json', './data/javaref/variable_cls_train_ids.pkl')
    process_data('./data/javaref/final_test.json', './data/javaref/variable_cls_test_ids.pkl')
    process_data('./data/javaref/final_valid.json', './data/javaref/variable_cls_valid_ids.pkl')
    
    
    
    process_data('./data/tl-codesum/final_train.json', './data/tl-codesum/variable_cls_train_ids.pkl')
    process_data('./data/tl-codesum/final_test.json', './data/tl-codesum/variable_cls_test_ids.pkl')
    process_data('./data/tl-codesum/final_valid.json', './data/tl-codesum/variable_cls_valid_ids.pkl')
    
