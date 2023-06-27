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

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, variable_len, code_ids, labels_ids, attention_mask, variable_ids, variable_attention_mask, variable_ids_before, variable_attention_mask_before):
        self.variable_len = variable_len
        
        self.code_ids = code_ids        
        self.labels_ids = labels_ids
        self.attention_mask = attention_mask
        
        self.variable_ids = variable_ids
        self.variable_attention_mask = variable_attention_mask
        
        
        self.variable_ids_before = variable_ids_before
        self.variable_attention_mask_before = variable_attention_mask_before
        

def process_data_top3(data_dir, path, prefix):
    path = os.path.join(data_dir, path)
    with open(path, 'r', encoding='utf-8') as f:
        res1 = []
        res2 = []
        res3 = []
        data = json.load(f)
        print("len of data:", len(data))
        for temp in tqdm(data):
            top1, top2, top3 = convert_examples_to_features_top3(temp)
            res1.append(top1)
            res2.append(top2)
            res3.append(top3)
        # print("len of res1:", len(res1))
        # print("len of res2:", len(res2))
        # print("len of res3:", len(res3))
        # print("=============================")
        
        output_path = os.path.join(data_dir, prefix+'1valid.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(res1, f)
        
        output_path = os.path.join(data_dir, prefix+'2valid.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(res2, f)
        
        output_path = os.path.join(data_dir, prefix+'3valid.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(res3, f)

def convert_examples_to_features_top3(ori):
    res = []
    
    for i in range(3):
        temp = ori
        # print('temp:',temp)
        mask_variable = temp['mask_variable']
        variable_tokens = tokenizer.tokenize(mask_variable)
        variable_ids = tokenizer.convert_tokens_to_ids(variable_tokens)
        masked_len = len(variable_ids)
        
        
        
        
        variable_attention_mask = [1]*masked_len + [0]*(max_code_length-masked_len)
        variable_attention_mask = torch.tensor(variable_attention_mask)
        
        var_name_before = temp['var_name_before']
        var_name_before = tokenizer.tokenize(var_name_before)
        var_name_before_ids = tokenizer.convert_tokens_to_ids(var_name_before)
        masked_len_before = len(var_name_before_ids)
        
        var_name_before_ids += [tokenizer.pad_token_id]*(max_code_length-masked_len_before)
        variable_ids_before = torch.tensor(var_name_before_ids)
        
        variable_attention_mask_before = [1]*masked_len_before + [0]*(max_code_length-masked_len_before)
        variable_attention_mask_before = torch.tensor(variable_attention_mask_before)
        
        
        predict_len = temp['top3'][i]+1
        variable_len = torch.tensor(predict_len)
        
        mask_code = temp['mask_code']
        mask_code = mask_code.replace('<mask>', '<mask>'*predict_len)
        code_tokens = tokenizer.tokenize(mask_code)
        
        
        temp = tokenizer.convert_tokens_to_ids(code_tokens)
        mask_idx = temp.index(tokenizer.mask_token_id)
        
       
            
        code_tokens = [tokenizer.cls_token]+code_tokens[:max_code_length-2]+[tokenizer.sep_token]
        code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
        
        if '<mask>' not in code_tokens:
            print('predict_len:',predict_len)
            print('mask_code:',mask_code)
            print('code_tokens:',code_tokens)
            return None
        
        temp_ids = code_ids.copy()
        
        # m = 0
        labels_ids = [-100]*max_code_length
        # while tokenizer.mask_token_id in temp_ids:
            # mask_idx = temp_ids.index(tokenizer.mask_token_id)
            # labels_ids[mask_idx : mask_idx+predict_len] = variable_ids[:predict_len]
            # temp_ids[mask_idx : mask_idx+predict_len] = variable_ids[:predict_len]
            
        
        labels_ids = torch.tensor(labels_ids)
        
        variable_ids += [tokenizer.pad_token_id]*(max_code_length-masked_len)
        variable_ids = torch.tensor(variable_ids)

        padding_length = max_code_length - len(code_ids)
        
        
        attention_mask = [1]*len(code_ids)
        attention_mask += [0]*padding_length
        attention_mask = torch.tensor(attention_mask)
        
        code_ids += [tokenizer.pad_token_id]*padding_length
        code_ids = torch.tensor(code_ids)
        
        # print('code_ids.size():',code_ids.size())
        # print('labels_ids.size():',labels_ids.size())
        assert code_ids.size()==labels_ids.size(), "=====1 and 2 Unequal！！======="
        assert code_ids.size()==attention_mask.size(), "=====1 and 3 Unequal！！======="
        assert labels_ids.size()==attention_mask.size(), "=====2 and 3 Unequal！！======="
        res.append(InputFeatures(variable_len, code_ids, labels_ids, attention_mask, variable_ids, variable_attention_mask, variable_ids_before, variable_attention_mask_before))
    
    return res


def process_data(path, output_path):
    with open(path, 'r', encoding='utf-8') as f:
        res = []
        data = json.load(f)
        print("len of data:", len(data))
        for temp in tqdm(data):
            exa = convert_examples_to_features(temp)
            if exa != None:
                res.append(exa)
        print("len of res:", len(res))
        with open(output_path, 'wb') as f:
            pickle.dump(res, f)



def convert_examples_to_features(temp):
    mask_variable = temp['mask_variable']
    variable_tokens = tokenizer.tokenize(mask_variable)
    variable_ids = tokenizer.convert_tokens_to_ids(variable_tokens)
    masked_len = len(variable_ids)
    variable_len = torch.tensor(masked_len-1)
    
    
    variable_attention_mask = [1]*masked_len + [0]*(max_code_length-masked_len)
    variable_attention_mask = torch.tensor(variable_attention_mask)
    
    var_name_before = temp['var_name_before']
    var_name_before = tokenizer.tokenize(var_name_before)
    var_name_before_ids = tokenizer.convert_tokens_to_ids(var_name_before)
    masked_len_before = len(var_name_before_ids)
    
    var_name_before_ids += [tokenizer.pad_token_id]*(max_code_length-masked_len_before)
    variable_ids_before = torch.tensor(var_name_before_ids)
    
    variable_attention_mask_before = [1]*masked_len_before + [0]*(max_code_length-masked_len_before)
    variable_attention_mask_before = torch.tensor(variable_attention_mask_before)
    
    
    
    
    mask_code = temp['mask_code']
    mask_code = mask_code.replace('<mask>', '<mask>'*masked_len)
    code_tokens = tokenizer.tokenize(mask_code)
    
    if masked_len>10:
        return None
    
    if '<mask>' not in code_tokens:
        return None
    
    temp = tokenizer.convert_tokens_to_ids(code_tokens)
    mask_idx = temp.index(tokenizer.mask_token_id)
    
    
    
    if mask_idx>=max_code_length-2:
        return None
        
    code_tokens = [tokenizer.cls_token]+code_tokens[:max_code_length-2]+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)

    temp_ids = code_ids.copy()
    
    # m = 0
    labels_ids = [-100]*max_code_length
    while tokenizer.mask_token_id in temp_ids:
        mask_idx = temp_ids.index(tokenizer.mask_token_id)
        # print("mask_idx:",mask_idx)
        labels_ids[mask_idx : mask_idx+masked_len] = variable_ids
        temp_ids[mask_idx : mask_idx+masked_len] = variable_ids
        # m = m + 1
    
    labels_ids = torch.tensor(labels_ids)
    
    variable_ids += [tokenizer.pad_token_id]*(max_code_length-masked_len)
    variable_ids = torch.tensor(variable_ids)

    padding_length = max_code_length - len(code_ids)
    
    
    attention_mask = [1]*len(code_ids)
    attention_mask += [0]*padding_length
    attention_mask = torch.tensor(attention_mask)
    
    code_ids += [tokenizer.pad_token_id]*padding_length
    code_ids = torch.tensor(code_ids)
    
    if code_ids.size()!=labels_ids.size():
        return None
    
    if code_ids.size()!=attention_mask.size():
        return None
    
    if attention_mask.size()!=labels_ids.size():
        return None
        
    assert code_ids.size()==labels_ids.size(), "=====1 and 2 Unequal！！======="
    assert code_ids.size()==attention_mask.size(), "=====1 and 3 Unequal！！======="
    assert labels_ids.size()==attention_mask.size(), "=====2 and 3 Unequal！！======="
    return InputFeatures(variable_len, code_ids, labels_ids, attention_mask, variable_ids, variable_attention_mask, variable_ids_before, variable_attention_mask_before)





if __name__ == '__main__':
    
    
    process_data('./data/javaref/final_train.json', './data/javaref/variable_train_ids.pkl')
    process_data('./data/javaref/final_test.json', './data/javaref/variable_test_ids.pkl')
    process_data('./data/javaref/final_valid.json', './data/javaref/variable_valid_ids.pkl')
    
    process_data('./data/tl-codesum/final_train.json', './data/tl-codesum/variable_train_ids.pkl')
    process_data('./data/tl-codesum/final_test.json', './data/tl-codesum/variable_test_ids.pkl')
    process_data('./data/tl-codesum/final_valid.json', './data/tl-codesum/variable_valid_ids.pkl')
    
    
