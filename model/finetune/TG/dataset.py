import json
from tqdm import tqdm
import re
import numpy as np
import pickle
import sys
import os
import torch 
import torch.utils.data as data


    
def read_file(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data
    
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
        

    
class MLMDataset(data.Dataset):
    def __init__(self, original_data_path, data_path, tokenizer, args):
        data_path = os.path.join(args.data_dir, data_path)
        self.examples = read_file(data_path)
        
        logger.info("***** len of %s is %s *****", data_path, len(self.examples))
        logger.info("***** Reading completely! *****")
    
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, item):
        
        return self.examples[item].variable_len,self.examples[item].code_ids, self.examples[item].labels_ids, self.examples[item].attention_mask, self.examples[item].variable_ids, self.examples[item].variable_attention_mask, self.examples[item].variable_ids_before, self.examples[item].variable_attention_mask_before
        # return self.examples[item][0],  self.examples[item][1], self.examples[item][2], self.examples[item][3]
        
