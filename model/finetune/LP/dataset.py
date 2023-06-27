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
    def __init__(self, code_ids, labels_ids, attention_mask, variable_ids):
        self.labels_ids = labels_ids
        self.code_ids = code_ids        
        self.attention_mask = attention_mask
        self.variable_ids = variable_ids



class MLMDataset(data.Dataset):
    def __init__(self, data_path, args):
        data_path = os.path.join(args.data_dir, data_path)
        examples = read_file(data_path)
        logger.info("***** len of %s is %s *****", data_path, len(examples))
        max_name_length = args.max_name_length
        
        self.examples = []
        for i in range(len(examples)):
            temp = examples[i]
            if temp.labels_ids >= max_name_length:
                temp.labels_ids = torch.tensor(max_name_length-1)
            self.examples.append(temp)

        logger.info("***** len of %s is %s *****", data_path, len(self.examples))
        logger.info("***** Reading completely! *****")
    
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, item):
        # assert self.examples[item][0]==self.examples[item][1], "=====Unequal！！======="
        return self.examples[item].code_ids, self.examples[item].labels_ids, self.examples[item].attention_mask, self.examples[item].variable_ids
        # return self.examples[item][0],  self.examples[item][1], self.examples[item][2], self.examples[item][3]
        
