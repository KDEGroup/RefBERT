from pathlib import Path
import os

import math
import pickle
import re 
import random
import numpy as np
import time
from datetime import datetime
from tensorboardX import SummaryWriter
import fastwer

import json
import torch
import torch.nn as nn


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



def split_camel(camel_str,test=''):
    try:
        split_str = re.sub(
            r'(?<=[a-z]|[0-9])(?=[A-Z])|_|[0-9]|(?<=[A-Z])(?=[A-Z][a-z])|\s+',
            '_',
            camel_str)
    except TypeError:
        return ['']
    try:
        if split_str[0] == '_':
            return [camel_str]
    except IndexError:
        return []
   
    return [i for i in re.split('\_|\.',split_str) if i != '']
    

def Accuracy(gold_set_parts, predicted_parts):
    """
    Get the accuracy for the given token.
    :param predicted_parts: a list of predicted parts
    :param gold_set_parts: a list of the golden parts
    :return: accuracy as floats
    """
    
    ground = split_camel(gold_set_parts)
    prediction = split_camel(predicted_parts)
    
    new_ground = ground.copy()
    # ground = [tok.lower() for tok in gold_set_parts]
    # prediction = list(predicted_parts)
    
    
    tp = 0
    for subtoken in prediction:
        if subtoken in new_ground:
            new_ground.remove(subtoken)
            tp += 1

    
    precision = float(tp) / len(prediction)
    
    return precision
    
def calculate_em(trgs, preds):
    trg_tokens = split_camel(trgs)
    pred_tokens = split_camel(preds) 
    
    trg_tokens = sorted(trg_tokens)
    pred_tokens = sorted(pred_tokens)
    
    if trg_tokens == pred_tokens:
        return 1
    else:
        return 0
    
    
