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


def Top3(top_3, label):
    flag = 0
    for i, per_mask in enumerate(top_3):
        if label[i] in per_mask:
            flag += 1
    
    return flag
        



def AccuarcyCompute(pred,label):
	pred = pred.cpu()
	label = label.cpu()
	
	correct = (pred == label).sum().item()
	# print('correct:',correct)
	# test_np = (np.argmax(pred,1) == label)
	# test_np = np.float32(test_np)
	# return np.mean(test_np)
	return correct
