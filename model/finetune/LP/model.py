
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
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
import Levenshtein
from info_nce import InfoNCE
import json
import torch
import argparse
import torch.nn as nn
import  gc
import torch.utils.data as data
from tqdm import tqdm
# torch.multiprocessing.set_start_method('spawn')

from torch.nn import functional as F
from torch.autograd import Variable
from collections import Counter
gc.collect()
torch.cuda.empty_cache()
torch.cuda.is_available()
import logging
# logger.info
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")




class myModel(nn.Module):
    def __init__(self, funetune, vocab_size, max_name_length):
        super().__init__()
        self.mlm = funetune
        self.linear = nn.Linear(vocab_size, max_name_length)
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, code_ids, attention_mask):
        
        outputs = self.mlm(input_ids=code_ids,attention_mask=attention_mask) 
        
        logits = outputs.logits
        
        cls_logits = logits[:, 0, :]  # [batch, vocab_size]
        # print('cls_logits:',cls_logits.shape)
        predicted_lengths = self.softmax(self.linear(cls_logits))
        # print('predicted_lengths_logits:',predicted_lengths.shape)
        
        return logits, predicted_lengths
        