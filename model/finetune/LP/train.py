from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import os

import math
import pickle
import re 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel, RobertaForMaskedLM, AutoTokenizer
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling,AutoModel
from transformers import Trainer, TrainingArguments, pipeline, Trainer, TrainingArguments

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import pipeline, RobertaTokenizerFast
from transformers import DataCollatorForLanguageModeling
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



from modules import get_cosine_schedule_with_warmup

from dataset import InputFeatures, MLMDataset
from utils import epoch_time, split_camel, Top3, AccuarcyCompute
from model import myModel

def train(model, data_loader, optimizer, scheduler, criterion, max_name_length, device, CLIP):
    model.train()
    epoch_loss = 0
    
    for batch in data_loader:  # data_loader: 41  <class 'torch.utils.data.dataloader.DataLoader'>
        optimizer.zero_grad()
        input_data = [batch_data.to(device) for batch_data in batch]

        code_ids = input_data[0]  #[1, 150]
        labels_ids = input_data[1]
        attention_mask = input_data[2]
        
        
        logits, predicted_lengths = model(code_ids,attention_mask) 
        
        loss = criterion(predicted_lengths.view(-1, max_name_length), labels_ids.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item()
        # epoch_loss += loss.mean().item()
    
    return epoch_loss / len(data_loader)

def evaluate(model, data_loader, tokenizer, criterion, max_name_length, device):
    model.eval()
    epoch_loss = 0.0
    
    with torch.no_grad():
    
        for batch in data_loader:
            # print('test==data_loader:',len(data_loader))
            input_data = [batch_data.to(device) for batch_data in batch]
        
            code_ids = input_data[0]  #[1, 150]
            labels_ids = input_data[1]
            attention_mask = input_data[2]
            logits, predicted_lengths = model(code_ids,attention_mask) 
            loss = criterion(predicted_lengths.view(-1, max_name_length), labels_ids.view(-1))
            
            epoch_loss += loss.item()
            # epoch_loss += loss.mean().item()
           
    eval_loss = epoch_loss / len(data_loader)
    perplexity = math.exp(eval_loss)
    return eval_loss, perplexity
    

def test_cls(model, data_loader, criterion, max_name_length, tokenizer, device):
    model.eval()
    epoch_loss = 0.0
    
    acc = 0
    nums = 0
    total = 0
    top_3 = 0
    res = []
    with torch.no_grad():
    
        for batch in data_loader:
            # print('test==data_loader:',len(data_loader))
            input_data = [batch_data.to(device) for batch_data in batch]
        
            code_ids = input_data[0]    #[batch, seq_len]
            labels_ids = input_data[1]
            attention_mask = input_data[2]
            variable_ids = input_data[3]
            
            logits, predicted_lengths = model(code_ids,attention_mask) 
            loss = criterion(predicted_lengths.view(-1, max_name_length), labels_ids.view(-1))
            
            num_per_batch = len(labels_ids)
            
            max_logits, max_idx = torch.max(predicted_lengths.data,1)
            
            top_3_len_ids = torch.topk(predicted_lengths, 3, dim=1).indices
                
            
            acc += AccuarcyCompute(max_idx,labels_ids)
            top_3 += Top3(top_3_len_ids, labels_ids)
            
            total += num_per_batch
            epoch_loss += loss.item()
            
            
            for i in range(num_per_batch):
                per_variable = []
                if tokenizer.pad_token_id in variable_ids[i]:
                    pad_idx = variable_ids[i].cpu().detach().numpy().tolist().index(tokenizer.pad_token_id)
                    per_variable = variable_ids[i][:pad_idx]
                variable_token = tokenizer.decode(per_variable).strip()
                
                res.append({"variable":variable_token, "ori_length":labels_ids[i].item(),"pred_length":max_idx[i].item()})
            
            
           
    eval_loss = epoch_loss / len(data_loader)
    
    with open(f'./output/{args.model}/result/predict.json', 'w+', encoding='utf-8') as f:
        json.dump(res, f)
    
    return eval_loss, acc*1.0 / total, top_3*1.0 / total



def run(args):
    timestamp = datetime.now().strftime('%Y%m%d%H%M') 
    t1 = time.perf_counter()
    
    os.makedirs(f'./output/{args.model}/models', exist_ok=True)
    os.makedirs(f'./output/{args.model}/result', exist_ok=True)
    
    
    fh = logging.FileHandler(f"./output/{args.model}/log.txt")
    logger.addHandler(fh)
    tb_writer = SummaryWriter(f"./output/{args.model}/logs/" ) if args.visual else None

    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # import pretrained model
    
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
    vocab_size = tokenizer.vocab_size
    mlm = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
    
    
    mlm.load_state_dict(torch.load(f'../../pretrain/output/csn_min_model/models/pytorch_model.bin'))
    model = myModel(mlm, vocab_size, args.max_name_length)
    model.to(device)
    
    
    logger.info("***** Reading data *****")
    
    # import data
    train_data = MLMDataset(args.train_data, args.train_ids,  tokenizer, args)
    test_data = MLMDataset(args.test_data, args.test_ids, tokenizer, args)
    valid_data = MLMDataset(args.valid_data, args.valid_ids, tokenizer, args)
    
    train_data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    test_data_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    valid_data_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)

    logger.info("  Num examples of train_data = %d", len(train_data_loader))
    logger.info("  Num examples of test_data = %d", len(test_data_loader))
    logger.info("  Num examples of valid_data = %d", len(valid_data_loader))
    
    # model.load_state_dict(torch.load('model.pt'))
    
    
    ###############################################################################
    # Prepare the Optimizer
    ###############################################################################

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                            num_training_steps=len(train_data_loader)*args.epochs)

    criterion = nn.NLLLoss(ignore_index=-100)
    
    t2 = time.perf_counter()
    print('Init: %.2f seconds' % (t2 - t1))
    model.to(device)
    
    k = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {k} trainable parameters')
    
    ###############################################################################
    # Starting the training
    ###############################################################################
    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data_loader))
    logger.info("  Num Epochs = %d", args.epochs)
    
    N_EPOCHS = args.epochs
    best_valid_loss = float('inf')
    
    if args.do_train:
        losses = []
        
        for epoch in tqdm(range(N_EPOCHS)):
            start_time = time.time()
            
            train_loss = train(model, train_data_loader, optimizer, scheduler, criterion, args.max_name_length, device, args.max_grad_norm)
            valid_loss, perplexity = evaluate(model, test_data_loader, tokenizer, criterion, args.max_name_length, device)
            
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), f'./output/{args.model}/models/pytorch_model.bin') #/output/{args.model}/{timestamp}/models
                tokenizer.save_pretrained(f'./output/{args.model}/models/')
                torch.save(args, os.path.join(f'./output/{args.model}/models/', 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", f'./output/{args.model}/models/')
                eval_loss, precision, top_3 = test_cls(model, valid_data_loader, criterion, args.max_name_length, tokenizer, device)
                print("Valid Loss %f | precision %f | top_3 precision %f" % (eval_loss, precision, top_3))
                print("==========================")
                
            losses.append(valid_loss)
            
            # losses = torch.cat(losses)
            # avg_loss = torch.mean(losses)
            avg_loss = sum(losses)*1.0/len(losses)
            
            tb_writer.add_scalars('Loss', {'Train':train_loss}, epoch)
            tb_writer.add_scalars('Loss', {'Test':valid_loss}, epoch)
            tb_writer.add_scalars('Loss', {'avg_loss':avg_loss}, epoch)
                
            print("Epoch %d: Train Loss %f | Time: %dm %ds" % (epoch+1, train_loss, epoch_mins, epoch_secs))
            print("Epoch %d: Test Loss %f | Time: %dm %ds" % (epoch+1, valid_loss, epoch_mins, epoch_secs))
            print("Epoch %d: Average Loss %f | Time: %dm %ds" % (epoch+1, avg_loss, epoch_mins, epoch_secs))
            print("Valid Loss %f | perplexity: %f " % (valid_loss, perplexity))
            
    
    
    if args.do_eval:
        model.load_state_dict(torch.load(f'./output/{args.model}/models/pytorch_model.bin'),strict=False)
        
        eval_loss, precision, top_3 = test_cls(model, valid_data_loader, criterion, args.max_name_length, tokenizer, device)
        print("Valid Loss %f | precision %f | top_3 precision %f" % (eval_loss, precision, top_3))


def parse_args():
    parser = argparse.ArgumentParser("Train and Validate The MLM Model")
    parser.add_argument('--model', type=str, default='cls_javaref_csn_mymodel', help='model name: Seq2Seq, SelfAttnModel')
    parser.add_argument('--model_path', type=str, default="./cls_javaref_csn_mymodel", help='the path of validing data')

    parser.add_argument('--reload_from', type=int, default=-1, help='epoch to reload from')

    # parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('-v', "--visual",action="store_true", default=True, help="Visualize training status in tensorboard")

    # Training Arguments
    parser.add_argument('--log_every', type=int, default=100, help='interval to log autoencoder training results')
    parser.add_argument('--valid_every', type=int, default=10000, help='interval to validation')
    parser.add_argument('--save_every', type=int, default=50000, help='interval to evaluation to concrete results')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    # parser.add_argument('--vocab', type=str, default="./data/csn/java/vocab.json", help='the path of training data vocab')
    parser.add_argument('--max_length', type=int, default=256, help='the max length of input code')
    parser.add_argument('--max_name_length', type=int, default=5, help='the max length of input code')
    parser.add_argument('--batch_size', type=int, default=4, help='the batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='adam epsilon')
    parser.add_argument('--warmup_steps', type=int, default=2000, help='warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--do_train', type=bool, default=True, help='weight decay')
    
    parser.add_argument('--do_eval', type=bool, default=True, help='weight decay')
    
    
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='CLIP')
    parser.add_argument('--data_dir', type=str, default="./data/javared/", help='the data dir path')
    #parser.add_argument('--data_dir', type=str, default="./data/tl-codesum", help='the data dir path')


    parser.add_argument('--train_ids', type=str, default="variable_cls_train_ids.pkl", help='the path of training data')
    parser.add_argument('--test_ids', type=str, default="variable_cls_test_ids.pkl", help='the path of testing data')
    parser.add_argument('--valid_ids', type=str, default="variable_cls_valid_ids.pkl", help='the path of validing data')
    

    return parser.parse_args()
    

if __name__ == "__main__":
    
    args = parse_args()
    torch.backends.cudnn.benchmark = True # speed up training by using cudnn
    torch.backends.cudnn.deterministic = True # fix the random seed in cudnn

    run(args)
    
