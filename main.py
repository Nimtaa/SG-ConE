import argparse
import json
import logging
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader



def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing ConE',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--do_train', action='store_true', help="do train")
    parser.add_argument('--do_valid', action='store_true', help="do valid")
    parser.add_argument('--do_test', action='store_true', help="do test")

    parser.add_argument('--data_path', type=str, default=None, help="KG data path")

    parser.add_argument('--batch_size', default=32)
   
    return parser.parse_args(args)



def load_data(args):
    
    with open(os.path.join(args.data_path, 'interesting_queries_mask.pkl'), 'rb') as f:
        valid_queries_mask = pickle.load(f)
    
    train_queries = torch.load(os.path.join(args.data_path,'train_input.pt'))
    test_queries = torch.load(os.path.join(args.data_path,'test_input.pt'))
    val_queries = torch.load(os.path.join(args.data_path,'val_input.pt'))
    train_answers = torch.load(os.path.join(args.data_path,'train_output.pt'))
    test_answers = torch.load(os.path.join(args.data_path,'test_output.pt'))
    val_answers = torch.load(os.path.join(args.data_path,'val_output.pt'))
    
    
    return train_queries, test_queries, val_queries, train_answers, test_answers, val_answers, valid_queries_mask



def main(args):
    train_queries, test_queries, val_queries, train_answers, test_answers, val_answers, valid_queries_mask = load_data(args)

    train_dataset = TensorDataset(train_queries, train_answers)
    test_dataset = TensorDataset(test_queries, test_answers)
    val_dataset = TensorDataset(val_queries, val_answers)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    
    



if __name__ == '__main__':
    main(parse_args())