import argparse
import json
import logging
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from models import KGReasoning



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

    train_indices = torch.from_numpy(torch.load(os.path.join(args.data_path,'train_indices.pt'))).cuda()
    val_indices = torch.from_numpy(torch.load(os.path.join(args.data_path,'val_indices.pt'))).cuda()
    test_indices = torch.from_numpy(torch.load(os.path.join(args.data_path,'test_indices.pt'))).cuda()
    
    

    return train_queries, test_queries, val_queries, train_answers, test_answers, val_answers, train_indices, val_indices, test_indices, valid_queries_mask



def main(args):

    with open('%s/stats.txt' % args.data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])

    args.nentity = nentity
    args.nrelation = nrelation

    args.hidden_dim = 256
    args.gamma = 12
    args.center_reg = 0.02
    args.cuda = True
    args.test_batch_size = 1
    args.drop = 0.05

    train_queries, test_queries, val_queries, train_answers, test_answers, val_answers, train_indices, val_indices, test_indices, valid_queries_mask = load_data(args)

    train_dataset = TensorDataset(train_queries, train_answers, train_indices)
    test_dataset = TensorDataset(test_queries, test_answers, test_indices)
    val_dataset = TensorDataset(val_queries, val_answers, val_indices)

    

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    
    model = KGReasoning(
    nentity=nentity,
    nrelation=nrelation,
    hidden_dim=args.hidden_dim,
    gamma=args.gamma,
    use_cuda=args.cuda,
    center_reg=args.center_reg,
    test_batch_size=args.test_batch_size,
    drop=args.drop
    )

    if args.cuda:
        model = model.cuda()

 
    current_learning_rate = 0.0001
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=current_learning_rate
    )

    model.train_step(model, optimizer, train_dataloader, args)




if __name__ == '__main__':
    main(parse_args())