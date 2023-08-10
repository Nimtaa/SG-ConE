import argparse
import json
import logging
import os
import pickle
import math
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
       
    train_queries = torch.load(os.path.join(args.data_path,'train_input.pt'))
    test_queries = torch.load(os.path.join(args.data_path,'test_input.pt'))
    val_queries = torch.load(os.path.join(args.data_path,'val_input.pt'))
    train_answers = torch.load(os.path.join(args.data_path,'train_output.pt'))
    test_answers = torch.load(os.path.join(args.data_path,'test_output.pt'))
    val_answers = torch.load(os.path.join(args.data_path,'val_output.pt'))

    train_indices = torch.from_numpy(torch.load(os.path.join(args.data_path,'train_indices.pt'))).cuda()
    val_indices = torch.from_numpy(torch.load(os.path.join(args.data_path,'val_indices.pt'))).cuda()
    test_indices = torch.from_numpy(torch.load(os.path.join(args.data_path,'test_indices.pt'))).cuda()
    
    train_positive_indices = []
    train_negative_indices = []
    test_positive_indices = []
    test_negative_indices = []
    val_positive_indices = []
    val_negative_indices = []

    for idx in train_indices:
        train_positive_indices.append(math.floor(idx/2) + (args.nentity//3))
        train_negative_indices.append(math.floor(idx/2) + (args.nentity//3) - 10)

    for idx in test_indices:
        test_positive_indices.append(math.floor(idx/2) + (args.nentity//3))
        test_negative_indices.append(math.floor(idx/2) + (args.nentity//3) - 10)

    for idx in val_indices:
        val_positive_indices.append(math.floor(idx/2) + (args.nentity//3))
        val_negative_indices.append(math.floor(idx/2) + (args.nentity//3) - 10)

    train_positive_indices = torch.from_numpy(np.array(train_positive_indices)).cuda()
    train_negative_indices = torch.from_numpy(np.array(train_negative_indices)).cuda()
    test_positive_indices = torch.from_numpy(np.array(test_positive_indices)).cuda()
    test_negative_indices = torch.from_numpy(np.array(test_negative_indices)).cuda()
    val_positive_indices = torch.from_numpy(np.array(val_positive_indices)).cuda()
    val_negative_indices = torch.from_numpy(np.array(val_negative_indices)).cuda()


    return train_queries, test_queries, val_queries, train_answers, test_answers, val_answers, train_indices, val_indices, test_indices, train_positive_indices, train_negative_indices, val_positive_indices, val_negative_indices, test_positive_indices, test_negative_indices



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

    train_queries, test_queries, val_queries, train_answers, test_answers, val_answers, train_indices, val_indices, test_indices, train_positive_indices, train_negative_indices, val_positive_indices, val_negative_indices, test_positive_indices, test_negative_indices = load_data(args)

    train_dataset = TensorDataset(train_queries, train_answers, train_indices, train_positive_indices, train_negative_indices)
    test_dataset = TensorDataset(test_queries, test_answers, test_indices, test_positive_indices, test_negative_indices)
    val_dataset = TensorDataset(val_queries, val_answers, val_indices, val_positive_indices, val_negative_indices)

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

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(20):
        print(f'epoch: {epoch+1}')
        log = model.train_step(model, optimizer, train_dataloader, args)





if __name__ == '__main__':
    main(parse_args())