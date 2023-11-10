import argparse
import json
import logging
import os
import pickle
import math
from collections import defaultdict

import numpy as np
import random
import torch
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

from models import KGReasoning



def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing ConE',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--do_train', action='store_true', help="do train")
    parser.add_argument('--do_valid', action='store_true', help="do valid")
    parser.add_argument('--do_test', action='store_true', help="do test")

    parser.add_argument('--data_path', type=str, default='./data/SG', help="KG data path")
    # parser.add_argument('--data_path', type=str, default='./data/simple-SG', help="KG data path")

    parser.add_argument('--batch_size', default=512)

    parser.add_argument('--save_path', default='saved_models')
   
    return parser.parse_args(args)



def load_data(args):
    
    train_positive_indices = [] 
    train_negative_indices = [] 
    test_positive_indices = [] 
    val_positive_indices = [] 
       
    train_queries = torch.load(os.path.join(args.data_path,'train_input.pt'))
    test_queries = torch.load(os.path.join(args.data_path,'test_input.pt'))
    val_queries = torch.load(os.path.join(args.data_path,'val_input.pt'))
    train_answers = torch.load(os.path.join(args.data_path,'train_output.pt'))
    test_answers = torch.load(os.path.join(args.data_path,'test_output.pt'))
    val_answers = torch.load(os.path.join(args.data_path,'val_output.pt'))
   
    
    train_indices = torch.from_numpy(np.load(os.path.join(args.data_path,'train_indices.npy'))).cuda()
    val_indices = torch.from_numpy(np.load(os.path.join(args.data_path,'val_indices.npy'))).cuda()
    test_indices = torch.from_numpy(np.load(os.path.join(args.data_path,'test_indices.npy'))).cuda()
    train_positive_indices  = torch.from_numpy(np.load(os.path.join(args.data_path, 'train_positive_indices.npy'))).cuda()
    train_negative_indices = torch.from_numpy(np.load(os.path.join(args.data_path, 'train_negative_indices.npy'))).cuda()
    test_positive_indices = torch.from_numpy(np.load(os.path.join(args.data_path, 'test_positive_indices.npy'))).cuda()
    test_negative_indices = torch.from_numpy(np.load(os.path.join(args.data_path, 'test_negative_indices.npy'))).cuda()
    val_positive_indices = torch.from_numpy(np.load(os.path.join(args.data_path, 'val_positive_indices.npy'))).cuda()
    val_negative_indices = torch.from_numpy(np.load(os.path.join(args.data_path, 'val_negative_indices.npy'))).cuda()

    entity_embs = torch.load(os.path.join(args.data_path, 'all_embs.pt'))

    return entity_embs, train_queries, test_queries, val_queries, train_answers, test_answers, val_answers, train_indices, val_indices, test_indices, train_positive_indices, train_negative_indices, val_positive_indices, val_negative_indices, test_positive_indices, test_negative_indices


# def load_data(args):
#     train_queries = torch.from_numpy(np.load(os.path.join(args.data_path, 'queries.npy'))).cuda()
#     pos_indices = torch.from_numpy(np.load(os.path.join(args.data_path, 'answers.npy'))).cuda()
#     neg_indices = torch.from_numpy(np.load(os.path.join(args.data_path, 'neg.npy'))).cuda()
    
#     return train_queries, pos_indices, neg_indices

def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )

def main(args):

    with open('%s/stats.txt' % args.data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])

    args.nentity = nentity
    args.nrelation = nrelation

    args.hidden_dim = 768
    args.gamma = 12
    args.center_reg = 0.02
    args.cuda = True
    args.test_batch_size = 1
    args.drop = 0.05

    entity_embs, train_queries, test_queries, val_queries, train_answers, test_answers, val_answers, train_indices,\
    val_indices, test_indices, train_positive_indices, train_negative_indices, val_positive_indices, val_negative_indices,\
    test_positive_indices, test_negative_indices = load_data(args)
    
    # train_queries, train_positive_indices, train_negative_indices = load_data(args)
    # train_dataset = TensorDataset(train_queries, train_positive_indices, train_negative_indices)
    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    # test_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    
    train_dataset = TensorDataset(train_queries, train_answers, train_indices, train_positive_indices, train_negative_indices)
    test_dataset = TensorDataset(test_queries, test_answers, test_indices, test_positive_indices, test_negative_indices)
    val_dataset = TensorDataset(val_queries, val_answers, val_indices, val_positive_indices, val_negative_indices)
    

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    
    model = KGReasoning(
    nentity=nentity,
    nrelation=nrelation,
    entity_embs = entity_embs,
    hidden_dim=args.hidden_dim,
    gamma=args.gamma,
    use_cuda=args.cuda,
    center_reg=args.center_reg,
    test_batch_size=args.test_batch_size,
    drop=args.drop
    )

    if args.cuda:
        model = model.cuda()

 
    current_learning_rate = 0.00001 
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=current_learning_rate
    )

    torch.autograd.set_detect_anomaly(True)

    
    # checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint'))
    # model.load_state_dict(checkpoint['model_state_dict'])
    min_loss = 10
    # for epoch in range(400):
    #     # if epoch == 40:
    #         # current_learning_rate /= 5
    #     #     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=current_learning_rate)
    
    #     print(f'epoch: {epoch+1}')
    #     log = model.train_step(model, optimizer, train_dataloader, args)
    #     print(log)
    #     if epoch %5 == 0:
            
    #         save_variable_list = {
    #                 'current_learning_rate': current_learning_rate,
    #             }
    #         if log['loss'] < min_loss:
    #             min_loss = log['loss']
    #             print('saved')
                
    #             save_model(model, optimizer, save_variable_list, args)

    # save_variable_list = {
    #                 'current_learning_rate': current_learning_rate,
    #             }
    
    # save_model(model, optimizer, save_variable_list, args)

    checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint'))
    model.load_state_dict(checkpoint['model_state_dict'])

    metrics = model.test_step(model, test_dataloader, train_indices)


if __name__ == '__main__':
    main(parse_args())