import collections
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math
from collections import defaultdict
import pickle

pi = 3.14159265358979323846

def convert_to_arg(x):
    y = torch.tanh(2 * x) * pi / 2 + pi / 2
    return y


def convert_to_axis(x):
    y = torch.tanh(x) * pi
    return y


class AngleScale:
    def __init__(self, embedding_range):
        self.embedding_range = embedding_range

    def __call__(self, axis_embedding, scale=None):
        if scale is None:
            scale = pi
        return axis_embedding / self.embedding_range * scale


class KGReasoning(nn.Module):
    def __init__(self, nentity, nrelation, entity_embs, hidden_dim, gamma, test_batch_size=1, use_cuda=False, query_name_dict=None, center_reg=None, drop=0.):
        super(KGReasoning, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.entity_embs = entity_embs
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.use_cuda = use_cuda
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1).cuda() if self.use_cuda else torch.arange(
            nentity).to(torch.float).repeat(test_batch_size, 1) 
        self.query_name_dict = query_name_dict

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        self.cen = center_reg

        # entity only have axis but no arg
        # self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim),
        #                                      requires_grad=True)  # axis for entities
        
        # self.entity_embedding = nn.Parameter(torch.clamp(entity_embs, -self.embedding_range.item(), 
        #                                                  self.embedding_range.item()), requires_grad=True)
        
        
        #TODO I changed to entity_embedding req grad False from true
        self.entity_embedding = nn.Parameter(entity_embs, requires_grad = False)
        print(self.entity_embedding.shape)

        self.angle_scale = AngleScale(self.embedding_range.item())  # scale axis embeddings to [-pi, pi]

        self.modulus = nn.Parameter(torch.Tensor([0.5 * self.embedding_range.item()]), requires_grad=True)

        self.axis_scale = 1.0
        self.arg_scale = 1.0

        # nn.init.uniform_(
        #     tensor=self.entity_embedding,
        #     a=-self.embedding_range.item(),
        #     b=self.embedding_range.item()
        # )

        self.axis_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim), requires_grad=True)
        nn.init.uniform_(
            tensor=self.axis_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.arg_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim), requires_grad=True)
        nn.init.uniform_(
            tensor=self.arg_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.cone_proj = ConeProjection(self.entity_dim, 1600, 2)
        # self.cone_intersection = ConeIntersection(self.entity_dim, drop)
        # self.cone_negation = ConeNegation()


    def train_step(self, model, optimizer, train_iterator, args):
        model.train()
        optimizer.zero_grad()
        # for train_queries, train_positive, train_negative in train_iterator:
        for train_queries, train_answers, train_indices, train_positive, train_negative in train_iterator:
            
            positive_logit, negative_logit, _ = model(train_answers, train_answers, train_queries, train_indices, train_positive, train_negative)
            # positive_logit, negative_logit, _ = model(train_queries, train_queries, train_queries, train_queries, train_positive, train_negative)
   
            subsampling_weight = 1

            negative_score = F.logsigmoid(-negative_logit).mean(dim=1).mean()
            positive_score = F.logsigmoid(positive_logit).squeeze(dim=1).mean()
            # positive_sample_loss = - (subsampling_weight * positive_score).sum()
            # negative_sample_loss = - (subsampling_weight * negative_score).sum()
            # positive_sample_loss /= subsampling_weight.sum()
            # negative_sample_loss /= subsampling_weight.sum()
            loss = ((-negative_score) + (-positive_score)) / 2
            
            # loss = (positive_sample_loss + negative_sample_loss) / 2
            loss.backward()
            optimizer.step()
            log = {
                'positive_sample_loss': -positive_score.item(),
                'negative_sample_loss': -negative_score.item(),
                'loss': loss.item(),
            }
        return log

    def test_step(self, model, test_iterator, train_indices):
        
        model.eval()
        eval_result = collections.defaultdict(list)
        pos_logit = [] 
        neg_logit = []
        answer_idx = [] 
        i = 0
        top5_answer = defaultdict(list)
        with torch.no_grad():
            # for test_queries, test_positive, test_negative in test_iterator:
            for test_queries, test_answers, test_indices, test_positive, _ in (test_iterator):
                test_negative = torch.LongTensor(range(self.nentity)).cuda()
                # test_negative = torch.LongTensor(range(118158,177237)).cuda()
                
                if i>100:
                    break
                
                tt = torch.IntTensor([i]).cuda()
                
                positive_logit, negative_logit, _ = model(None, test_queries, test_queries, test_indices, None, test_negative)
            
                # positive_sample, negative_sample, batch_queries, query_indices, positive_indices, negative_indices): 
                
                # pos_logit.append(positive_logit.item())
                # print(negative_logit.shape)
                neg_logit.append(negative_logit)
                
                
                # answer_idx.append(torch.argsort(negative_logit, dim=1, descending=True).tolist()[0].index(math.floor(test_indices/2) + (177237//3)*2))
                
                # idx = list(range(118158, 177237)).index(math.floor(test_indices/2) + (177237//3)*2)
                # answer_idx.append(torch.argsort(negative_logit, dim=1, descending=True).tolist()[0].index(idx))
                
                
                # idx = math.floor(tt.item()/2) + (177237//3)*2
                idx = math.floor(test_indices.item()/2) + (177237//3)*2
                answer_idx.append(torch.argsort(negative_logit, dim=1, descending=True).tolist()[0].index(idx))
                
                # print(torch.argsort(negative_logit, dim=1, descending=True).tolist()[0].index(idx))
                print(torch.argsort(negative_logit, dim=1, descending=True).tolist()[0].index(idx))
                # print(np.mean(np.array(answer_idx)))
                top5_answer[test_indices.item()].extend([torch.argsort(negative_logit, dim=1, descending=True).cpu().tolist()[0][:10]])
                
                # print(train_indices[i])
                # print(test_indices)
                # print(torch.argsort(negative_logit, dim=1, descending=True)[0, :10])
                i +=1
                
                # print('###gt###', test_positive[:10])s
                # neg_logit.append(negative_logit.mean().item())
                #  positive_sample, negative_sample, batch_queries, query_indices, positive_indices, negative_indices):
                
                # neg_argsort = torch.argsort(negative_logit, dim=1, descending=True)
                # neg_ranking = neg_argsort.clone().to(torch.float)
                # pos_argsort = torch.argsort(positive_logit, dim=1, descending=True)
                # pos_ranking = pos_argsort.clone().to(torch.float)

        # print(np.mean(pos_logit), 'pos logit')
        # print(np.mean(neg_logit), 'neg logit')
        
        # print(np.mean(np.array(answer_idx)))
        with open('top5_answer.pkl', 'wb') as f:
            pickle.dump(top5_answer, f)
        return None


    def embed_query_cone(self, queries, query_structure, idx):

        axis_entity_embedding = torch.index_select(self.entity_embedding, dim=0, index=idx)
        

        axis_entity_embedding = self.angle_scale(axis_entity_embedding, self.axis_scale)
        axis_entity_embedding = convert_to_axis(axis_entity_embedding)

        if self.use_cuda:
            arg_entity_embedding = torch.zeros_like(axis_entity_embedding).cuda()
        else:
            arg_entity_embedding = torch.zeros_like(axis_entity_embedding)

        axis_embedding = axis_entity_embedding
        arg_embedding = arg_entity_embedding
        
        axis_r_embedding = self.axis_embedding[0]
        arg_r_embedding = self.arg_embedding[0]

        # axis_r_embedding = torch.index_select(self.axis_embedding, dim=0, index=torch.from_numpy(np.array([0])))
        # arg_r_embedding = torch.index_select(self.arg_embedding, dim=0, index=torch.from_numpy(np.array([0])))

        axis_r_embedding = self.angle_scale(axis_r_embedding, self.axis_scale)
        arg_r_embedding = self.angle_scale(arg_r_embedding, self.arg_scale)

        axis_r_embedding = convert_to_axis(axis_r_embedding)
        arg_r_embedding = convert_to_axis(arg_r_embedding)
        
        axis_embedding, arg_embedding = self.cone_proj(axis_embedding, arg_embedding, axis_r_embedding, arg_r_embedding)
        
        return axis_embedding, arg_embedding, idx
    

    # implement distance function
    def cal_logit_cone(self, entity_embedding, query_axis_embedding, query_arg_embedding):
        delta1 = entity_embedding - (query_axis_embedding - query_arg_embedding)
        delta2 = entity_embedding - (query_axis_embedding + query_arg_embedding)

        distance2axis = torch.abs(torch.sin((entity_embedding - query_axis_embedding) / 2))
        distance_base = torch.abs(torch.sin(query_arg_embedding / 2))

        indicator_in = distance2axis < distance_base
        distance_out = torch.min(torch.abs(torch.sin(delta1 / 2)), torch.abs(torch.sin(delta2 / 2)))
        distance_out[indicator_in] = 0.

        distance_in = torch.min(distance2axis, distance_base)

        distance = torch.norm(distance_out, p=1, dim=-1) + self.cen * torch.norm(distance_in, p=1, dim=-1)
        logit = self.gamma - distance * self.modulus

        return logit

    # implement formatting forward method
    def forward(self, positive_sample, negative_sample, batch_queries, query_indices, positive_indices, negative_indices):
        
        all_idxs, all_axis_embeddings, all_arg_embeddings = [], [], []

        # for query, idx in zip(batch_queries, query_indices):
        
        axis_embedding, arg_embedding, _ = self.embed_query_cone(batch_queries, [], query_indices)
        all_axis_embeddings.append(axis_embedding)
        all_arg_embeddings.append(arg_embedding)

            
        if len(all_axis_embeddings) > 0:
            all_axis_embeddings = torch.cat(all_axis_embeddings, dim=0).unsqueeze(1)
            all_arg_embeddings = torch.cat(all_arg_embeddings, dim=0).unsqueeze(1)
        
       
        # if type(subsampling_weight) != type(None):
        #     subsampling_weight = subsampling_weight[all_idxs]
        

        if type(positive_sample) != type(None):
            if len(all_axis_embeddings) > 0:
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_indices).unsqueeze(1)
    	        
                positive_embedding = self.angle_scale(positive_embedding, self.axis_scale)
                positive_embedding = convert_to_axis(positive_embedding)

                positive_logit = self.cal_logit_cone(positive_embedding, all_axis_embeddings, all_arg_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            positive_logit = torch.cat([positive_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            
            if len(all_axis_embeddings) > 0:
                
                # negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample.shape
                # batch_size, negative_size = negative_indices.shape
                # negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_indices.view(-1)).view(batch_size, 128, -1)
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_indices.view(-1))
                 

                # negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                negative_embedding = self.angle_scale(negative_embedding, self.axis_scale)
                negative_embedding = convert_to_axis(negative_embedding)

                negative_logit = self.cal_logit_cone(negative_embedding, all_axis_embeddings, all_arg_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            negative_logit = torch.cat([negative_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, all_idxs



class ConeProjection(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers):
        super(ConeProjection, self).__init__()
        self.entity_dim = dim
        self.relation_dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim)  
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim + self.relation_dim)  
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)

    def forward(self, source_embedding_axis, source_embedding_arg, r_embedding_axis, r_embedding_arg):
        
        x = torch.cat([source_embedding_axis + r_embedding_axis, source_embedding_arg + r_embedding_arg], dim=-1)

        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)
        axis, arg = torch.chunk(x, 2, dim=-1)
        
        axis_embeddings = convert_to_axis(axis)
        arg_embeddings = convert_to_arg(arg)
        
        return axis_embeddings, arg_embeddings