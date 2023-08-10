import collections
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm


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
    def __init__(self, nentity, nrelation, hidden_dim, gamma, test_batch_size=1, use_cuda=False, query_name_dict=None, center_reg=None, drop=0.):
        super(KGReasoning, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
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
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim),
                                             requires_grad=True)  # axis for entities
        self.angle_scale = AngleScale(self.embedding_range.item())  # scale axis embeddings to [-pi, pi]

        self.modulus = nn.Parameter(torch.Tensor([0.5 * self.embedding_range.item()]), requires_grad=True)

        self.axis_scale = 1.0
        self.arg_scale = 1.0

        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

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

        for train_queries, train_answers, train_indices, train_positive, train_negative in train_iterator:
            # break
                
            
            # positive_sample, negative_sample, subsampling_weight, batch_queries = next(train_iterator)

            positive_logit, negative_logit, _ = model(train_answers, train_answers, train_queries, train_indices, train_positive, train_negative)
            # positive_logit, negative_logit, subsampling_weight, _ = model(positive_sample, negative_sample, subsampling_weight, batch_queries)

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
                'positive_sample_loss': positive_score.item(),
                'negative_sample_loss': negative_score.item(),
                'loss': loss.item(),
            }
            print(log)
        return log

    def test_step(self, model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, save_result=False, save_str="", save_empty=False):
        model.eval()

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)

        with torch.no_grad():
            for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader, disable=not args.print_on_screen):
                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)
                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)
                for query_structure in batch_queries_dict:
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
                    else:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                if args.cuda:
                    negative_sample = negative_sample.cuda()

                _, negative_logit, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict)

                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]
                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                if len(argsort) == args.test_batch_size: 
                    ranking = ranking.scatter_(1, argsort, model.batch_entity_range) 
                else: 
                    if args.cuda:
                        ranking = ranking.scatter_(1, argsort, torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 1).cuda())  
                    else:
                        ranking = ranking.scatter_(1, argsort, torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 1)) 

                for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
                    hard_answer = hard_answers[query]
                    easy_answer = easy_answers[query]

                    num_hard = len(hard_answer)
                    num_easy = len(easy_answer)
                    assert len(hard_answer.intersection(easy_answer)) == 0

                    cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy
                    if args.cuda:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                    else:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float)
                    cur_ranking = cur_ranking - answer_list + 1  # filtered setting
                    cur_ranking = cur_ranking[masks]  # only take indices that belong to the hard answers

                    mrr = torch.mean(1. / cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                    logs[query_structure].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                    })
                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1
        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]]) / len(
                    logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics


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
        

        for query, idx in zip(batch_queries, query_indices):
            
            axis_embedding, arg_embedding, _ = self.embed_query_cone(query, [], idx)
            
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
                # batch_size, negative_size = negative_sample_regular.shape

                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_indices).unsqueeze(1)

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