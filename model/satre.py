"""
SAT model for relation extraction.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.tree import head_to_tree, tree_to_adj
from utils import constant, torch_utils


class GCNClassifier(nn.Module):
    """ A wrapper classifier for RelationModel. """

    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.re_model = RelationModel(opt, emb_matrix=emb_matrix)
        in_dim = opt['hidden_dim']
        self.classifier = nn.Linear(in_dim, opt['num_class'])
        self.opt = opt

    def forward(self, inputs):
        outputs, pooling_output = self.re_model(inputs)
        logits = self.classifier(outputs)
        return logits, pooling_output


class RelationModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix

        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        embeddings = (self.emb, self.pos_emb, self.ner_emb)
        self.init_embeddings()

        # sat layer
        self.satre = SATRE(opt, embeddings)

        # mlp output layer
        in_dim = opt['hidden_dim'] * 3
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers'] - 1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        # decide finetuning
        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: torch_utils.keep_partial_grad(x, self.opt['topn']))
        else:
            print("Finetune all embeddings.")

    def forward(self, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs # unpack
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(l)

        def inputs_to_tree_reps(head, l):
            trees = [head_to_tree(head[i], l[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=False).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return Variable(adj.cuda()) if self.opt['cuda'] else Variable(adj)

        adj = inputs_to_tree_reps(head.data, l)
        h, pool_mask = self.satre(adj, inputs)

        # pooling
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2)  # invert mask
        pool_type = self.opt['pooling']
        h_out = pool(h, pool_mask, type=pool_type)
        subj_out = pool(h, subj_mask, type="max")
        obj_out = pool(h, obj_mask, type="max")
        outputs = torch.cat([h_out, subj_out, obj_out], dim=1)
        outputs = self.out_mlp(outputs)

        return outputs, h_out


class SATRE(nn.Module):
    def __init__(self, opt, embeddings):
        super().__init__()
        self.opt = opt
        self.in_dim = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']
        self.emb, self.pos_emb, self.ner_emb = embeddings
        self.use_cuda = opt['cuda']
        self.mem_dim = opt['hidden_dim']

        # rnn layer
        if self.opt.get('rnn', False):
            self.input_W_R = nn.Linear(self.in_dim, opt['rnn_hidden'])
            self.rnn = nn.LSTM(opt['rnn_hidden'], opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, \
                               dropout=opt['rnn_dropout'], bidirectional=True)
            self.in_dim = opt['rnn_hidden'] * 2
            self.rnn_drop = nn.Dropout(opt['rnn_dropout'])  # use on last layer output
        self.input_W_G = nn.Linear(self.in_dim, self.mem_dim)

        self.in_drop = nn.Dropout(opt['input_dropout'])
        self.num_layers = opt['num_layers']
        self.heads = opt['heads']
        self.sat = clones(SATLayer(opt, self.mem_dim), self.heads)
        self.linears = nn.Linear((self.heads + 1) * self.mem_dim, self.mem_dim)
        self.relu = nn.ReLU()


    def encode_with_rnn(self, rnn_inputs, masks, batch_size):
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'])
        
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        self.rnn.flatten_parameters()
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs # unpack
        
        src_mask = (words != constant.PAD_ID).unsqueeze(-2)
        word_embs = self.emb(words)
        embs = [word_embs]

        if self.opt['pos_dim'] > 0:
            embs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            embs += [self.ner_emb(ner)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        if self.opt.get('rnn', False):
            embs = self.input_W_R(embs)
            sat_inputs = self.rnn_drop(self.encode_with_rnn(embs, masks, words.size()[0]))
            sat_inputs = self.input_W_G(sat_inputs)
            sat_inputs += embs
        else:
            sat_inputs = embs
            sat_inputs = self.input_W_G(sat_inputs)

        cache_list = [sat_inputs]
        mask = masks.unsqueeze(2)
        for i in range(self.heads):
            outputs = self.sat[i](adj, sat_inputs, mask)
            outputs += sat_inputs
            cache_list.append(outputs)
        aggregation =  torch.cat(cache_list, -1)
        outputs = self.relu(self.linears(aggregation))

        return outputs, mask


class SATLayer(nn.Module):
    def __init__(self, opt, mem_dim):
        super(SATLayer, self).__init__()
        self.opt = opt
        self.mem_dim = mem_dim
        self.sat = SAT(self.mem_dim)
        self.sat_drop = nn.Dropout(opt['gcn_dropout'])

    def forward(self, adj, sat_inputs, adj_mask):

        outputs = self.sat(sat_inputs, sat_inputs, sat_inputs, adj)
        outputs = pool_subtree(outputs, adj_mask, type = 'max')

        return outputs

def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)

def pool_subtree(h, mask, type='max'):
    mask = mask.unsqueeze(-1)
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.cuda(), c0.cuda()


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def sast(query, key, value, mask, dropout=None):
    "Compute 'Self-Attention over subtree'"
    

    _, nsteps, d_k = query.size()
    
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    scores = torch.exp(scores)
    scores = scores.unsqueeze(1).repeat(1,nsteps,1,1)

    value = value.unsqueeze(1).repeat(1,nsteps,1,1)
    mask = mask.unsqueeze(-1)
    mask = torch.matmul(mask, mask.transpose(-2, -1))

    scores = scores.masked_fill(mask == 0, 0)
    sum_scores = torch.sum(scores, dim =-1).unsqueeze(-1)
    p_attn = torch.div(scores, sum_scores)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class SAT(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(SAT, self).__init__()
        
        
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask):
        # Same mask applied to all h heads.
        nbatches, nsteps, d_model = query.shape

        query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))]
        
        x, self.attn = sast(query, key, value, mask=mask, 
                                 dropout=self.dropout)

        return self.linears[-1](x)