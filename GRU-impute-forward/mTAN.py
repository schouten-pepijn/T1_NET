# import libraries
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import DCE_matt as dce
import numpy as np
import functions
import matplotlib


class T10_transformer(nn.Module):
    def __init__(self, hp):
        super(T10_transformer, self).__init__()
        
        # network parameters
        self.hp = hp
        self.len_concat = hp.len_concat
        self.fa_concat = hp.fa_concat
        self.learn_emb = hp.learn_emb
        self.embed_time = hp.embed_time
        self.dim = hp.input_dim
        self.device = hp.device
        self.nhidden = hp.n_hidden
        self.num_heads = hp.n_heads
        self.gru_layers = hp.gru_layers
        self.dropout_p = hp.dropout_p
        self.gru_bi = hp.gru_bi
        self.mh_att = hp.mh_attention
        self.one_reg = hp.one_reg
        self.ref_angle = hp.ref_angle
        self.layer_bool = hp.layer_bool

        # full flip angle range
        if self.mh_att:
            print('using multihead attention')
            
            # self.query = torch.FloatTensor([math.radians(i) for i in range(*hp.xFA[2])])
            # self.query /= self.query.max()
            # self.q_len = len(self.query)
            self.query = torch.linspace(0, 1, 34).to(self.hp.device)
            self.input_len = int(self.hp.xFA[2][1] - self.hp.xFA[2][0])
            # self.query = torch.arange(hp.xFA[2][0], hp.xFA[2][1],
            #                           1, dtype=torch.float)
            # self.layernorm_in = nn.LayerNorm(self.input_len,
            #                                   elementwise_affine=False)
            # self.layernorm_ref = nn.LayerNorm(50)

            # attention layer
            # self.attention = MultiTimeAttention(d_input=self.dim,
            #                                     d_hidden=self.nhidden,
            #                                     d_time=self.embed_time,
            #                                     num_heads=self.num_heads,
            #                                     dropout_p=0.)
    
            # rnn layer
            self.rnn = nn.GRU(input_size=2, 
                              hidden_size=self.nhidden,
                              num_layers=1,
                              dropout=0,
                              bidirectional=self.gru_bi,
                              batch_first=True) 
            
            hidden_last = 2*self.nhidden if self.gru_bi else self.nhidden

            # self.score_rnn = nn.Sequential(nn.Linear(hidden_last, hidden_last, bias=False),
            #                                 nn.Softmax(dim=1))
        
            num_layers = 1 if self.gru_layers == 1 else int(self.gru_layers-1)
            drop_out = 0 if num_layers == 1 else 0.2
            # self.rnn_last = nn.GRU(input_size=hidden_last,
            #                         hidden_size=self.nhidden,
            #                         num_layers=num_layers,  # was 3
            #                         dropout=0,
            #                         bidirectional=self.gru_bi,
            #                         batch_first=True)
            
            # linear/periodic time embedding
            if self.learn_emb:
                if hp.time_embedding_test:
                    self.time_embedding = TimeEmbedding_test(hp=self.hp, 
                                                        in_features=1,
                                                        out_features=self.embed_time,
                                                        dropout_p=self.dropout_p)
                else:
                    self.time_embedding = TimeEmbedding(hp=self.hp, 
                                                        in_features=1,
                                                        out_features=self.embed_time,
                                                        dropout_p=self.dropout_p)

        # uni/bi directional
        if self.gru_bi:
            # hidden_reg = 2*self.nhidden
            hidden_reg = 2*self.nhidden
        else:
            hidden_reg = self.nhidden
        
        self.score_T10 = nn.Sequential(nn.Linear(hidden_reg, 1,
                                                 bias=False),
                                        nn.Softmax(dim=1))
        self.score_M0 = nn.Sequential(nn.Linear(hidden_reg, 1,
                                                 bias=False),
                                        nn.Softmax(dim=1))

        if self.len_concat:
            hidden_reg += 1
        
        if self.fa_concat:
            if self.hp.xFA[1][1] is None:
                hidden_reg += self.hp.xFA[1][0]
            else:
                hidden_reg += self.hp.xFA[1][1]
        
        self.reg_T10 = nn.Sequential(nn.Dropout(self.dropout_p),
                                     nn.Linear(int(hidden_reg), 200),
                                     nn.ELU(),
                                     nn.Dropout(self.dropout_p),
                                     nn.Linear(200, 200),
                                     nn.ELU(),
                                     nn.Linear(200, 1))
        self.reg_M0 = nn.Sequential(nn.Dropout(self.dropout_p),
                                     nn.Linear(int(hidden_reg), 200),
                                     nn.ELU(),
                                     nn.Dropout(self.dropout_p),
                                     nn.Linear(200, 200),
                                     nn.ELU(),
                                     nn.Linear(200, 1))



    def forward(self, X_fa_in, fa_vals, fa_mask, fa_len, TR_vals):
        # adjusting input
        in_shapes = X_fa_in.size()
      
        fa_vals_in = fa_vals.clone()
        X_fa = X_fa_in.clone()
        for row_idx in range(X_fa_in.size(dim=0)):
            temp_X = 0
            temp_fa = 0
            for col_idx in range(X_fa_in.size(dim=1)):
                if X_fa[row_idx, col_idx] != 0:
                    temp_X = X_fa_in[row_idx, col_idx]
                    temp_fa = fa_vals_in[row_idx, col_idx]
                else:
                    X_fa[row_idx, col_idx] = temp_X
                    fa_vals_in[row_idx, col_idx] = temp_fa
    
        X_fa = torch.cat((X_fa.unsqueeze(dim=-1),
                          fa_vals_in.unsqueeze(dim=-1)), dim=-1)

        out, _ = self.rnn(X_fa)

        hidden_T10 = torch.sum(out*self.score_T10(out), dim=1)
        hidden_M0 = torch.sum(out*self.score_M0(out), dim=1)
            
        # regress T10 value
        T10 = self.reg_T10(hidden_T10).squeeze()
        M0 = self.reg_M0(hidden_M0).squeeze()

        # # update T10
        T10_diff = self.hp.simulations.T10bounds[1] - self.hp.simulations.T10bounds[0]
        T10 = self.hp.simulations.T10bounds[0] + torch.sigmoid(T10.unsqueeze(1)) * T10_diff
        M0_diff = self.hp.simulations.M0bounds[1] - self.hp.simulations.M0bounds[0]
        M0 = self.hp.simulations.M0bounds[0] + torch.sigmoud(M0.unsqueeze(1)) * M0_diff
        
        
        R1 = 1 / T10
        X_out = torch.mul(
            (1 - torch.exp(torch.mul(-TR_vals, R1))), torch.sin(fa_vals)) / (
            1 - torch.mul(torch.cos(fa_vals),
                          torch.exp(torch.mul(-TR_vals, R1))))

        X_out = X_out * M0

        return X_out, T10, M0

class MultiTimeAttention(nn.Module):
    def __init__(self, d_input, d_hidden=16, d_time=4,
                 num_heads=1, dropout_p=None, layer_bool=None):
        super(MultiTimeAttention, self).__init__()
        assert d_time % num_heads == 0
        
        # network parameters
        self.d = d_time
        self.d_k = d_time // num_heads
        self.h = num_heads
        self.dim = d_input
        self.d_h = d_hidden
        self.dropout_p = dropout_p
        self.layer_bool = layer_bool

        # linear layers
        self.k_linear = nn.Linear(d_time, d_time)
        self.q_linear = nn.Linear(d_time, d_time)
        self.out = nn.Linear(d_input*num_heads, d_hidden)
        
        if dropout_p is not None:
            self.dropout = nn.Dropout(dropout_p)

    def create_mask(self, mask, dim):
        mask = mask.unsqueeze(dim=-1)
        mask = mask.repeat_interleave(dim, dim=-1)
        mask = mask.unsqueeze(dim=1)
        mask = ~mask
        return mask

    def attention(self, q, k, v, mask=None):
        # value/query size
        d_v, d_q = v.size(dim=-1), q.size(dim=-1)

        # scaled dot product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_q)
        scores = scores.unsqueeze(dim=-1).repeat_interleave(d_v, dim=-1)

        # update scores with padding mask
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(dim=-3), -1e9)

        # normalize scoring to probabilities
        p_scores = F.softmax(scores, dim=-2)

        # apply dropout
        if self.dropout_p is not None:
            p_scores = self.dropout(p_scores)

        output = torch.sum(p_scores*v.unsqueeze(dim=-3), dim=-2)
        return output


    def forward(self, q, k, v, mask=None):
        bs, seq_len, dim = v.size()
        v = v.unsqueeze(1)

        if mask is not None:
            mask = self.create_mask(mask, dim)

        q = self.q_linear(q).view(q.size(dim=0), -1, self.h, self.d_k)
        k = self.k_linear(k).view(k.size(dim=0), -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        scores = self.attention(q, k, v, mask)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.h*dim)

        return self.out(concat)


def PositionalEncoding(pos, d_model, device):
    assert d_model % 2 == 0
    
    pos = pos.to(device)
    pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(device)
    position = 48. * pos.unsqueeze(dim=2)
    
    div_term = torch.exp(
        torch.arange(0, d_model, 2) * -(math.log(10.0) / d_model))
    div_term = div_term.to(device)
    
    pe[:, :, 0::2] = torch.sin(position * div_term)
    pe[:, :, 1::2] = torch.cos(position * div_term)
    pe = pe.to(device)
    return pe

class TimeEmbedding(nn.Module):
    def __init__(self, hp, in_features, out_features, dropout_p=0.1):
        super(TimeEmbedding, self).__init__()

        self.hp = hp
        self.periodic = nn.Linear(in_features, out_features-1)
        self.linear = nn.Linear(in_features, in_features)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, t):
        t = t.unsqueeze(dim=-1).to(self.hp.device)
        out_linear = self.linear(t)
        out_periodic = self.periodic(t).sin()
        out = torch.cat((out_linear, out_periodic), dim=-1)
        return self.dropout(out)


class TimeEmbedding_test(nn.Module):
    def __init__(self, hp, in_features, out_features, dropout_p=0.2):
        super(TimeEmbedding_test, self).__init__()

        self.hp = hp
        self.periodic = nn.Linear(in_features, out_features-1)
        self.A = nn.Linear(out_features-1, out_features-1, bias=False)
        self.linear = nn.Linear(in_features, in_features)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, t):
        t = t.unsqueeze(dim=-1).to(self.hp.device)
        out_linear = self.linear(t)
        out_periodic = self.periodic(t).sin()
        out_periodic = self.A(out_periodic)
        out = torch.cat((out_linear, out_periodic), dim=-1)
        return self.dropout(out)


def push_zeros(a, device):
    valid_mask = a!=0
    flipped_mask = torch.fliplr(
        torch.sum(valid_mask, dim=1, keepdim=True)
        > torch.arange(a.shape[1]-1, -1, -1).to(device))
    a[flipped_mask] = a[valid_mask]
    a[~flipped_mask] = 0
    return a


class TransformerEnc(nn.Module):
    def __init__(self, d_input, num_heads=1, dropout=0, d_ff=2048):
        super(TransformerEnc, self).__init__()
        
        self.d_input = d_input
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        
        self.attention = nn.MultiheadAttention(self.d_input, self.num_heads)
        self.norm1 = nn.LayerNorm(self.d_input)
        self.linear = nn.Sequential(
            nn.Linear(self.d_input, self.d_ff),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff, self.d_input))
        self.norm2 = nn.LayerNorm(self.d_input)
        
    def forward(self, X_in):
        X, _ = self.attention(X_in, X_in, X_in)
        X = torch.add(X, X_in)
        X = self.norm1(X)
        X_out = self.linear(X)
        X_out = torch.add(X_out, X)
        X_out = self.norm2(X_out)
        return X_out
    