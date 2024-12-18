# import libraries
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.jit as jit



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

        self.rnn_f = GRUDmodel(input_dim=1,
                              hidden_dim=self.nhidden,
                              device=self.device,
                              reverse=False)

        hidden_reg = self.nhidden
        self.score_T10 = nn.Sequential(nn.Linear(hidden_reg, 1,
                                                 bias=False),
                                        nn.Softmax(dim=1))
        
        self.score_M0 = nn.Sequential(nn.Linear(hidden_reg, 1,
                                                 bias=False),
                                        nn.Softmax(dim=1))
    
            
        # flip angle length
        if self.len_concat:
            hidden_reg += 1
        
        if self.fa_concat:
            if self.hp.xFA[1][1] is None:
                hidden_reg += self.hp.xFA[1][0]
            else:
                hidden_reg += self.hp.xFA[1][1]
        
        self.reg_T10 = nn.Sequential(nn.Dropout(self.dropout_p),
                                     nn.Linear(int(hidden_reg), 300),
                                     nn.ELU(),
                                     nn.Dropout(self.dropout_p),
                                     nn.Linear(300, 300),
                                     nn.ELU(),
                                     nn.Linear(300, 1))
        
        self.reg_M0 = nn.Sequential(nn.Dropout(self.dropout_p),
                                     nn.Linear(int(hidden_reg), 300),
                                     nn.ELU(),
                                     nn.Dropout(self.dropout_p),
                                     nn.Linear(300, 300),
                                     nn.ELU(),
                                     nn.Linear(300, 1))



    def forward(self, X_fa, X_last, X_mean, X_mask, X_delta, fa_vals, fa_len, TR_vals):
        
        X_fa_in = X_fa

        out, X_gru = self.rnn_f(X_in=X_fa_in.unsqueeze(dim=-1),
                                X_mask=X_mask.unsqueeze(dim=-1),
                                X_delta=X_delta.unsqueeze(dim=-1),
                                X_last=X_last.unsqueeze(dim=-1),
                                X_mean=X_mean.unsqueeze(dim=-1))

        hidden_T10 = torch.sum(out*self.score_T10(out), dim=1)
        hidden_M0 = torch.sum(out*self.score_M0(out), dim=1)
    
        # regress T10 value
        T10 = self.reg_T10(hidden_T10).squeeze()
        M0 = self.reg_M0(hidden_M0).squeeze()

        # # update T10
        T10_diff = self.hp.simulations.T10bounds[1] - self.hp.simulations.T10bounds[0]
        T10 = self.hp.simulations.T10bounds[0] + torch.sigmoid(T10.unsqueeze(1)) * T10_diff
        M0_diff = self.hp.simulations.M0bounds[1] - self.hp.simulations.M0bounds[0]
        M0 = self.hp.simulations.M0bounds[0] + torch.sigmoid(M0.unsqueeze(1)) * M0_diff

        R1 = 1 / T10
        X_out = torch.mul(
            (1 - torch.exp(torch.mul(-TR_vals, R1))), torch.sin(fa_vals)) / (
            1 - torch.mul(torch.cos(fa_vals),
                          torch.exp(torch.mul(-TR_vals, R1))))

        X_out *= M0

        return X_out, T10, M0, out, X_gru


def push_zeros(a, device):
    valid_mask = a!=0
    flipped_mask = torch.fliplr(
        torch.sum(valid_mask, dim=1, keepdim=True)
        > torch.arange(a.shape[1]-1, -1, -1).to(device))
    a[flipped_mask] = a[valid_mask]
    a[~flipped_mask] = 0
    return a


# class GRUDcell(nn.Module):
class GRUDcell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):        
        super(GRUDcell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.x_combined = nn.Linear(input_size, 3 * hidden_size, bias=True)
        
        self.h_n = nn.Linear(hidden_size, hidden_size, bias=False)
        self.h_combined = nn.Linear(hidden_size, 2 * hidden_size, bias=False)

        self.m_combined = nn.Linear(input_size, 3 * hidden_size, bias=False)
        
        self.gamma_x = nn.Linear(input_size, input_size, bias=True)
        self.gamma_h = nn.Linear(input_size, hidden_size, bias=True)

        self.relu = nn.ReLU()
        
        self.zeros_in = torch.zeros(input_size)
        self.zeros_hidden = torch.zeros(hidden_size)

        self.reset_parameters()
      
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    @jit.script_method
    def forward(self, X_in, X_mask, X_delta, X_last, X_mean, hidden):
        
        X_in = X_in.view(-1, X_in.size(1))
        X_mask = X_mask.view(-1, X_mask.size(1))
        X_delta = X_delta.view(-1, X_delta.size(1))
        X_last = X_last.view(-1, X_last.size(1))
        X_mean = X_mean.view(-1, X_mean.size(1))
        
        # print(X_mask)
        # input()

        gamma_x = torch.exp(-1.*torch.maximum(self.zeros_in, self.gamma_x(X_delta)))
        
        # X = (X_mask * X_in) + (1 - X_mask) * ((gamma_x * X_last) + (1 - gamma_x) * X_mean)
        
        # X = (X_mask * X_in) + (1 - X_mask) * ((gamma_x * X_last) + (1 - gamma_x) * X_mean)
        
        X = (X_mask * X_in) + (1 - X_mask) * gamma_x * X_last + (1 - X_mask) * ( 1 - gamma_x) * X_mean
        
        # X = X_in
        gamma_h = torch.exp(-1.*torch.maximum(self.zeros_hidden, self.gamma_h(X_delta)))
        
        hidden = torch.squeeze(gamma_h * hidden)
        
        x_combined = self.x_combined(X)
        x_z, x_r, x_n = x_combined.chunk(3, 1)
        
        h_combined = self.h_combined(hidden)
        h_z, h_r = h_combined.chunk(2, 1)
        
        m_combined = self.m_combined(X_mask)
        m_z, m_r, m_n = m_combined.chunk(3, 1)
        
        z_gate = torch.sigmoid(x_z + h_z + m_z)
        r_gate = torch.sigmoid(x_r + h_r + m_r)
        
        h_n = self.h_n(r_gate * hidden)
        
        h_tilde = torch.tanh(x_n + h_n + m_n)
        
        h_t = (1 - z_gate) * hidden + z_gate * h_tilde
        
        # h_t = X_mask * h_t + (1 - X_mask) * hidden
        # hidden = X_mask * h_t + (1 - X_mask) * torch.squeeze(gamma_h * hidden)
        
        # return hidden, X
        return h_t, X
    

# class GRUDmodel(nn.Module):
class GRUDmodel(jit.ScriptModule):
    def __init__(self, input_dim, hidden_dim, device, reverse=False, bias=True):
        super(GRUDmodel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.reverse = reverse
        
        self.gru_cell = GRUDcell(input_dim, hidden_dim)
        
    @jit.script_method 
    def forward(self, X_in, X_mask, X_delta, X_last, X_mean):
        hn = torch.zeros(X_in.size(0), self.hidden_dim).to(self.device)

        hn_out = []
        X_out = []
        for i in range(X_in.size(1)):
            hn, X = self.gru_cell(
                        X_in=X_in[:, i],
                        X_mask=X_mask[:, i],
                        X_delta=X_delta[:, i],
                        X_last=X_last[:, i],
                    X_mean=X_mean[:, i],
                    hidden=hn)
            hn_out.append(hn.unsqueeze(1))
            X_out.append(X)
            
        hn_out = torch.cat(hn_out, dim=1)
        X_out = torch.cat(X_out, dim=-1)
        return hn_out, X_out