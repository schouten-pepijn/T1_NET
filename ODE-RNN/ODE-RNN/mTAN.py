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
from torchdiffeq import odeint_adjoint as odeint
# from torchdiffeq import odeint
from torch import utils


def init_network_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.1)
            nn.init.constant_(m.bias, val=0)

class GRU_Unit(nn.Module):
    def __init__(self, latent_dim, input_dim, n_units=100):
        super(GRU_Unit, self).__init__()

        self.update_gate = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())
        init_network_weights(self.update_gate)

        self.reset_gate = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())
        init_network_weights(self.reset_gate)

        self.new_state_net = nn.Sequential(
            nn.Linear(latent_dim  + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim ))
        init_network_weights(self.new_state_net)


    def forward(self, x, y_mean, mask):
        y_concat = torch.cat((y_mean, x), dim=-1)
        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = torch.cat((y_mean * reset_gate, x), dim=-1)
        new_state = self.new_state_net(concat)

        output_y = (1 - update_gate) * new_state + update_gate * y_mean
      
        # mask = (torch.sum(mask, -1, keepdim=True) > 0).float()
        mask = mask.unsqueeze(-1).float()
        new_y = mask * output_y + (1 - mask) * y_mean

        return new_y

def create_net(n_inputs, n_outputs, n_layers, n_units, nl=nn.Tanh):
    layers = [nn.Linear(n_inputs, n_units)]
    for i in range(n_layers):
        layers.append(nl())
        layers.append(nn.Linear(n_units, n_units))
    
    layers.append(nl())
    layers.append(nn.Linear(n_units, n_outputs))
    
    return nn.Sequential(*layers)


class DiffeqSolver(nn.Module):
    def __init__(self, ode_func, method, odeint_rtol, odeint_atol):
        super(DiffeqSolver, self).__init__()
        
        self.ode_method = method
        self.ode_func = ode_func
        
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol
        
    def forward(self, first_point, time_steps_to_predict):
        
        # n_traj_samples, n_traj = first_point.size(0), first_point.size(1)
        
        pred_y = odeint(self.ode_func, first_point, time_steps_to_predict,
                        rtol=self.odeint_rtol, atol=self.odeint_atol,
                        method=self.ode_method)
        
        pred_y = pred_y.permute(1,2,0,3)
        
        return pred_y
        


class ODEFunc(nn.Module):
    def __init__(self, ode_func_net):
        super(ODEFunc, self).__init__()
        
        self.gradient_net = ode_func_net
        
    def forward(self, t_local, y):
        return self.get_ode_gradient_nn(t_local, y)
    
    def get_ode_gradient_nn(self, t_local, y):
        return self.gradient_net(y)
        


class T10_transformer(nn.Module):
    def __init__(self, hp):
        super(T10_transformer, self).__init__()
        
        # network parameters
        self.hp = hp
        self.device = hp.device
        self.nhidden = hp.n_hidden

        self.input_size = hp.input_size
        self.rnn_unit = hp.rnn_unit
     
        hidden_reg = int(self.nhidden)   
     
        self.score_T10 = nn.Sequential(nn.Linear(hidden_reg, 1, bias=False),
                                        nn.Softmax(dim=1))
        
        self.score_M0 = nn.Sequential(nn.Linear(hidden_reg, 1, bias=False),
                                        nn.Softmax(dim=1))

        self.reg_T10 = nn.Sequential(nn.Linear(hidden_reg, 300),
                                     nn.ELU(),
                                     nn.Linear(300, 300),
                                     nn.ELU(),
                                     nn.Linear(300, 1))

        self.reg_M0 = nn.Sequential(nn.Linear(hidden_reg, 300),
                                    nn.ELU(),
                                    nn.Linear(300, 300),
                                    nn.ELU(),
                                    nn.Linear(300, 1))

        latent_dim = self.nhidden
        layers = 3
        units = 124
        ode_func_net = create_net(latent_dim, latent_dim,
                                  n_layers=layers,
                                  n_units=units,
                                  nl=nn.Tanh)
        
        self.x0 = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                nn.Tanh())

        ode_method = 'rk4'
        rec_ode_func = ODEFunc(ode_func_net=ode_func_net)
        self.ode_solver = DiffeqSolver(rec_ode_func, ode_method, odeint_rtol=1e-3, odeint_atol=1e-4)

        if self.rnn_unit == 'RNN':
            self.rnn_cell = nn.RNNCell(1, self.nhidden)
        elif self.rnn_unit == 'GRU':
            self.rnn_cell = nn.GRUCell(1, self.nhidden)
        #     self.rnn_cell_2 = nn.GRUCell(self.nhidden, self.nhidden)
        
        # self.rnn_cell = GRU_Unit(self.nhidden, 1)

    def forward(self, X_fa_in, fa_vals, fa_mask, fa_len, TR_vals, fa_union):
        X_fa = X_fa_in

        # fa_full = torch.FloatTensor([math.radians(i) for i in range(1,fa_vals.size(-1)+1)])
        
        # fa_mask_in = (fa_mask == 1).any(axis=0)
        # fa_union = fa_full[fa_mask_in]
        
        X_fa = X_fa[:, :len(fa_union)]
        fa_mask = fa_mask[:, :len(fa_union)]
        
        prev_hidden = torch.zeros((X_fa.size(0), self.nhidden)).to(self.hp.device)
        # prev_hidden_2 = torch.zeros((X_fa.size(0), self.nhidden)).to(self.hp.device)

        # fa_union = fa_full
        
        # prev_hidden = self.x0(prev_hidden)
        ode_trajectory = prev_hidden.unsqueeze(dim=1).clone().detach()
        
        num_steps = 4
        
        """ first ODE step """
        time_points = torch.linspace(0, fa_union[0], num_steps).to(self.hp.device)
        ode_sol_steps = self.ode_solver(prev_hidden.unsqueeze(0), time_points)
             
        prev_hidden = ode_sol_steps[0][:, -1]
        
        """ first RNN step """
        prev_hidden_rnn = self.rnn_cell(X_fa[:, 0].unsqueeze(-1),
                                        prev_hidden,
                                        # mask=fa_mask[:, 0]
                                        )

        mask = fa_mask[:, 0].unsqueeze(-1).float()
        prev_hidden = prev_hidden_rnn * mask + (1 - mask) * prev_hidden
 
        hidden_ode = prev_hidden.unsqueeze(1)
        
        ode_trajectory = ode_sol_steps[0].clone()
        ode_trajectory[:, -1] = prev_hidden.clone().detach()
        
        trajectory = ode_trajectory

        for i in range(1, len(fa_union)):
            time_points = torch.linspace(fa_union[i-1], fa_union[i], num_steps).to(self.hp.device)
            ode_sol_steps = self.ode_solver(prev_hidden.unsqueeze(0), time_points)
            prev_hidden = ode_sol_steps[0][:, -1]
            
            prev_hidden_rnn = self.rnn_cell(X_fa[:, i].unsqueeze(-1),
                                            prev_hidden,
                                            # mask=fa_mask[:, i]
                                            )

            mask = fa_mask[:, i].unsqueeze(-1).float()
            prev_hidden = prev_hidden_rnn * mask + (1 - mask) * prev_hidden

            hidden_ode = torch.cat((hidden_ode,
                                    prev_hidden.unsqueeze(1)), dim=1)
            
            ode_trajectory = ode_sol_steps[0].clone().detach()
            ode_trajectory[:, -1] = prev_hidden.clone().detach()
            
            trajectory = torch.cat((trajectory, 
                                    ode_trajectory), dim=1)

        hidden_T10 = torch.sum(self.score_T10(hidden_ode)*hidden_ode, dim=1)
        hidden_M0 = torch.sum(self.score_M0(hidden_ode)*hidden_ode, dim=1)
        
        T10 = self.reg_T10(hidden_T10).squeeze()
        M0 = self.reg_M0(hidden_M0).squeeze()        

        # # update T10
        T10_diff = self.hp.simulations.T10bounds[1] - self.hp.simulations.T10bounds[0]
        T10 = self.hp.simulations.T10bounds[0] + torch.sigmoid(T10.unsqueeze(1)) * T10_diff
        
        M0_diff = self.hp.simulations.M0bounds[1] - self.hp.simulations.M0bounds[0]
        M0 = self.hp.simulations.M0bounds[0] + torch.sigmoid(M0.unsqueeze(1)) * M0_diff
    
        # if ~self.hp.sup:
        R1 = 1 / T10
        X_out = torch.mul(
            (1 - torch.exp(torch.mul(-TR_vals, R1))), torch.sin(fa_vals)) / (
            1 - torch.mul(torch.cos(fa_vals), torch.exp(torch.mul(-TR_vals, R1))))

        X_out *= M0

        
        # M0 = torch.ones_like(T10)
        return X_out, T10, M0, trajectory

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
    