# import libraries
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import DCE_matt as dce
import numpy as np
import functions
import matplotlib


class T10_transformer(nn.Module):
    def __init__(self, hp):
        super(T10_transformer, self).__init__()
        self.hp = hp   

        self.encoder_lin = nn.Sequential(nn.Linear(6, 320),
                                         nn.GELU(),
                                         nn.Linear(320, 640),
                                         nn.GELU(),
                                         nn.Linear(640, 320),
                                         nn.GELU(),
                                         nn.Linear(320, 60),
                                         nn.GELU(),
                                         nn.Linear(60, 1)
                                         )
        
    def forward(self, X_fa, fa_vals, fa_len, fa_mask):

        # shapes_fa = X_fa.size()

        X_in = push_zeros(X_fa, self.hp.device)

        X_in = X_in[:, :6].contiguous()
        
        T10 = self.encoder_lin(X_in)

        T10_diff = self.hp.simulations.T10bounds[1, 0] - self.hp.simulations.T10bounds[0, 0]
        # S0_diff = self.hp.simulations.T10bounds[1, 1] - self.hp.simulations.T10bounds[0, 1]

        T10 = self.hp.simulations.T10bounds[0, 0] + torch.sigmoid(T10.unsqueeze(1)) * T10_diff
        # S0 = self.hp.simulations.T10bounds[0, 1] + torch.sigmoid(S0.unsqueeze(1)) * S0_diff

        # R1 = 1 / T10
        # R1_in_fa = torch.tile(R1, (1, shapes_fa[1])).to(self.hp.device)
        # TR_fa = torch.tile(TR_vals, (1, shapes_fa[1])).to(self.hp.device)

        # X_fa = torch.mul(
        #     (1 - torch.exp(torch.mul(-TR_fa, R1_in_fa))), torch.sin(fa_mask)) / (
        #     1 - torch.mul(torch.cos(fa_mask), torch.exp(torch.mul(-TR_fa, R1_in_fa))))
        # X_fa *= S0

        return T10


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(dropout_p)
        pos_encoding = torch.zeros(1, max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(1, -1, 1)
        div_term = torch.exp(torch.arange(0, dim_model).float() * (-math.log(10000))/ dim_model)
        pos_encoding= torch.sin(positions_list * div_term)
        self.register_buffer('pos_encoding', pos_encoding)
        
    def forward(self, embedding):
        return self.dropout(embedding + self.pos_encoding)


def t2v(tau, f, out_features, w, b, w0, b0):
    tau = tau.clone()
    w = w.clone()
    w0 = w0.clone()
    # k-1 periodic features
    v1 = f(torch.matmul(tau, w) + b)
    # One Non-periodic feature
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 2)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)
    
    
def push_zeros(a, device):
    valid_mask = a!=0

    flipped_mask = torch.fliplr(
        torch.sum(valid_mask, dim=1, keepdim=True)
        > torch.arange(a.shape[1]-1, -1, -1).to(device))
    a[flipped_mask] = a[valid_mask]
    a[~flipped_mask] = 0

    return a