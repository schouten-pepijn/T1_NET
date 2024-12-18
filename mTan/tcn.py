# import libraries
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils

class T10_transformer(nn.Module):
    # def __init__(self, hp, seq_len, d_model, N_layers, heads):
    def __init__(self, hp):
        super().__init__()
        
        self.hp = hp
        
        seq_len = hp.xFA[2][1] - hp.xFA[2][0]
        d_model = 2
        d_hidden = d_model
        heads = 1
        N_layers = 12
        
        print(f'max length: {seq_len}')
        print(f'model dim: {d_model}')
        print(f'num heads: {heads}')
        print(f'num ff layers: {N_layers}') 
        
        self.conv1 = nn.Conv1d(d_model, d_model, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(d_model, d_model, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(d_model, d_model, 3, padding=3, dilation=3)
        self.conv4 = nn.Conv1d(d_model, d_model, 3, padding=4, dilation=4)
        self.conv5 = nn.Conv1d(d_model, d_model, 3, padding=5, dilation=5)
        self.conv6 = nn.Conv1d(d_model, d_model, 3, padding=6, dilation=6)

        self.gelu = nn.ReLU()
        
        self.dropout = nn.Dropout(p=0.1)
        
        # self.norm = utils.weight_norm()
        
        self.tcn_1 = nn.Sequential(self.conv1, self.gelu, self.dropout,
                                   self.conv2, self.gelu, self.dropout,
                                   self.conv3, self.gelu, self.dropout,
                                   self.conv4, self.gelu, self.dropout,
                                   self.conv5, self.gelu, self.dropout,
                                   self.conv5, self.gelu, self.dropout)
        # self.tcn_2 = nn.Sequential(self.conv1, self.gelu, self.dropout,
        #                            self.conv2, self.gelu, self.dropout,
        #                            self.conv3, self.gelu, self.dropout,
        #                            self.conv4, self.gelu, self.dropout)
        # self.tcn_3 = nn.Sequential(self.conv1, self.gelu, self.dropout,
        #                            self.conv2, self.gelu, self.dropout,
        #                            self.conv3, self.gelu, self.dropout,
        #                            self.conv4, self.gelu, self.dropout)
        # self.tcn_4 = nn.Sequential(self.conv1, self.gelu, self.dropout,
        #                            self.conv2, self.gelu, self.dropout,
        #                            self.conv3, self.gelu, self.dropout,
        #                            self.conv4, self.gelu, self.dropout)
        
        self.in_tcn = nn.Sequential(nn.Conv1d(1, d_model, 1), nn.ReLU())
        
        self.res_tcn_1 = nn.Sequential(nn.Conv1d(d_model, d_model, 1), nn.ReLU())
        # self.res_tcn_2 = nn.Sequential(nn.Conv1d(d_model, d_model, 1), nn.ReLU())
        # self.res_tcn_3 = nn.Sequential(nn.Conv1d(d_model, d_model, 1), nn.ReLU())
        # self.res_tcn_4 = nn.Sequential(nn.Conv1d(d_model, d_model, 1), nn.ReLU())
        
        
        self.end_tcn = nn.Sequential(nn.Conv1d(d_model, 1, 1), nn.ReLU())
        
        d_hidden=1
        
        self.regressor = nn.Sequential(nn.Linear(seq_len*d_hidden, seq_len*d_hidden//2),
                                        nn.ReLU(),
                                        nn.Linear(seq_len*d_hidden//2, seq_len*d_hidden//4),
                                        nn.ReLU(),
                                        nn.Linear(seq_len*d_hidden//4, 1)
                                        )
        
    def forward(self, X_fa, fa_vals, fa_len, fa_mask):

        shapes_fa = X_fa.size()
        
        X = X_fa.unsqueeze(dim=1)
        
        X = self.in_tcn(X)
        
        X = X + self.res_tcn_1(self.tcn_1(X))
        
        # X = X + self.res_tcn_2(self.tcn_2(X))
        
        # X = X + self.res_tcn_3(self.tcn_3(X))
        
        # X = X + self.res_tcn_4(self.tcn_4(X))
        
        # X = self.res_tcn_1(self.tcn_1(X))

        # X = self.res_tcn_2(self.tcn_2(X))

        # X = self.res_tcn_3(self.tcn_3(X))

        # X = self.res_tcn_4(self.tcn_4(X))
        
        
        
        X = self.end_tcn(X)
        
        X = self.regressor(X)
        


        T10 = X.squeeze()
        
        # T10 = self.regressor(X)

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


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        if d_model > 1:
            self.k_linear = nn.Linear(d_model, d_model)
            self.q_linear = nn.Linear(d_model, d_model)
            self.v_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(dim=0)
        
        # perform linear operations and split in h heads
        if self.d_model > 1:
            k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
            v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        else:
            k = k.view(bs, -1, self.h, self.d_k)
            q = q.view(bs, -1, self.h, self.d_k)
            v = v.view(bs, -1, self.h, self.d_k)
        
        # transpose to get dim bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # calculate attention 
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and into final dense layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        
        return output
    
def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(dim=1)
        scores = scores.masked_fill(mask, 1e-9)
        
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
    
    output = torch.matmul(scores, v)

    return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_out, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

# class Norm(nn.Module):
#     def __init__(self, d_model, eps=1e-6):
#         super().__init__()
        
#         self.size = d_model
#         # two parameters to lear normalisation
#         self.alpha = nn.Parameter(torch.ones(self.size))
#         self.bias = nn.Parameter(torch.zeros(self.size))
#         self.eps = eps
#     def forward(self, x):
#         norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
#             / (x.std(dim=-1, keepdim=True) * self.eps) * self.bias
#         return norm

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.norm = nn.LayerNorm(d_model, eps)
    def forward(self, x):
        return self.norm(x) 
      
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_hidden, heads, dropout=0.1):
        super().__init__()
        
        self.d_hidden = d_hidden
        
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_hidden)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        shapes = x.size()

        x_2 = self.norm_1(x)

        x = x + self.dropout_1(self.attn(x_2, x_2, x_2, mask))
        x_2 = self.norm_2(x)
        
        x = x.expand(shapes[0], shapes[1], self.d_hidden) + self.dropout_2(self.ff(x_2))
        # x = self.dropout_2(self.ff(x_2))
        return x

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
     

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