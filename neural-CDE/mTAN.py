# import libraries
import torch
import torch.nn as nn
import torchcde

"""From https://github.com/patrick-kidger/torchcde/blob/master/example/time_series_classification.py 
Hij gebruikt dit als CDE function.""" 
# class CDE_Func(nn.Module):
class CDE_Func(nn.Module):
    def __init__(self, hp, input_channels=2, hidden_channels=2):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDE_Func, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.device = hp.device
        
        self.linear1 = torch.nn.Linear(hidden_channels, 256).to(self.device)
        self.linear2 = torch.nn.Linear(256, input_channels*hidden_channels).to(self.device)

    ######################
    # For most purposes the t argument can probably be ignored; unless you want your CDE to behave differently at
    # different times, which would be unusual. But it's there if you need it!
    ######################
    # @jit.script_method
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z.to(self.device))
        z = z.relu()
        z = self.linear2(z.to(self.device))
        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        z = z.tanh()
        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        
        # z = torch.swapaxes(z, -1, -2)
        
        return z.to(self.device)


# class T10_transformer(nn.Module):
class T10_transformer(nn.Module):
    def __init__(self, hp, input_channels=2, hidden_channels=8, output_channels=4):
        super(T10_transformer, self).__init__()

        # self.batch_size = hp.batch_size
        # self.latent_dim = hp.lstm_hidden_dim
        self.device = hp.device
        self.interpolation = "cubic"
        self.hp = hp
        # only the signals
        self.features = 1
        self.hidden_size = hidden_channels
        
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
        self.T10_under = hp.simulations.T10bounds[0]
        self.T10_upper = hp.simulations.T10bounds[1]

        self.initial = torch.nn.Linear(input_channels, hidden_channels).to(hp.device)
        
        # function to give to the ODE solver
        self.ode_net = CDE_Func(hp, input_channels, hidden_channels)
        
        # Output MLP that gives the four parameters
        # if self.gru_bi:
        #     # hidden_reg = 2*self.nhidden
        #     hidden_reg = 2*self.nhidden
        # else:
        #     hidden_reg = self.nhidden

        hidden_reg = self.hidden_size

        if self.len_concat:
            hidden_reg += 1
        
        if self.fa_concat:
            if self.hp.xFA[1][1] is None:
                hidden_reg += self.hp.xFA[1][0]
            else:
                hidden_reg += self.hp.xFA[1][1]

        self.reg_T10 = nn.Sequential(nn.Dropout(self.dropout_p),
                                     nn.Linear(int(hidden_reg), 200),
                                     nn.GELU(),
                                     nn.Dropout(self.dropout_p),
                                     nn.Linear(200, 200),
                                     nn.GELU(),
                                     nn.Linear(200, 1))
    
    # @jit.script_method
    def forward(self, X_fa_in, fa_vals_in, fa_mask, fa_len, TR_vals):

        mean = torch.sum(X_fa_in, dim=1) / fa_len.squeeze()
        X_fa_in = X_fa_in / mean.unsqueeze(dim=1)
        
        X_fa = push_zeros(X_fa_in.clone(), device=self.device)
        fa_vals = push_zeros(fa_vals_in.clone(), device=self.device)

        last_X = X_fa[
            torch.arange(X_fa.size(0)),
            fa_len.squeeze().long()-1].unsqueeze(-1).expand(*X_fa.size())
        
        last_fa = fa_vals_in[
            torch.arange(X_fa_in.size(0)),
            fa_len.squeeze().long()-1].unsqueeze(-1).expand(*X_fa.size())
        
        X_fa = torch.where(X_fa == 0., last_X, X_fa)
        fa_vals = torch.where(fa_vals == 0., last_fa, fa_vals)
        
        x = torch.cat((
            fa_vals.unsqueeze(dim=-1),
            X_fa.unsqueeze(dim=-1)),
            dim=-1)
        
        # waarom moet dit ?????
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)

        # ik snap ook niet wat hier gebeurd? 
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        X0 = X.evaluate(X.interval[0]).to(self.device)
        z0 = self.initial(X0)

        # waarom moet ik overal een float van maken?? 
        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.ode_net,
                              t=X.interval)

        z_T = z_T[:, 1]
        
        # regress T10 value
        T10 = self.reg_T10(z_T).squeeze()
        # # M0 = self.reg_M0(hidden_M0).squeeze()

        # # update T10
        T10_diff = self.T10_upper - self.T10_under
        T10 = self.T10_under + torch.sigmoid(T10.unsqueeze(1)) * T10_diff

        R1 = 1 / T10
        X_out = torch.mul(
            (1 - torch.exp(torch.mul(-TR_vals, R1))), torch.sin(fa_vals_in)) / (
            1 - torch.mul(torch.cos(fa_vals_in),
                          torch.exp(torch.mul(-TR_vals, R1))))
       
        mean = torch.sum(X_out, dim=1) / fa_len.squeeze()
        X_out = X_out / mean.unsqueeze(dim=1)

        M0 = torch.ones_like(T10)
        return X_out, T10, M0



# # class Encoder(nn.Module):
# class Encoder(jit.ScriptModule):
#     def __init__(self, hp):
#         super(Encoder, self).__init__()
        
#         # network parameters
#         self.hp = hp
#         self.len_concat = hp.len_concat
#         self.fa_concat = hp.fa_concat
#         self.learn_emb = hp.learn_emb
#         self.embed_time = hp.embed_time
#         self.dim = hp.input_dim
#         self.device = hp.device
#         self.nhidden = hp.n_hidden
#         self.num_heads = hp.n_heads
#         self.gru_layers = hp.gru_layers
#         self.dropout_p = hp.dropout_p
#         self.gru_bi = hp.gru_bi
#         self.mh_att = hp.mh_attention
#         self.one_reg = hp.one_reg
#         self.ref_angle = hp.ref_angle
#         self.layer_bool = hp.layer_bool
#         self.T10_under = hp.simulations.T10bounds[0]
#         self.T10_upper = hp.simulations.T10bounds[1]
    
#         # rnn layer
#         self.rnn = nn.GRU(input_size=1, 
#                           hidden_size=self.nhidden,
#                           num_layers=1,
#                           dropout=0,
#                           bidirectional=self.gru_bi,
#                           batch_first=True) 

#         # uni/bi directional
#         if self.gru_bi:
#             # hidden_reg = 2*self.nhidden
#             hidden_reg = 2*self.nhidden
#         else:
#             hidden_reg = self.nhidden
        
#         self.score_T10 = nn.Sequential(nn.Linear(hidden_reg, 1,
#                                                  bias=False),
#                                         nn.Softmax(dim=1))

        

#         if self.len_concat:
#             hidden_reg += 1
        
#         if self.fa_concat:
#             if self.hp.xFA[1][1] is None:
#                 hidden_reg += self.hp.xFA[1][0]
#             else:
#                 hidden_reg += self.hp.xFA[1][1]
        
#         self.reg_T10 = nn.Sequential(nn.Dropout(self.dropout_p),
#                                      nn.Linear(int(hidden_reg), 200),
#                                      nn.GELU(),
#                                      nn.Dropout(self.dropout_p),
#                                      nn.Linear(200, 200),
#                                      nn.GELU(),
#                                      nn.Linear(200, 1))

#     @jit.script_method
#     def forward(self, X_fa, fa_vals, fa_mask, fa_len, TR_vals):
#         # adjusting input
#         # in_shapes = X_fa_in.size()
    
#         # print(X_fa.size())
#         # input()
    
#         # out, _ = self.rnn(X_fa)
        
#         # # print(out.size())
#         # # input()
#         score_T1 = self.score_T10(X_fa)
#         # # print(score_T1.size())
#         # # input()

#         hidden_T10 = torch.sum(X_fa*score_T1, dim=1)
#         # # print(hidden_T10.size())
#         # # input()
        
#         # concat T1 map len
#         if self.len_concat:
#             hidden_T10 = torch.cat((hidden_T10, fa_len), dim=1)
#             # hidden_M0 = torch.cat((hidden_M0, fa_len), dim=1)
            
#         # regress T10 value
#         T10 = self.reg_T10(hidden_T10).squeeze()
#         # # M0 = self.reg_M0(hidden_M0).squeeze()

#         # # update T10
#         T10_diff = self.T10_upper - self.T10_under
#         T10 = self.T10_under + torch.sigmoid(T10.unsqueeze(1)) * T10_diff

#         R1 = 1 / T10
#         X_out = torch.mul(
#             (1 - torch.exp(torch.mul(-TR_vals, R1))), torch.sin(fa_vals)) / (
#             1 - torch.mul(torch.cos(fa_vals),
#                           torch.exp(torch.mul(-TR_vals, R1))))
       
#         mean = torch.sum(X_out, dim=1) / fa_len.squeeze()
#         X_out = X_out / mean.unsqueeze(dim=1)

#         M0 = torch.ones_like(T10)
#         return X_out, T10, M0


def push_zeros(a, device):
    valid_mask = a!=0
    flipped_mask = torch.fliplr(
        torch.sum(valid_mask, dim=1, keepdim=True)
        > torch.arange(a.shape[1]-1, -1, -1).to(device))
    a[flipped_mask] = a[valid_mask]
    a[~flipped_mask] = 0
    return a

"""
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
"""