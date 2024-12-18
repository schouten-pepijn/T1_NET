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
matplotlib.use('TkAgg')

# np.random.seed(42)
# torch.manual_seed(42)
class DCE_transformer(nn.Module):
    def __init__(self, hp):
        super(DCE_transformer, self).__init__()
        self.hp = hp
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=12,
                                                        nhead=3,
                                                        batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         num_layers=2)


        hidden_dim = 12
        self.score_ke = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))
        self.score_ve = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))
        self.score_vp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))
        self.score_dt = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))

         

        self.encoder_ke = nn.Sequential(nn.Linear(hidden_dim, int((hidden_dim)/2)),
                                        #nn.BatchNorm1d(int((self.hp.network.layers[0]+1)/2)),
                                        nn.ELU(),
                                        nn.Linear(int((hidden_dim)/2), 1)
                                        )
        self.encoder_ve = nn.Sequential(nn.Linear(hidden_dim, int((hidden_dim)/2)),
                                        #nn.BatchNorm1d(int((self.hp.network.layers[0]+1)/2)),
                                        nn.ELU(),
                                        nn.Linear(int((hidden_dim)/2), 1)
                                        )
        self.encoder_vp = nn.Sequential(nn.Linear(hidden_dim, int((hidden_dim)/2)),
                                        #nn.BatchNorm1d(int((self.hp.network.layers[0]+1)/2)),
                                        nn.ELU(),
                                        nn.Linear(int((hidden_dim)/2), 1)
                                        )
        self.encoder_dt = nn.Sequential(nn.Linear(hidden_dim, int((hidden_dim)/2)),
                                        #nn.BatchNorm1d(int((self.hp.network.layers[0]+1)/2)),
                                        nn.ELU(),
                                        nn.Linear(int((hidden_dim)/2), 1)
                                        )

        self.enc_all = nn.Sequential(nn.Linear(hidden_dim, int(hidden_dim/2)),
                                     nn.ELU(),
                                     nn.Linear(int(hidden_dim/2), 4)
                                     )

    def forward(self, X, aif):
        
        X = X.unsqueeze(dim=2)
        N, seq_len, z = X.size()
        
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.hp.device)
        positions = positions.unsqueeze(dim=2)

        div_term = torch.exp(positions * (-math.log(10000.0) / N))
        
        positions = torch.sin(positions * div_term)
        
        # print(torch.min(positions), torch.max(positions))
        # input('')
        
        X += positions
    
        X = torch.cat(tuple(X for _ in range(12)), dim=2)

        output = self.transformer_encoder(X)
        
        
        
        # output = torch.sum(output, dim=1)
        
        # output = self.enc_all(output)
        
        # ke = output[:, 0]
        # ve = output[:, 1]
        # vp = output[:, 2]
        # dt = output[:, 3]

        # print(output.size())
        # input()
        score_ke = self.score_ke(output)
        score_ve = self.score_ve(output)
        score_vp = self.score_vp(output)
        score_dt = self.score_dt(output)

        hidden_ke = torch.sum(output*score_ke, dim=1)
        hidden_ve = torch.sum(output*score_ve, dim=1)
        hidden_vp = torch.sum(output*score_vp, dim=1)
        hidden_dt = torch.sum(output*score_dt, dim=1)

            
        ke = self.encoder_ke(hidden_ke).squeeze()
        ve = self.encoder_ve(hidden_ve).squeeze()
        vp = self.encoder_vp(hidden_vp).squeeze()
        dt = self.encoder_dt(hidden_dt).squeeze()

        ke_diff = self.hp.simulations.bounds[1, 0] - self.hp.simulations.bounds[0, 0]
        ve_diff = self.hp.simulations.bounds[1, 1] - self.hp.simulations.bounds[0, 1]
        vp_diff = self.hp.simulations.bounds[1, 2] - self.hp.simulations.bounds[0, 2]
        dt_diff = self.hp.simulations.bounds[1, 3] - self.hp.simulations.bounds[0, 3]

        ke = self.hp.simulations.bounds[0, 0] + torch.sigmoid(ke.unsqueeze(1)) * ke_diff
        ve = self.hp.simulations.bounds[0, 1] + torch.sigmoid(ve.unsqueeze(1)) * ve_diff
        vp = self.hp.simulations.bounds[0, 2] + torch.sigmoid(vp.unsqueeze(1)) * vp_diff
        dt = self.hp.simulations.bounds[0, 3] + torch.sigmoid(dt.unsqueeze(1)) * dt_diff
        
        t_dce = torch.FloatTensor(self.hp.acquisition.timing).to(self.hp.device)

        aif = aif.to(self.hp.device)

        C_dw = functions.Cosine8AIF_ExtKety_pepijn(t_dce, aif, ke, dt, ve, vp, self.hp.device)
        
        return C_dw, ke, dt, ve, vp
    

class DCE_NET(nn.Module):
    def __init__(self, hp):
        super(DCE_NET, self).__init__()
        self.hp = hp

        self.hp.network.layers = [32, 4]          


        self.rnn = nn.GRU(1, self.hp.network.layers[0], self.hp.network.layers[1], batch_first=True)
        hidden_dim = self.hp.network.layers[0]


        self.score_ke = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))
        self.score_ve = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))
        self.score_vp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))
        self.score_dt = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))

         

        self.encoder_ke = nn.Sequential(nn.Linear(hidden_dim, int((hidden_dim)/2)),
                                        #nn.BatchNorm1d(int((self.hp.network.layers[0]+1)/2)),
                                        nn.ELU(),
                                        nn.Linear(int((hidden_dim)/2), 1)
                                        )
        self.encoder_ve = nn.Sequential(nn.Linear(hidden_dim, int((hidden_dim)/2)),
                                        #nn.BatchNorm1d(int((self.hp.network.layers[0]+1)/2)),
                                        nn.ELU(),
                                        nn.Linear(int((hidden_dim)/2), 1)
                                        )
        self.encoder_vp = nn.Sequential(nn.Linear(hidden_dim, int((hidden_dim)/2)),
                                        #nn.BatchNorm1d(int((self.hp.network.layers[0]+1)/2)),
                                        nn.ELU(),
                                        nn.Linear(int((hidden_dim)/2), 1)
                                        )
        self.encoder_dt = nn.Sequential(nn.Linear(hidden_dim, int((hidden_dim)/2)),
                                        #nn.BatchNorm1d(int((self.hp.network.layers[0]+1)/2)),
                                        nn.ELU(),
                                        nn.Linear(int((hidden_dim)/2), 1)
                                        )

    def forward(self, X, aif, Hct=None, first=False, epoch=0):

        X = X.unsqueeze(dim=2)
            
        output, hn = self.rnn(X)

        score_ke = self.score_ke(output)
        score_ve = self.score_ve(output)
        score_vp = self.score_vp(output)
        score_dt = self.score_dt(output)

        hidden_ke = torch.sum(output*score_ke, dim=1)
        hidden_ve = torch.sum(output*score_ve, dim=1)
        hidden_vp = torch.sum(output*score_vp, dim=1)
        hidden_dt = torch.sum(output*score_dt, dim=1)

            
        ke = self.encoder_ke(hidden_ke).squeeze()
        ve = self.encoder_ve(hidden_ve).squeeze()
        vp = self.encoder_vp(hidden_vp).squeeze()
        dt = self.encoder_dt(hidden_dt).squeeze()

        ke_diff = self.hp.simulations.bounds[1, 0] - self.hp.simulations.bounds[0, 0]
        ve_diff = self.hp.simulations.bounds[1, 1] - self.hp.simulations.bounds[0, 1]
        vp_diff = self.hp.simulations.bounds[1, 2] - self.hp.simulations.bounds[0, 2]
        dt_diff = self.hp.simulations.bounds[1, 3] - self.hp.simulations.bounds[0, 3]

        ke = self.hp.simulations.bounds[0, 0] + torch.sigmoid(ke.unsqueeze(1)) * ke_diff
        ve = self.hp.simulations.bounds[0, 1] + torch.sigmoid(ve.unsqueeze(1)) * ve_diff
        vp = self.hp.simulations.bounds[0, 2] + torch.sigmoid(vp.unsqueeze(1)) * vp_diff
        dt = self.hp.simulations.bounds[0, 3] + torch.sigmoid(dt.unsqueeze(1)) * dt_diff
        
        t_dce = torch.FloatTensor(self.hp.acquisition.timing).to(self.hp.device)

        aif = aif.to(self.hp.device)

        C_dw = functions.Cosine8AIF_ExtKety_pepijn(t_dce, aif, ke, dt, ve, vp, self.hp.device)

        return C_dw, ke, dt, ve, vp
        # return C_dw, ke_true, dt_true, ve_true, vp_true



class T10_NET(nn.Module):
    def __init__(self, hp):
        super(T10_NET, self).__init__()
        self.hp = hp
        
        self.TR_len_test = 0
        self.fa_len_test = 0
        
        self.hp.network.layers = [32, 4]
        # fa_dim = hp.xFA[1][1] + 1
        
        # if self.hp.xTR_fa[0]:
        #     tr_dim = 1
        # else:
        #     tr_dim = 0
        
        if self.hp.xFA[0]:
            if self.hp.xFA[1][1] is None:
                fa_dim = self.hp.xFA[1][0]
                len_dim = 0
            else:
                fa_dim = self.hp.xFA[1][1]
                len_dim = 1
        else:
            fa_dim = 0
            len_dim = 0
            
        self.fa_dim = fa_dim
        tr_dim = 0
        
        # T10
        self.rnn_T10 = nn.GRU(1, self.hp.network.layers[0], self.hp.network.layers[1], batch_first=True)
        hidden_dim_T10 = self.hp.network.layers[0]
        
        self.score_T10 = nn.Sequential(nn.Linear(hidden_dim_T10, 1), nn.Softmax(dim=1))
        self.score_S0 = nn.Sequential(nn.Linear(hidden_dim_T10, 1), nn.Softmax(dim=1))

        # T10 (!)
        self.encoder_T10 = nn.Sequential(nn.Linear(hidden_dim_T10 + fa_dim + tr_dim + len_dim, int((hidden_dim_T10 + fa_dim + tr_dim + len_dim)/2)),
                                         #nn.BatchNorm1d(int((self.hp.network.layers[0]+1)/2)),
                                         nn.ELU(),
                                         nn.Linear(int((hidden_dim_T10 + fa_dim + tr_dim + len_dim)/2), 1)
                                         )
        self.encoder_S0 = nn.Sequential(nn.Linear(hidden_dim_T10 + fa_dim + tr_dim + len_dim, int((hidden_dim_T10 + fa_dim + tr_dim + len_dim)/2)),
                                         #nn.BatchNorm1d(int((self.hp.network.layers[0]+1)/2)),
                                         nn.ELU(),
                                         nn.Linear(int((hidden_dim_T10 + fa_dim + tr_dim + len_dim)/2), 1)
                                         )
        
        
    def forward(self, X_fa, fa_vals, fa_len, TR_vals,
                T10_true, S0_dce, X_dce, first=False, epoch=0):
        
        # if TR_vals.size(dim=1) != self.TR_len_test:
        #     print(' tr len ', TR_vals.size(dim=1))
        #     self.TR_len_test = TR_vals.size(dim=1)
            # input('')
      
        # print(fa_vals.size())
        # input('')

        output_fa, hn_fa = self.rnn_T10(X_fa.unsqueeze(2))
        # output, _ = nn.utils.rnn.pad_packed_sequence(output,
        #                                              batch_first=True)

        score_T10 = self.score_T10(output_fa)
        score_S0 = self.score_S0(output_fa)

        hidden_T10 = torch.sum(output_fa*score_T10, dim=1)
        hidden_S0 = torch.sum(output_fa*score_S0, dim=1)

        if self.hp.xFA[0]:
            if self.hp.xFA[1][1] is None:
                hidden_T10 = torch.cat((hidden_T10, fa_vals), dim=1)
                hidden_S0 = torch.cat((hidden_S0, fa_vals), dim=1)
            else:
                hidden_T10 = torch.cat((hidden_T10, fa_vals, fa_len), dim=1)
                hidden_S0 = torch.cat((hidden_S0, fa_vals, fa_len), dim=1)

        T10 = self.encoder_T10(hidden_T10).squeeze()
        S0_fa = self.encoder_S0(hidden_S0).squeeze()

        T10_diff = self.hp.simulations.T10bounds[1, 0] - self.hp.simulations.T10bounds[0, 0]
        S0_diff = self.hp.simulations.T10bounds[1, 1] - self.hp.simulations.T10bounds[0, 1]

        T10 = self.hp.simulations.T10bounds[0, 0] + torch.sigmoid(T10.unsqueeze(1)) * T10_diff
        S0_fa = self.hp.simulations.T10bounds[0, 1] + torch.sigmoid(S0_fa.unsqueeze(1)) * S0_diff
        
        # X, _ = nn.utils.rnn.pad_packed_sequence(X, batch_first=True,
        #                                         total_length=self.hp.xFA[1])
        
        # T10
        shapes_fa = X_fa[:, :self.fa_dim].squeeze().size()

        R1 = 1 / T10
        R1_in_fa = torch.tile(R1, (1, shapes_fa[1])).to(self.hp.device)
        TR_fa = torch.tile(TR_vals, (1, shapes_fa[1])).to(self.hp.device)
        FA_fa = fa_vals

        X_fa = torch.mul(
            (1 - torch.exp(torch.mul(-TR_fa, R1_in_fa))), torch.sin(FA_fa)) / (
            1 - torch.mul(torch.cos(FA_fa), torch.exp(torch.mul(-TR_fa, R1_in_fa))))
        X_fa *= S0_fa
                
        # DCE
        shapes_dce = X_dce.squeeze().size()
        S0_dce = S0_dce.unsqueeze(dim=1).repeat(1, shapes_dce[1])

        # R1_dce = R1.unsqueeze(dim=1).repeat(1, shapes_dce[1])
        
        R1_dce = R1.repeat(1, shapes_dce[1])
        TR_dce = torch.full((*shapes_dce,),
                            self.hp.acquisition.TR_dce,
                            device=self.hp.device)
        FA_dce = torch.full((*shapes_dce,),
                            self.hp.acquisition.FA_dce,
                            device=self.hp.device)
        
        A = torch.divide(X_dce, S0_dce)
        E0 = torch.exp(-torch.mul(R1_dce, TR_dce))
        E = (1 - A + torch.mul(A, E0) - torch.mul(E0, torch.cos(FA_dce))) / (
            1 - torch.mul(A, torch.cos(FA_dce)) + torch.mul(A, torch.mul(E0, torch.cos(FA_dce))) - torch.mul(E0, torch.cos(FA_dce)))

        R1eff_dce = torch.mul(-1 / TR_dce, torch.log(E))
        C_dce = (R1eff_dce - R1_dce) / self.hp.acquisition.r1
        C_dce[torch.isnan(C_dce)] = 0

        return X_fa, C_dce, T10, S0_fa


class T10_transformer(nn.Module):
    def __init__(self, hp):
        super(T10_NET, self).__init__()
        self.hp = hp
        
        self.TR_len_test = 0
        self.fa_len_test = 0
        
        self.hp.network.layers = [32, 4]
        # fa_dim = hp.xFA[1][1] + 1
        
        # if self.hp.xTR_fa[0]:
        #     tr_dim = 1
        # else:
        #     tr_dim = 0
        
        if self.hp.xFA[0]:
            if self.hp.xFA[1][1] is None:
                fa_dim = self.hp.xFA[1][0]
                len_dim = 0
            else:
                fa_dim = self.hp.xFA[1][1]
                len_dim = 1
        else:
            fa_dim = 0
            len_dim = 0
            
        self.fa_dim = fa_dim
        tr_dim = 0
        
        # T10
        self.rnn_T10 = nn.GRU(1, self.hp.network.layers[0], self.hp.network.layers[1], batch_first=True)
        hidden_dim_T10 = self.hp.network.layers[0]
        
        self.score_T10 = nn.Sequential(nn.Linear(hidden_dim_T10, 1), nn.Softmax(dim=1))
        self.score_S0 = nn.Sequential(nn.Linear(hidden_dim_T10, 1), nn.Softmax(dim=1))

        # T10 (!)
        self.encoder_T10 = nn.Sequential(nn.Linear(hidden_dim_T10 + fa_dim + tr_dim + len_dim, int((hidden_dim_T10 + fa_dim + tr_dim + len_dim)/2)),
                                         #nn.BatchNorm1d(int((self.hp.network.layers[0]+1)/2)),
                                         nn.ELU(),
                                         nn.Linear(int((hidden_dim_T10 + fa_dim + tr_dim + len_dim)/2), 1)
                                         )
        self.encoder_S0 = nn.Sequential(nn.Linear(hidden_dim_T10 + fa_dim + tr_dim + len_dim, int((hidden_dim_T10 + fa_dim + tr_dim + len_dim)/2)),
                                         #nn.BatchNorm1d(int((self.hp.network.layers[0]+1)/2)),
                                         nn.ELU(),
                                         nn.Linear(int((hidden_dim_T10 + fa_dim + tr_dim + len_dim)/2), 1)
                                         )
        
        
    def forward(self, X_fa, fa_vals, fa_len, TR_vals,
                T10_true, S0_dce, X_dce, first=False, epoch=0):
        
        # if TR_vals.size(dim=1) != self.TR_len_test:
        #     print(' tr len ', TR_vals.size(dim=1))
        #     self.TR_len_test = TR_vals.size(dim=1)
            # input('')
      
        # print(fa_vals.size())
        # input('')

        output_fa, hn_fa = self.rnn_T10(X_fa.unsqueeze(2))
        # output, _ = nn.utils.rnn.pad_packed_sequence(output,
        #                                              batch_first=True)

        score_T10 = self.score_T10(output_fa)
        score_S0 = self.score_S0(output_fa)

        hidden_T10 = torch.sum(output_fa*score_T10, dim=1)
        hidden_S0 = torch.sum(output_fa*score_S0, dim=1)

        if self.hp.xFA[0]:
            if self.hp.xFA[1][1] is None:
                hidden_T10 = torch.cat((hidden_T10, fa_vals), dim=1)
                hidden_S0 = torch.cat((hidden_S0, fa_vals), dim=1)
            else:
                hidden_T10 = torch.cat((hidden_T10, fa_vals, fa_len), dim=1)
                hidden_S0 = torch.cat((hidden_S0, fa_vals, fa_len), dim=1)

        T10 = self.encoder_T10(hidden_T10).squeeze()
        S0_fa = self.encoder_S0(hidden_S0).squeeze()

        T10_diff = self.hp.simulations.T10bounds[1, 0] - self.hp.simulations.T10bounds[0, 0]
        S0_diff = self.hp.simulations.T10bounds[1, 1] - self.hp.simulations.T10bounds[0, 1]

        T10 = self.hp.simulations.T10bounds[0, 0] + torch.sigmoid(T10.unsqueeze(1)) * T10_diff
        S0_fa = self.hp.simulations.T10bounds[0, 1] + torch.sigmoid(S0_fa.unsqueeze(1)) * S0_diff
        
        # X, _ = nn.utils.rnn.pad_packed_sequence(X, batch_first=True,
        #                                         total_length=self.hp.xFA[1])
        
        # T10
        shapes_fa = X_fa[:, :self.fa_dim].squeeze().size()

        R1 = 1 / T10
        R1_in_fa = torch.tile(R1, (1, shapes_fa[1])).to(self.hp.device)
        TR_fa = torch.tile(TR_vals, (1, shapes_fa[1])).to(self.hp.device)
        FA_fa = fa_vals

        X_fa = torch.mul(
            (1 - torch.exp(torch.mul(-TR_fa, R1_in_fa))), torch.sin(FA_fa)) / (
            1 - torch.mul(torch.cos(FA_fa), torch.exp(torch.mul(-TR_fa, R1_in_fa))))
        X_fa *= S0_fa
                
        # DCE
        shapes_dce = X_dce.squeeze().size()
        S0_dce = S0_dce.unsqueeze(dim=1).repeat(1, shapes_dce[1])

        # R1_dce = R1.unsqueeze(dim=1).repeat(1, shapes_dce[1])
        
        R1_dce = R1.repeat(1, shapes_dce[1])
        TR_dce = torch.full((*shapes_dce,),
                            self.hp.acquisition.TR_dce,
                            device=self.hp.device)
        FA_dce = torch.full((*shapes_dce,),
                            self.hp.acquisition.FA_dce,
                            device=self.hp.device)
        # TR_dce = torch.FloatTensor(
        #     np.repeat(
        #         [self.hp.acquisition.TR_dce],
        #         shapes_dce[0]*shapes_dce[1]).reshape(*shapes_dce)).to(self.hp.device)
        # FA_dce = torch.FloatTensor(
        #     np.repeat(
        #         [self.hp.acquisition.FA_dce],
        #         shapes_dce[0]*shapes_dce[1]).reshape(*shapes_dce)).to(self.hp.device)
        
        # S0_dce = torch.mean(X_dce[:, :160//10], axis=1).unsqueeze(dim=1).repeat(1, shapes_dce[1])
        # E0 = torch.exp(-1.0 * R1_dce * TR_dce)
        # A =     (((S0_dce * E0 * torch.cos(FA_dce) - S0_dce) / torch.sin(FA_dce)) / (E0 - 1.0))
        # sin_fa_s0 = torch.sin(FA_dce) * A
        # a = sin_fa_s0 - X_dce
        # b = sin_fa_s0 - torch.cos(FA_dce) * X_dce
        # c = sin_fa_s0 - torch.cos(FA_dce) * S0_dce
        # d = sin_fa_s0 - S0_dce
        
        # R1eff_dce = -1.0 * torch.log((a / b) * (c / d)) / TR_dce

        # C_dce = R1eff_dce /  self.hp.acquisition.r1
        
        A = torch.divide(X_dce, S0_dce)
        E0 = torch.exp(-torch.mul(R1_dce, TR_dce))
        E = (1 - A + torch.mul(A, E0) - torch.mul(E0, torch.cos(FA_dce))) / (
            1 - torch.mul(A, torch.cos(FA_dce)) + torch.mul(A, torch.mul(E0, torch.cos(FA_dce))) - torch.mul(E0, torch.cos(FA_dce)))

        R1eff_dce = torch.mul(-1 / TR_dce, torch.log(E))
        C_dce = (R1eff_dce - R1_dce) / self.hp.acquisition.r1
        C_dce[torch.isnan(C_dce)] = 0

        return X_fa, C_dce, T10, S0_fa

def load_optimizer(net, hp):
    if hp.training.optim == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=hp.training.lr, weight_decay=hp.training.weight_decay)
    elif hp.training.optim == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=hp.training.lr, momentum=0.9, weight_decay=hp.training.weight_decay)
    elif hp.training.optim == 'adagrad':
        optimizer = torch.optim.Adagrad(net.parameters(), lr=hp.training.lr, weight_decay=hp.training.weight_decay)
    else:
        raise Exception(
            'No valid optimiser is chosen. Please select a valid optimiser: training.optim = ''adam'', ''sgd'', ''adagrad''')

    scheduler_1 = optim.lr_scheduler.LinearLR(optimizer, start_factor=hp.training.warm_up_mult,
                                              end_factor=1, total_iters=hp.training.warm_up_epoch, verbose=True)

    scheduler_2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=hp.training.lr_mult_down,
                                                       patience=hp.training.optim_patience_down, verbose=True)

  

    return optimizer, scheduler_1, scheduler_2