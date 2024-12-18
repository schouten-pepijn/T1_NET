'''
Mar 2018 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/deep-learning/ivim
'''
import torch
import math
import numpy as np


class network_training_hyper_parameters:
    def __init__(self):
        self.lr = 1e-4  # was 1e-4
        self.lr_mult_down = 0.1
        self.T10_epochs = 80 # was 200
        self.DCE_epochs = 50
        self.optim = 'adam'  # adam 0.0001; sgd 0.1
        self.patience = 5
        self.optim_patience_down = 2  # 0 is disabled
        self.warm_up_mult = 0.5
        self.warm_up_epoch = 1
        self.batch_size = 128  # was 1024
        self.val_batch_size = 512  # was 1280
        self.split = 0.95
        self.totalit = 2000  # was 1000
        self.save_train_fig = True
        self.weight_decay = 0


class network_building_hyper_parameters:
    def __init__(self):
        self.dropout = 0
        self.nn = 'gru'  # ['linear', 'convlin', 'lstm']
        self.layers = [32, 4]


class simulation_hyper_parameters:
    def __init__(self):
        self.num_samples = 200000  # was 2000000
        self.num_samples_leval = 5000
        self.data_length = 160
        self.vp_min = 0.01  # was 0.001
        self.vp_max = 0.04  # was 0.06
        self.ve_min = 0.01  # was 0.01
        self.ve_max = 0.6  # was 0.8
        self.kep_min = 0.04  # was 0.04
        self.kep_max = 2.   # was 3
        self.R1_min = 1/3  # was 1/3
        self.R1_max = 1/0.2  # was 1/0.2
        self.time = 1.75 # 1.632 - 2.894
        self.dt_min = self.time / 60 * 5
        self.dt_max = self.time / 60 * 8
        self.what_to_sim = "nn"  # T1fit, lsq or nn
        self.plot = True

        self.T10bounds = torch.FloatTensor([1e-6, 4])
        M0_low, M0_high = (100, 5000)
        self.M0simulation = torch.FloatTensor([M0_low, M0_high])
        # self.M0bounds = torch.FloatTensor([0.9*M0_low, 1.1*M0_high])
        self.M0bounds = torch.FloatTensor([80, 5500])
        # self.M0bounds = torch.FloatTensor([1e-6, 500])
        
        # self.M0bounds = torch.FloatTensor([1e-6,
        #                                     2])

        self.bounds = torch.FloatTensor(((1e-8, 1e-6, 1e-6, 1e-2),
                                         (3, 0.8, 0.07, 1.5)
                                         ))  # ke, ve, vp, dt, ((min), (max))


class acquisition_parameters:
    def __init__(self):
        self.r1 = 5
        self.TR_dce = 3.2e-3
        self.FA_dce = math.radians(20)
        self.TR_fa = 3.8e-3  # T1 map was 2.8e-3
        # self.FA3 = [math.radians(i) for i in range(1, 40, 2)]  # T1 map
        # self.FA3 = [math.radians(i) for i in (2, 5, 10, 15, 20, 25)]  # T1 map
        self.FA3 = [math.radians(i) for i in (2, 8, 14, 20)]
        # self.FA3 = [math.radians(i) for i in (2, 10, 15, 20, 25, 30)]
        # self.FA3 = [math.radians(i) for i in (3, 7, 12, 16, 17, 21, 26, 32)]
        self.rep0 = 10
        self.rep1 = 1
        self.rep2 = 161


class AIF_parameters:
    def __init__(self):
        self.Hct = 0.40
        self.aif = {'ab': 7.9785,
                    'ae': 0.5216,
                    'ar': 0.0482,
                    'mb': 32.8855,
                    'me': 0.1811,
                    'mm': 9.1868,
                    'mr': 15.8167,
                    't0': 0,  # 0.4307
                    'tr': 0.2533}


class Hyperparams:
    def __init__(self):
        '''Hyperparameters'''
        self.create_name = 'simulations_data.p'
        self.supervised = False
        self.max_rep = 160
        # main
        self.training = network_training_hyper_parameters()
        self.network = network_building_hyper_parameters()
        self.simulations = simulation_hyper_parameters()
        self.acquisition = acquisition_parameters()
        self.aif = AIF_parameters()
        self.use_cuda = True
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.jobs = 14
        self.out_fold = 'results'
        
        self.pretrained = True

        # self.lsq_T10_bounds = ([0, self.simulations.T10bounds[0][0]],
                               # [np.inf, self.simulations.T10bounds[1][0]])

        self.generate_data = False
        self.save = False
        self.train_data_path = '/scratch/pschouten/SYNTH_DATA/T10_net/M0_var/6ssFA_80_snr_2_ref_angle'
        self.second_val_set = False
        self.val_data_path = '/scratch/pschouten/SYNTH_DATA/T10_net/[4_6]FA_no_snr_test'
        self.second_test_set = False
        # self.test_data_path = '/scratch/pschouten/SYNTH_DATA/T10_net/M0_var/4sFA_10_100_snr_2_ref_angle_train'
        self.test_data_path = '/scratch/pschouten/SYNTH_DATA/T10_net/M0_var/46sFA_10_100_snr_2_ref_angle'

        # mTAN architecture
        self.input_dim = 1
        self.learn_emb = True
        self.embed_time = 128  # was 128
        self.n_hidden = 50  # was 64
        self.n_heads = 1  # was 2
        self.gru_bi = True
        self.gru_layers = 4
        self.dropout_p = 0.
        self.layer_bool = False
        self.mh_attention = True
        self.one_reg = False
        self.ref_angle_sim = True
        self.ref_angle = False
        self.eps = 1e-9

        self.layer_norm_in = False
        self.norm = 'None'  # normalization / standardization / None

        self.len_concat = False
        self.fa_concat = False
        self.time_embedding_test = False
        self.v_lin = False

        self.sup = False

        self.chunked = True

        self.SNR = [
            None,
            [10, 50]]  # on  min/max (7, 100)

        self.xFA = [
            True,
            [4, 6],
            [1, 35]]  # on, len:min/max(None), tot:min/max/step, shuffle, chunked
