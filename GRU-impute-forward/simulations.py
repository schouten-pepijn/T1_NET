import torch
import torch.utils.data as utils
import torch.nn as nn
import processing.pros as pros
import DCE_matt as dce
import pydcemri.dcemri as classic
import numpy as np
import train
import os
import copy
import pickle
from tqdm import tqdm
from collections import namedtuple
from contextlib import redirect_stdout
import mTAN as trans
# bland altman
import matplotlib.pyplot as plt
import statsmodels.api as sm
# T10 net
import more_itertools as mit
# lsq
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
# qq pot
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot_2samples
# pretrained
import model
# histogram
import seaborn as sns
# MAE, MSe, RMSE, RMSLE, R squared
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
# density plots
from scipy.interpolate import interpn
# fa full
import math
import transformer
# np.random.seed(42)
# torch.manual_seed(42)

def run_simulations(hp, eval=False, var_seq=False):
    print(hp.device)
    if hp.generate_data:
        # T10 data
        print(f'generating T1 data #{hp.simulations.num_samples}')
        params_T1_true = generate_T1_data(hp)
    else:
        print('loading trainng data from  disk')
        print(f'path: {hp.train_data_path}')
        params_T1_true = load_numpy(hp.train_data_path)
        hp.xFA[1][1] = params_T1_true.fa.shape[1]

    # if hp.second_test_set:
    #     params_T1_test = load_numpy(hp.test_data_path)

    # if hp.second_val_set:
    #     params_T1_val = load_numpy(hp.val_data_path)
    # else:
    #     params_T1_val = None
    
    """
    T10 net training
    """
    if hp.pretrained:
        print('evaluating on pretrained T10-transformer')
        file_T10 = next(os.walk('pretrained', (None, None, [])))[2]
        net_T10 = trans.T10_transformer(copy.deepcopy(hp)).to(hp.device)
        net_T10.load_state_dict(torch.load(
            os.path.join('pretrained', file_T10[0])))
        net_T10.to(hp.device)
    else:
        print('training T10-transformer')
        net_T10 = train.train_T1_transformer(hp, params_T1_true)
        torch.save(net_T10.state_dict(), 'pretrained/pretrained.pt')

    # make T10 prediction
    print('predicting T10 map')
    # if hp.second_test_set:
    #     params_T1_pred = predict_T10(hp, net_T10, params_T1_test)
    #     params_T1_true = params_T1_test 
    #     # params_T1_pred = predict_T10(hp, net_T10, params_T1_true)
    # else:
    params_T1_pred = predict_T10(hp, net_T10, params_T1_true)

    print('generating T10 statistics')
    np.save('results/X_fa_pred.npy', params_T1_pred.X_fa)
    np.save('results/T10_pred.npy', params_T1_pred.T10)

    T10_errors_net, X_fa_errors_net = sim_T10_results(params_T1_true.X_fa,
                                                      params_T1_true.T10,
                                                      params_T1_pred.X_fa,
                                                      params_T1_pred.T10)

    x_x_plot(params_T1_true.T10, params_T1_pred.T10, 'True', 'Pred',
             'true vs net', 'T10_true_vs_net', s=5, marker='.')
    
    bland_altman_one(params_T1_true.T10,
                      params_T1_pred.T10,
                      'true vs net',
                      os.path.join(f'{hp.out_fold}',
                                    'T10_bland_altman_T10net.png'),
                      s=5, marker='.')

    diff_net_T10 = np.abs(params_T1_true.T10.squeeze() - params_T1_pred.T10)
    q_95 = np.quantile(diff_net_T10, 0.95)

    histogram(
        diff_net_T10[diff_net_T10 < q_95],
        f'T10_hist_err_q_{q_95:.3f}_T10net.png',
        f'T10 net q95={q_95:.2f}',
        stat='density')

    histogram(
        diff_net_T10,
        'T10_hist_err_T10net.png',
        'T10 net',
        stat='density')


def predict_T10(hp, net, params):
    net.eval()

    indices = np.arange(len(params.X_fa), dtype=int)
    inferloader = utils.DataLoader(indices,
                                   batch_size=hp.training.val_batch_size,
                                   shuffle=False,
                                   drop_last=False)

    # perform inference
    X_fa_out = torch.zeros((*params.X_fa.shape,))
    
    first = True
    with torch.no_grad():
        for i, X_idx in enumerate(tqdm(inferloader, position=0, leave=True), 0):            
            
            X_fa_batch = torch.FloatTensor(params.X_fa[X_idx, :])
            fa_batch = torch.FloatTensor(params.fa[X_idx, :])
            fa_mask_batch = torch.BoolTensor(params.fa_mask[X_idx, :])
            fa_len_batch = torch.FloatTensor(params.fa_len[X_idx, np.newaxis])
            TR_batch = torch.FloatTensor(params.TR[X_idx, np.newaxis])

            if hp.norm == 'normalization':
                # min-max normalization
                X_fa_batch = (X_fa_batch - params.X_fa.min()) / \
                    (params.X_fa.max() - params.X_fa.min())

            X_fa_batch = X_fa_batch.to(hp.device)
            fa_batch = fa_batch.to(hp.device)
            fa_mask_batch = fa_mask_batch.to(hp.device)
            fa_len_batch = fa_len_batch.to(hp.device)
            TR_batch = TR_batch.to(hp.device)

            X_pred, T10_pred, M0_pred = net(X_fa_in=X_fa_batch,
                                            fa_vals=fa_batch,
                                            fa_mask=fa_mask_batch,
                                            fa_len=fa_len_batch,
                                            TR_vals=TR_batch)
            
            if first:
                X_out = X_pred.detach().cpu().numpy().squeeze()
                T10_out = T10_pred.detach().cpu().numpy().squeeze()
                M0_out = M0_pred.detach().cpu().numpy().squeeze()
                first = False
            else:
                X_pred = X_pred.detach().cpu().numpy().squeeze()
                T10_pred = T10_pred.detach().cpu().numpy().squeeze()
                M0_pred = M0_pred.detach().cpu().numpy().squeeze()

                X_out = np.concatenate((X_out, X_pred), axis=0)
                T10_out = np.concatenate((T10_out, T10_pred))
                M0_out = np.concatenate((M0_out, M0_pred))

    params_T1_pred_container = namedtuple('params',
                                          ['X_fa', 'T10', 'M0'])
    params_T1_pred = params_T1_pred_container(X_out, T10_out, M0_out)
    return params_T1_pred


def generate_T1_data(hp):
    rng = np.random.default_rng()
    
    M0 = rng.uniform(
        hp.simulations.M0simulation[0],
        hp.simulations.M0simulation[1],
        size=(hp.simulations.num_samples))
    
    T10 = rng.uniform(
        1/hp.simulations.R1_max, 1/hp.simulations.R1_min,
        size=(hp.simulations.num_samples))
    R1 = 1 / T10

    print(f'T10 | min: {np.min(T10):.4f} | max: {np.max(T10):.4f}')
    print(f'M0 | min: {np.min(M0):.4f} | max: {np.max(M0):.4f}')
    if hp.SNR[0] is True:
        SNR = np.linspace(hp.SNR[1][0], hp.SNR[1][1], num=hp.simulations.num_samples)
        print(f'SNR | min: {np.min(SNR)} | max: {np.max(SNR)}')
    elif hp.SNR[0] is False:
        print(f'SNR is {hp.SNR[1][1]}')
        SNR = hp.SNR[1][1] * np.ones(hp.simulations.num_samples)
    else:
        print('no SNR')
        SNR = np.zeros(hp.simulations.num_samples)

    TR_fa = hp.acquisition.TR_fa * np.ones(hp.simulations.num_samples)
    fa_idx = np.array([i-1 for i in range(*hp.xFA[2])])
    fa_full = np.array([math.radians(i) for i in range(*hp.xFA[2])])

    """
    test with 35
    """
    fa_ref = 1
    
    print(f'fa range: {np.min(fa_full)} to {np.max(fa_full)}')
    
    if hp.xFA[0]:
        if hp.xFA[1][1] is None:
            fa = np.zeros((hp.simulations.num_samples, len(fa_full)))
            # fa = np.zeros((hp.simulations.num_samples, 8))
            fa_len = np.repeat([hp.xFA[1][0]], hp.simulations.num_samples)
            
           
        else:
            # fa_len = rng.integers(hp.xFA[1][0], hp.xFA[1][1],
                                   # hp.simulations.num_samples, endpoint=False)
            fa_len = rng.choice([hp.xFA[1][0], hp.xFA[1][1]],
                                size=hp.simulations.num_samples)
            fa = np.zeros((hp.simulations.num_samples, len(fa_full)))
          
        if hp.chunked:
            splits = [np.array_split(fa_idx[fa_ref+1:28], fa_len[i]-1)
                               for i in range(hp.simulations.num_samples)]
    else:
        print(f'using fa: {hp.acquisition.FA3}')
        fa = np.zeros((hp.simulations.num_samples, len(fa_full)))
        fa_len = np.repeat([len(hp.acquisition.FA3)], hp.simulations.num_samples)
    
    X_fa = np.zeros_like(fa)

    """
    test 35 ref
    """
    
    for i in tqdm(range(len(fa))):   
        if hp.xFA[0]:
            if hp.chunked:
                idx = np.array([fa_ref] + [rng.choice(i) for i in splits[i]])
                # idx = np.insert(idx, 0, fa_ref, axis=0)
            else:
                # idx = np.asarray(mit.random_combination(fa_idx, fa_len[i]))
                # if fa_ref not in idx:
                #     idx[0] = fa_ref

                if fa_len[i] == 6:
                    idx = [1, 4, 9, 14, 19, 24]
                elif fa_len[i] == 4:
                    idx = [1, 7, 13, 19]
            
            # idx = [1, 20]
        else:
            idx = [int(i / np.pi * 180 - 1) for i in hp.acquisition.FA3]
            sel = rng.choice([0, 1])
            if sel == 0:
                idx = [1, 7, 13, 18]
            elif sel == 1:
                idx = [4, 9, 14, 19]    

        fa[i, idx] = fa_full[idx]
 
        X_temp = classic.r1eff_to_dce(R1[i], TR_fa[i], fa[i])
        X_max = X_temp.max()

        if hp.SNR[0] is not None:
            X_temp /= X_max
            noise = rng.normal(0, 1/SNR[i], len(fa_full))
            X_temp[idx] += noise[idx] 
            X_temp *= X_max

        X_temp *= M0[i]
        X_fa[i] = X_temp

    fa_mask = fa.astype(bool)

    print(f'X_fa | min: {np.min(X_fa):.4f} mean: {np.mean(X_fa):.4f} '\
          f'max: {np.max(X_fa):.4f}')

    params_container = namedtuple('params', ['X_fa', 'fa', 'fa_mask',
                                             'R1', 'T10', 'TR',
                                             'fa_len', 'M0', 'fa_full', 'fa_ref'])
    params = params_container(X_fa, fa, fa_mask, R1, T10,
                              TR_fa, fa_len, M0, fa_full, fa_ref)

    if hp.save:
        np.save('/scratch/pschouten/SYNTH_DATA/T10_net/X_fa.npy', X_fa)
        np.save('/scratch/pschouten/SYNTH_DATA/T10_net/fa.npy', fa)
        np.save('/scratch/pschouten/SYNTH_DATA/T10_net/fa_mask.npy', fa_mask)
        np.save('/scratch/pschouten/SYNTH_DATA/T10_net/R1.npy', R1)
        np.save('/scratch/pschouten/SYNTH_DATA/T10_net/T10.npy', T10)
        np.save('/scratch/pschouten/SYNTH_DATA/T10_net/TR.npy', TR_fa)
        np.save('/scratch/pschouten/SYNTH_DATA/T10_net/fa_len.npy', fa_len)
        np.save('/scratch/pschouten/SYNTH_DATA/T10_net/M0.npy', M0)
        np.save('/scratch/pschouten/SYNTH_DATA/T10_net/fa_full.npy', fa_full)
        
        plt.figure(dpi=300)
        plt.hist(T10)
        plt.title('T10 distribution')
        plt.gcf()
        plt.savefig('/scratch/pschouten/SYNTH_DATA/T10_net/T10_hist.png', format='png')
        
        plt.figure(dpi=300)
        plt.hist(R1)
        plt.title('R1 distribution')
        plt.gcf()
        plt.savefig('/scratch/pschouten/SYNTH_DATA/T10_net/R1_hist', format='png')
    return params


def load_numpy(data_path):
    data_container_T10 = namedtuple('params', ['X_fa', 'fa', 'fa_mask',
                                               'R1', 'T10', 'TR',
                                               'fa_len', 'M0'])
    params_T1_true = data_container_T10(
        np.load(os.path.join(data_path, 'X_fa.npy')),
        np.load(os.path.join(data_path, 'fa.npy')),
        np.load(os.path.join(data_path, 'fa_mask.npy')),
        np.load(os.path.join(data_path, 'R1.npy')),
        np.load(os.path.join(data_path, 'T10.npy')),
        np.load(os.path.join(data_path, 'TR.npy')),
        np.load(os.path.join(data_path, 'fa_len.npy')),
        np.load(os.path.join(data_path, 'M0.npy')))
    return params_T1_true


def sim_T10_results(X_true, T10_true, X_pred, T10_pred):
    X_true = X_true.squeeze()
    T10_true = T10_true.squeeze()
    X_pred = X_pred.squeeze()
    T10_pred = T10_pred.squeeze()
    
    # abs?
    error_T10 = T10_true - T10_pred
    randerror_T10 = np.std(error_T10)
    syserror_T10 = np.mean(error_T10)
    
    error_X_fa = X_true - X_pred
    randerror_X_fa = np.std(error_X_fa)
    syserror_X_fa = np.mean(error_X_fa)
    
    MAE_X_fa = mean_absolute_error(X_true, X_pred)
    MAE_T10 = mean_absolute_error(T10_true, T10_pred)
    
    MSE_X_fa = mean_squared_error(X_true, X_pred)
    MSE_T10 = mean_squared_error(T10_true, T10_pred)
    
    RB_X_fa = np.sum(X_pred - X_true) / np.sum(X_true)
    RB_T10 = np.sum(T10_pred - T10_true) / np.sum(T10_true)
    
    CV_X_fa_true = np.std(X_true) / np.mean(X_true)
    CV_X_fa_pred = np.std(X_pred) / np.mean(X_pred)
    CV_T10_true = np.std(T10_true) / np.mean(T10_true)
    CV_T10_pred = np.std(T10_pred) / np.mean(T10_pred)
    
    R2_X_fa = r2_score(X_true, X_pred) 
    R2_T10 = r2_score(T10_true, T10_pred)
    adjR2_T10 = 1 - (1 - R2_T10)*(len(T10_true) - 1)/(len(T10_true) - 1 - 1)
    adjR2_X_fa = 1 - (1 - R2_X_fa)*(len(X_true) - 1)/(len(X_true) - 1 - 1)

    quantiles_T10 = np.zeros(4)
    quantiles_X_fa = np.zeros(4)
    for i, q_x in enumerate((0.25, 0.5, 0.75, 0.95)):
        quantiles_T10[i] = np.quantile(np.abs(error_T10), q_x)
        quantiles_X_fa[i] = np.quantile(np.abs(error_X_fa), q_x)

    outliers_mask = (np.abs(error_T10) > quantiles_T10[-1])

    T10_pred_outliers = T10_pred[outliers_mask]
    T10_true_outliers = T10_true[outliers_mask]
    T10_outliers = np.array([T10_true_outliers, T10_pred_outliers])

    X_pred_outliers = X_pred[outliers_mask, :]
    X_true_outliers = X_true[outliers_mask, :]
    X_outliers = np.array([X_true_outliers, X_pred_outliers])

    error_container = namedtuple('errors', ['randerror', 'syserror',
                                            'MAE', 'MSE',
                                            'R2', 'adjR2',
                                            'quantiles', 'outliers',
                                            'RB', 'CV_true', 'CV_pred'])

    T10_errors = error_container(randerror_T10, syserror_T10,
                                 MAE_T10, MSE_T10,
                                 R2_T10, adjR2_T10,
                                 quantiles_T10,
                                 T10_outliers, RB_T10,
                                 CV_T10_true, CV_T10_pred)
    X_fa_errors = error_container(randerror_X_fa, syserror_X_fa,
                                  MAE_X_fa, MSE_X_fa,
                                  R2_X_fa, adjR2_X_fa,
                                  quantiles_X_fa,
                                  X_outliers, RB_X_fa,
                                  CV_X_fa_true, CV_X_fa_pred)

    with open(os.path.join('results', 'T10_net_errors.txt'), 'w') as f:
        with redirect_stdout(f):
            print(f'X_fa  | randerr: {X_fa_errors.randerror:11.8f}  syserr: {X_fa_errors.syserror:11.8f}')
            print(f'X_fa  |     MAE: {X_fa_errors.MAE:11.8f}     MSE: {X_fa_errors.MSE:11.8f}')
            print(f'X_fa  |      R2: {X_fa_errors.R2:11.8f}   adjR2: {X_fa_errors.adjR2:11.8f}')
            print(f'X_fa  |   rel.B: {X_fa_errors.RB:11.8f}')
            print(f'X_fa  | CV_true: {X_fa_errors.CV_true:11.8f} CV_pred: {X_fa_errors.CV_pred:11.8f}')
            print(f'X_fa  | quant.: {quantiles_X_fa}')
            print('')
            print(f'T10   | randerr: {T10_errors.randerror:11.8f} syserr: {T10_errors.syserror:11.8f}')
            print(f'T10   |     MAE: {T10_errors.MAE:11.8f}    MSE: {T10_errors.MSE:11.8f}')
            print(f'T10   |      R2: {T10_errors.R2:11.8f}  adjR2: {T10_errors.adjR2:11.8f}')
            print(f'X_fa  |   rel.B: {T10_errors.RB:11.8f}')
            print(f'X_fa  | CV_true: {T10_errors.CV_true:11.8f} CV_pred: {T10_errors.CV_pred:11.8f}')
            
            print(f'T10   | quant.: {quantiles_T10}')
    return T10_errors, X_fa_errors


def bland_altman_one(data1, data2, title, path, *args, **kwargs):
    data1 = np.asarray(data1).squeeze()
    data2 = np.asarray(data2).squeeze()
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    md = np.mean(diff)
    sd = np.std(diff, axis=0)
    
    plus_95 = md + 1.96*sd
    min_95 = md - 1.96*sd
    
    plt.figure(dpi=300)
    plt.title(title)
    plt.axhline(md, color='gray',
                linestyle='--', label= f'mean: {md:.2f}')
    plt.axhline(plus_95, color='gray',
                linestyle='-.', label=f'+95%: {plus_95:.2f}')
    plt.axhline(min_95, color='gray',
                linestyle='-.', label=f'-95%: {min_95:.2f}')
    plt.xlabel('average')
    plt.ylabel('difference')
    plt.legend()
    if 'c' in (kwargs):
        p = plt.scatter(mean, diff, *args, **kwargs)
        cbar = plt.colorbar(p)
    else:
        data, x_e, y_e = np.histogram2d(mean, diff, bins=100, density=True)
        z = interpn(( 0.5*(x_e[1:]+x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])),
                    data, np.vstack([mean, diff]).T, method='splinef2d',
                    bounds_error=False)
        z[np.where(np.isnan(z))] = 0.
        if np.min(z) < 0:
            z += np.abs(np.min(z))
        elif np.min(z) > 0:
            z -= np.min(z)
        idx = z.argsort()
        mean, diff, z = mean[idx], diff[idx], z[idx]
        p = plt.scatter(mean, diff, c=z, *args, **kwargs)
        cbar = plt.colorbar(p)
        cbar.set_label('scatter density')
    plt.gcf()
    plt.savefig(path, format='png')


def x_x_plot(x1, x2, x_lab, y_lab, tit, save, *args, **kwargs):
    x1 = x1.squeeze()
    x2 = x2.squeeze()
    
    max_45 = np.max(x1)
    min_45 = np.min(x1)
    line_45 = [min_45, max_45]
    
    plt.figure(dpi=300)
    plt.plot(line_45, line_45, color='gray',
             linestyle='--', label='y=x')
    p = plt.scatter(x1, x2, *args, **kwargs)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.legend()
    if 'c' in (kwargs):
        p = plt.scatter(x1, x2, *args, **kwargs)
        cbar = plt.colorbar(p) 
    else:
        data, x_e, y_e = np.histogram2d(x1, x2, bins=100, density=True)
        z = interpn(( 0.5*(x_e[1:]+x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])),
                    data, np.vstack([x1, x2]).T, method='splinef2d',
                    bounds_error=False)

        z[np.where(np.isnan(z))] = 0.
        if np.min(z) < 0:
            z += np.abs(np.min(z))
        elif np.min(z) > 0:
            z -= np.min(z)

        idx = z.argsort()
        x1, x2, z = x1[idx], x2[idx], z[idx]
        p = plt.scatter(x1, x2, c=z, *args, **kwargs)
        cbar = plt.colorbar(p)
        cbar.set_label('scatter density')
    plt.title(tit)
    plt.gcf()
    plt.savefig(os.path.join('results', save) + '.png', format='png')


def hist2d_plot(x1, x2, x_lab, y_lab, tit, save, *args, **kwargs):
    x1 = x1.squeeze()
    x2 = x2.squeeze()
    
    plt.figure(dpi=300)
    plt.hist2d(x1, x2, *args, **kwargs, cmap=plt.cm.jet)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(tit)
    plt.gcf()
    plt.savefig(os.path.join('results', save) + '.png', format='png')
    
    
def histogram(data, f_name, title, *args, **kwargs):
    fig, ax = plt.subplots(1, 1, dpi=300)
    sns.histplot(data, ax=ax, *args, **kwargs)
    ax.set_title(title)
    plt.gcf()
    plt.savefig(os.path.join('results', f_name), format='png')