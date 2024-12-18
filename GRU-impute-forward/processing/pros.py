import numpy as np
import pickle
import torch
import os
import itertools
import nibabel as nib


def mask_extremes(signal, bounds):
    low, high = bounds[0], bounds[1]
    print(f'masking extreme signal voxels < {low} and > {high}')
    mask_low = (signal < high).all(axis=1)
    mask_high = (signal > low).all(axis=1)
    mask = np.logical_and(mask_low, mask_high)
    signal = np.where(mask[:, np.newaxis], signal, 0)
 
    return signal


def dt_shifter(aif_vals, t0_value, t0_bound, aif_type):
    if t0_value is not None:
        lower = t0_value - t0_bound
        upper = t0_value + t0_bound
    else:
        if aif_type == 'pop':
            lower = aif_vals[0].cpu() - t0_bound
            upper = aif_vals[0].cpu() + t0_bound
        elif aif_type == 'pat':
            lower = torch.max(
                aif_vals[:, 0].cpu()) - t0_bound
            upper = torch.max(
                aif_vals[:, 0].cpu()) + t0_bound
    return lower, upper

def mask_zeros(signal, cutoff):
    if cutoff == 0:
        print('masking zero signal voxels')
        mask = (signal != 0).any(axis=1)
        signal = signal[mask, :]
    else:
        print(f'masking mean < {cutoff} signal voxels')
        mask = np.mean(signal, axis=1) > cutoff
        signal = signal[mask, :]
    print(f'data size out: {signal.shape}')
    return signal, mask


def data_loader(paths, tag):
    print('loading data')
    if len(paths) == 2:
        path_clin, path_synth = paths
        if tag == 'dce':
            signal = np.concatenate((np.load(path_clin), np.load(path_synth)), axis=1)
            dce_shape = signal.shape
            param_shape = dce_shape[1:]
            signal = signal.reshape(65, -1).T
            print(f'dce data size: {signal.shape}')
            return signal, dce_shape, param_shape
        elif tag == 'fa':
            signal = np.concatenate((np.load(path_clin), np.load(path_synth)), axis=0)
            fa_mean = np.mean(signal[:, 1, :, :, :], axis=(1, 2, 3))
            signal = np.swapaxes(signal, 0, 1).astype(np.float64)
            fa_shape = signal.shape
            print(f'fa data size: {fa_shape}')
            return signal, fa_mean, fa_shape
    elif len(paths) == 1:
        if tag == 'fa':
            signal = np.load(paths[0]).astype(np.float64)
            if signal.shape[2] != 13:
                signal = signal[:, :, 2:15, :, :]
            fa_mean = np.mean(signal[-2], axis=(1, 2, 3))
            fa_shape = signal.shape
            print(f'fa data size: {fa_shape}')
            return signal, fa_mean, fa_shape
            


def aif_loader_pop(path, t0=None, device=None):
    print('loading population based aif data')
    with open(path, 'rb') as file:
            aif_dicts = pickle.load(file)
    if t0 is not None:
            aif_vals = torch.FloatTensor([t0 if key == 't0'
                                          else aif_dicts[key]
                                          for key 
                                          in ('t0', 'ab', 'mb', 'ae', 'me')])
    else:
        aif_vals = torch.FloatTensor([aif_dicts[key]
                                      for key
                                      in ('t0', 'ab', 'mb', 'ae', 'me')])
    if device:
        aif_vals = aif_vals.to(device)
    return aif_vals


def aif_chain(aif_vals_list, signal_len, instance_len, device=None):
    aif_vals = list(
    itertools.chain.from_iterable(
        itertools.repeat(
            x, int(signal_len/instance_len))
        for x in aif_vals_list))
    if device is not None:
        aif_vals = torch.FloatTensor(aif_vals).to(device)
    return aif_vals


def aif_loader_pat(paths, shape, device, t0=None,
                   mask=None, full_interference=False):
    print('loading manual selected aif')
    signal_len = np.prod(shape[1:])
    aif_clin, aif_synth = paths

    aif_dirs_clin = sorted(
        [os.path.join(aif_clin, d) for d in os.listdir(aif_clin)])
    aif_dirs_synth = sorted(
        [os.path.join(aif_synth, d) for d in os.listdir(aif_synth)])
    aif_dirs = aif_dirs_clin + aif_dirs_synth
    
    aif_dicts = []
    for d in aif_dirs:
        with open(d, 'rb') as file:
            aif_dicts.append(pickle.load(file))

    if isinstance(t0, float):
        print(f'setting aif t0 to {t0}')
        aif_vals_list = [[t0 if key == 't0' else d[key]
                          for key in ('t0', 'ab', 'mb', 'ae', 'me')]
                          for d in aif_dicts]
    else:
        aif_vals_list = [[d[key] for key in ('t0', 'ab', 'mb', 'ae', 'me')]
                          for d in aif_dicts]
    aif_vals_tot = aif_chain(aif_vals_list, signal_len, shape[1])

    if mask is not None:
        mask_idx = np.arange(0, len(aif_vals_tot))
        mask_idx = mask_idx[mask]
        aif_vals = torch.FloatTensor(
            [aif_vals_tot[i] for i in mask_idx]).to(device)
    else:
        aif_vals = torch.FloatTensor(aif_vals_tot).to(device)

    return aif_vals, aif_vals_list


def interleaver(signal, aif_vals, split):
    print(f'adding {split*100}% interleaved aifs to data')
    signal_len = len(signal)
    cutoff_val = int(split * signal_len)
    dce_idx = torch.randperm(signal_len)
    aif_idx = torch.randperm(signal_len)
    dce_idx = dce_idx[:cutoff_val]
    aif_idx = aif_idx[:cutoff_val]
    assert not torch.equal(aif_idx, dce_idx)

    dce_interleaved = np.concatenate((signal, signal[dce_idx, :]), axis=0)
    aif_interleaved = torch.cat((aif_vals, aif_vals[aif_idx, :]), axis=0)
    print(f'data size: {dce_interleaved.shape}')
    return dce_interleaved, aif_interleaved


def time_loader(path, device=None):
    t = torch.FloatTensor(np.load(path))
    if device:
        t = t.to(device)
    print(f'timing (min/max): {t.min()}/{t.max()}')
    return t


def param_map(params, shapes, mask=None,
              save_path=None, save_np=False, save_nii=True):
    X_pred, C_pred, ke, dt, ve, vp, ktrans, T10 = params
    dce_shape, param_shape = shapes
    
    if mask is not None:
        ke_map, dt_map, ve_map, vp_map, ktrans_map = (
            np.zeros((np.prod(param_shape))) for _ in range(5))
        X_pred_map, C_pred_map = (
            np.zeros((len(mask), X_pred.shape[-1])) for I in range(2))

        ke_map[mask], ke_map[~mask] = (ke, 0)
        dt_map[mask], dt_map[~mask] =(dt, 0)
        ve_map[mask], ve_map[~mask] = (ve, 0)
        vp_map[mask], vp_map[~mask] = (vp, 0)
        ktrans_map[mask], ktrans_map[~mask] = (ktrans, 0) 
        X_pred_map[mask, :], X_pred_map[~mask, :] = (X_pred, 0)
        C_pred_map[mask, :], C_pred_map[~mask, :] = (C_pred, 0)

        ke_map = ke_map.reshape(*param_shape)
        dt_map = dt_map.reshape(*param_shape)
        ve_map = ve_map.reshape(*param_shape)
        vp_map = vp_map.reshape(*param_shape)
        ktrans_map = ktrans_map.reshape(*param_shape)
        X_pred_map = np.swapaxes(X_pred_map.T.reshape(*dce_shape), 0, 1)
        C_pred_map = np.swapaxes(C_pred_map.T.reshape(*dce_shape), 0, 1)
    else:
        ke_map = ke.reshape(*param_shape)
        dt_map = dt.reshape(*param_shape)
        ve_map = ve.reshape(*param_shape)
        vp_map = vp.reshape(*param_shape)
        ktrans_map = ktrans.reshape(*param_shape)
        T10_map = T10.reshape(*param_shape)
        X_pred_map = np.swapaxes(X_pred.T.reshape(*dce_shape), 0, 1)
        C_pred_map = np.swapaxes(C_pred.T.reshape(*dce_shape), 0, 1)
    params = [X_pred_map, C_pred_map, ke_map,
              dt_map, ve_map, vp_map, ktrans_map, T10_map]

    if save_path is not None:  
        if save_np:
            np.save(os.path.join(save_path, 'ktrans.npy'), ktrans_map)
        if save_nii:
            for tag, item in zip(
                    ('ke_map', 'dt_map', 've_map',
                     'vp_map', 'ktrans_map', 'T10_map'),
                    params[2:]):
                array_to_nii(item,
                             path=save_path,
                             file_name=f'{tag}.nii.gz',
                             transpose=True)
            for i, X in enumerate(X_pred_map):
                array_to_nii(X,
                              path=save_path,
                              file_name=f'S_pred_pat_{i//2+1}_vis_{i%2+1}.nii.gz',
                              transpose=True)
            for i, C in enumerate(C_pred_map):
                array_to_nii(C,
                              path=save_path,
                              file_name=f'C_pred_pat_{i//2+1}_vis_{i%2+1}.nii.gz',
                              transpose=True)
    return params


def array_to_nii(data, path, file_name, transform=None, transpose=False):
        # x y z t
        print(f'saving {file_name} to nifti')
        if isinstance(transform, tuple):
            print(f'in shape: {data.shape}')
            data = np.moveaxis(data, transform[0], transform[1])
            print(f'out shape: {data.shape}')
        if transpose:
            data = data.T

        if not '.nii.gz' in file_name:
            if '.nii' in file_name:
                file_name.replace('.nii', '.nii.gz')
            else:
                file_name += '.nii.gz'
        ni_img = nib.Nifti1Image(data, np.eye(4))
        nib.save(ni_img, os.path.join(path, file_name))
