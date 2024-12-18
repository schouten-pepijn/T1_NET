# import libraries
import torch
import torch.nn as nn
import torch.linalg as la
import torch.utils.data as utils
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import model
import matplotlib
from tqdm import tqdm

from sklearn.metrics import mean_squared_error

# density plots
from scipy.interpolate import interpn

# import transformer as trans
# import transformer_new as trans
# import transformer_upscale as trans
# import linear as trans
# import tcn as trans
import mTAN as trans
matplotlib.use('TkAgg')

# np.random.seed(142)
# torch.manual_seed(142)

def weighted_mse_loss(input, target, weight):
    return (weight * (input - target)**2).mean()

def square_root_scheduler(lr, exp, update):
    if exp > 0:
        exp = -exp  
    factor = lr * pow(update + 1, exp)
    if factor < 1.5:
        factor = 1.5
    return factor

def NMSE_loss(input, target, weight=None, reduction='mean'):
    if weight is not None:
        all_losses = weight * ((input-target)**2) / (target**2)
        if reduction == 'mean':
            loss = torch.sum(all_losses) / torch.sum(weight)
        elif reduction == 'eval':
            loss = torch.mean(all_losses, axis=1)
        else:
            raise ValueError('not a valid reduction')
    else:
        all_losses = ((input-target)**2) / (target**2)
        if reduction == 'mean':
            loss = torch.mean(all_losses)
        elif reduction == 'eval':
            loss = torch.mean(all_losses, axis=1)
        else:
            raise ValueError('not a valid reduction')
    return loss


def normalize_params(pred_params, orig_params, bounds):
    pred_params = pred_params.T
    for i in range(len(bounds)):
        pred_params[:, i] /= (bounds[1, i] - bounds[0, i])
        orig_params[:, i] /= (bounds[1, i] - bounds[0, i])

    return pred_params, orig_params


def interpolate(inp, fi):
    i, f = int(fi//1), fi%1
    j = i+1 if f > 0 else i

    return (1-f) * inp[i] + f*inp[j]


def train_T1_transformer(hp, params_T1_true, net=None,
                         orig_params=None):

    if hp.use_cuda:
        torch.backends.cudnn.benchmark = True

    if net is None:
        net = trans.T10_transformer(copy.deepcopy(hp)).to(hp.device)

    # Loss function and optimizer
    criterion = nn.MSELoss().to(hp.device)

    # Data loader
    # if not hp.second_val_set:
    print('using split of training set as validation')
    split = int(np.floor(len(params_T1_true.X_fa)*hp.training.split))

    indices = np.arange(len(params_T1_true.X_fa), dtype=int)
    train_idx, val_idx = utils.random_split(
        indices, [split, len(indices) - split])

    trainloader = utils.DataLoader(train_idx,
                                    batch_size=hp.training.batch_size,
                                    shuffle=True,
                                    num_workers=1,
                                    drop_last=True)
    valloader = utils.DataLoader(val_idx,
                                  batch_size=hp.training.val_batch_size,
                                  shuffle=False,
                                  num_workers=1,
                                  drop_last=True)

    num_batches = len(train_idx) // hp.training.batch_size
    num_batches2 = len(val_idx) // hp.training.val_batch_size    

    if num_batches > hp.training.totalit:
        totalit = hp.training.totalit
    else:
        totalit = num_batches

    if not os.path.exists(hp.out_fold):
        os.makedirs(hp.out_fold)

    optimizer, scheduler_wu, scheduler = model.load_optimizer(net, hp)

    params_total = sum(p.numel() for p in net.parameters())
    train_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    # fix for sudden nan values in patient data
    # for name, p in net.named_parameters():
    #     if p.requires_grad:
    #         p.register_hook(lambda grad: torch.nan_to_num(grad))

    print(f'{params_total} params in total')
    print(f'{train_params} trainable params in total')

    best = 1e16
    num_bad_epochs = 0
    loss_train = []
    loss_val = []

    for epoch in range(hp.training.T10_epochs):
        print("-----------------------------------------------------------------")
        print("\nEpoch: {}: Current best val_loss: {}".format(epoch, best))

        train_loss = 0.
        val_loss = 0.
        train_loss_curve = 0.
        val_loss_curve = 0.

        # training
        net = net.train()
        for i, X_idx in enumerate(trainloader):
            if i == totalit:
                break
            
            # for i in range(100000):
    
            # T10
            X_fa_batch = torch.FloatTensor(params_T1_true.X_fa[X_idx, :])
            fa_batch = torch.FloatTensor(params_T1_true.fa[X_idx, :])
            fa_mask_batch = torch.BoolTensor(params_T1_true.fa_mask[X_idx, :])
            fa_len_batch = torch.FloatTensor(params_T1_true.fa_len[X_idx, np.newaxis])
            TR_batch = torch.FloatTensor(params_T1_true.TR[X_idx, np.newaxis])
            
            if hp.norm == 'normalization':
                # min-max normalization
                X_fa_batch = (X_fa_batch - params_T1_true.X_fa.min()) / \
                    (params_T1_true.X_fa.max() - params_T1_true.X_fa.min())
            
            X_fa_batch = X_fa_batch.to(hp.device)
            fa_batch = fa_batch.to(hp.device)
            fa_mask_batch = fa_mask_batch.to(hp.device)
            fa_len_batch = fa_len_batch.to(hp.device)
            TR_batch = TR_batch.to(hp.device)
    
            # T10 test
            T10_batch = torch.FloatTensor(params_T1_true.T10[X_idx])
            M0_batch = torch.FloatTensor(params_T1_true.M0[X_idx])
            T10_batch = T10_batch.to(hp.device)
            M0_batch = M0_batch.to(hp.device)
    
            optimizer.zero_grad()   

            X_pred, T10_pred, M0_pred = net(X_fa_in=X_fa_batch,
                                          fa_vals=fa_batch,
                                          fa_mask=fa_mask_batch,
                                          fa_len=fa_len_batch,
                                          TR_vals=TR_batch)
            
            
            batch_loss = X_fa_batch.squeeze()   
            pred_loss = X_pred.squeeze()

            loss = criterion(pred_loss[fa_mask_batch],
                             batch_loss[fa_mask_batch])

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
    
            if not i % 100:
                X_print = la.norm(torch.abs(batch_loss - pred_loss)) / la.norm(batch_loss)
                T10_print = la.norm(torch.abs(T10_batch - T10_pred.squeeze())) / la.norm(T10_batch)
                M0_print = la.norm(torch.abs(M0_batch - M0_pred.squeeze())) / la.norm(M0_batch)
                # X_print = -100
                print(f'epoch: {epoch:2} '\
                      f'batch: {i:4}\{totalit} '\
                      f'X_diff: {X_print:.4f} '\
                      f'M0: {torch.mean(M0_pred.detach().cpu()).numpy():.4f} '\
                      f'M0_diff: {M0_print:.4f} '\
                      f'T10: {torch.mean(T10_pred.detach().cpu()).numpy():.4f} '\
                      f'T10_diff: {T10_print:.4f} '\
                      f'loss: {loss.item():.8f}')

        # # evaluation
        params = params_T1_true
        net = net.eval()
        with torch.no_grad():
            for i, X_idx in enumerate(tqdm(valloader, position=0, leave=True), 0):
  
                # T10
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

                # T10 test
                T10_batch_eval = torch.FloatTensor(params.T10[X_idx])
                T10_batch_eval = T10_batch_eval.to(hp.device)

                X_eval, T10_eval, M0_eval = net(X_fa_in=X_fa_batch,
                                                fa_vals=fa_batch,
                                                fa_mask=fa_mask_batch,
                                                fa_len=fa_len_batch,
                                                TR_vals=TR_batch)

                eval_loss = X_eval.squeeze()
                batch_loss = X_fa_batch.squeeze()
                loss = criterion(eval_loss[fa_mask_batch],
                                 batch_loss[fa_mask_batch])

                val_loss += loss.item()
                
                
            print(f'\n--- validation ---\n'
                  f'epoch: {epoch:2} '\
                  f'T10_sys: {torch.mean(T10_batch-T10_eval):.4f} '\
                  f'T10_rand: {torch.std(T10_batch-T10_eval):.4f}')

        # net = net.train()
        scheduler_wu.step()  # warm up rate

        # scale losses
        train_loss = train_loss/totalit*1000
        val_loss = val_loss/num_batches2*1000
        loss_train.append(train_loss)
        loss_val.append(val_loss)
        
        # x-y plot
        loss_plot(loss_train, loss_val,
                  os.path.join('results', 'training', 'T10',
                               'loss', f'loss_curve_{epoch}.png'))
        
        x_x_plot(T10_batch.cpu().numpy(),
                 T10_pred.detach().cpu().numpy(),
                 'True', 'Pred',
                 'true vs net',
                 os.path.join('results','training', 'T10',
                              'x_y', 'train',
                              f'T10_true_vs_pred_train_{epoch}') + '.png',
                 s=5, marker='.')
        
        x_x_plot(T10_batch_eval.cpu().numpy(),
                 T10_eval.cpu().numpy(),
                 'True', 'Pred',
                 'true vs net',
                 os.path.join('results','training', 'T10',
                              'x_y', 'eval',
                              f'T10_true_vs_pred_eval_{epoch}') + '.png',
                 s=5, marker='.')

        torch.save(net.state_dict(), f'results/training/T10/state_dict/net_state_dict_{epoch}.pt')

        if hp.training.optim_patience_down > 0:
            scheduler.step(val_loss)

        if val_loss < best:
            print("\n############### Saving good model ###############################")
            final_model = copy.deepcopy(net.state_dict())
            best = val_loss
            num_bad_epochs = 0
            if hp.supervised:
                print("\nLoss: {}; val_loss: {}; loss_curve: {}; val_loss_curve: {}; bad epochs: {}".format(train_loss,
                                                                                                            val_loss,
                                                                                                            train_loss_curve,
                                                                                                            val_loss_curve,
                                                                                                            num_bad_epochs))

            else:
                print("\nLoss: {}; val_loss: {}; bad epochs: {}".format(train_loss,
                                                                        val_loss,
                                                                        num_bad_epochs))

        else:
            num_bad_epochs += 1
            if hp.supervised:
                print("\nLoss: {}; val_loss: {}; loss_curve: {}; val_loss_curve: {}; bad epochs: {}".format(train_loss,
                                                                                                            val_loss,
                                                                                                            train_loss_curve,
                                                                                                            val_loss_curve,
                                                                                                            num_bad_epochs))

            else:
                print("\nLoss: {}; val_loss: {}; bad epochs: {}".format(train_loss,
                                                                        val_loss,
                                                                        num_bad_epochs))

        # early stopping
        if num_bad_epochs == hp.training.patience:
            print("\nEarly stopping, best val loss: {}".format(best))
            print("Done with fitting")
            break

    print("Done")
    net.load_state_dict(final_model)
    return net


def idx_shuffler(x_idx, split):
    split = int(split * len(x_idx))
    shuffled = x_idx[:split]
    idx = torch.randperm(len(shuffled))
    x_idx = torch.cat((shuffled[idx], x_idx[split:]))
    return x_idx


def do_plots(hp, epoch, X_batch, X_pred, loss_train,
             loss_val, values, loss_train_curve=None,
             loss_val_curve=None, name=None, net_name=None):
    # plot loss history
    plt.close('all')

    labels = ['worst', 'median', 'best']
    fig, axs = plt.subplots(int(len(values)/2)+1, 2, figsize=(6, 5))

    for i in range(len(values)):
        axs[int(i/2), i%2].plot(X_batch.data[i])
        axs[int(i/2), i%2].plot(X_pred.data[i])
        axs[int(i/2), i%2].set_title('{} {}, loss:{:.2e}'.format(
            labels[int(i/2)], (i%2)+1, values[i].item()))

    for ax in axs.flat:
        ax.set(xlabel='time (m)', ylabel='signal (a.u.)')

    for ax in axs.flat:
        ax.label_outer()

    axs[3, 0].plot(loss_train)
    axs[3, 0].plot(loss_val)
    axs[3, 0].set_yscale('log')
    axs[3, 0].set_xlabel('epoch')
    axs[3, 0].set_ylabel('loss')

    plt.ion()
    plt.tight_layout()
    # plt.show()
    # plt.pause(0.001)

    if hp.training.save_train_fig:
        if not os.path.isdir(f'{hp.out_fold}/training'):
            os.makedirs(f'{hp.out_fold}/training')
        # plt.gcf()
        plt.savefig(f'{hp.out_fold}/training/{net_name}/{name}_fit_{epoch}.png')

    return fig


def push_zeros(a, device):
    valid_mask = a!=0

    flipped_mask = torch.fliplr(
        torch.sum(valid_mask, dim=1, keepdim=True)
        > torch.arange(a.shape[1]-1, -1, -1).to(device))
    a[flipped_mask] = a[valid_mask]
    a[~flipped_mask] = 0

    return a

def loss_plot(train_loss, val_loss, save, *args, **kwargs):
    
    fig = plt.figure(dpi=300)
    plt.plot(train_loss, label='Train', *args, **kwargs)
    plt.plot(val_loss, label='Val', *args, **kwargs)
    plt.title('loss curve')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.gcf()
    plt.savefig(save, format='png')
    

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
    plt.savefig(save, format='png')