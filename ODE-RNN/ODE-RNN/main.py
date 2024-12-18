import hyperparams
import argparse
import simulations as sim
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
"""
DCE loss between true and DCE pred 
            or 8
DCE loss between T10 pred 
and DCE pred
"""

# np.random.seed(42)
# torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--nn', type=str, default='linear')
parser.add_argument('--layers', type=int, nargs='+', default=[160, 160, 160])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--attention', action='store_true', default=False)
parser.add_argument('--bidirectional', action='store_true', default=False)
parser.add_argument('--supervised', action='store_true', default=False)
parser.add_argument('--results', action='store_true', default=False)
args = parser.parse_args()
hp = hyperparams.Hyperparams()

# create save name for framework
hp.exp_name = ''
arg_dict = vars(args)
for i, arg in enumerate(arg_dict):
    if i == len(arg_dict)-2:
        hp.exp_name += str(arg_dict[arg])
        break
    else:
        hp.exp_name += '{}_'.format(arg_dict[arg])

print(f'network type: {hp.network.nn}')
print(f'dce epochs: {hp.training.DCE_epochs}')
print(f'T10 epochs: {hp.training.T10_epochs}')
print(f'total its: {hp.training.totalit}')
print(f'training batch size: {hp.training.batch_size}')

sim.run_simulations(hp)