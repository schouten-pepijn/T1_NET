import hyperparams
import argparse
import simulations as sim
import numpy as np
import torch
"""
DCE loss between true and DCE pred 
            or 8
DCE loss between T10 pred 
and DCE pred
"""

hp = hyperparams.Hyperparams()

print(f'network type: {hp.network.nn}')
print(f'dce epochs: {hp.training.DCE_epochs}')
print(f'T10 epochs: {hp.training.T10_epochs}')
print(f'total its: {hp.training.totalit}')
print(f'training batch size: {hp.training.batch_size}')

sim.run_simulations(hp)