#!/bin/bash

#SBATCH --job-name=3_T10_DCE-NET
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=18G
#SBATCH --cpus-per-task=1
#SBATCH --time=40:00:00
#SBATCH --reservation=gpu
#SBATCH --nice=1000

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# show CPU device
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# modules
module load devel/cuda/9.1

# conda
echo "activating conda"
cd ~/scratch/T10-NET/transformer/big_net/T10_6xFA_7_100_SNR_sampled
source /scratch/pschouten/virtualenv/env1/bin/activate
conda activate /scratch/pschouten/virtualenv/env1
echo "conda activated"

# python
echo "starting python"
nice python main.py
echo "finished python"

# mail
echo "run time mailed"


