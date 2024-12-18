#!/bin/sh
#SBATCH --job-name=ode-rnn
#SBATCH --mem=18G               
#SBATCH --cpus-per-task=1      
#SBATCH --time=3-20:00         
#SBATCH --partition=rng-long
#SBATCH --nice=1000      

# python
echo "starting python"
nice python main.py
echo "finished python"



