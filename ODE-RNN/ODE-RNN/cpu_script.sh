#!/bin/bash

#SBATCH --job-name=NCDE_2M0
#SBATCH --mem=18G
#SBATCH --cpus-per-task=6
#SBATCH --time=6-00:00
#SBATCH --partition=luna-long
#SBATCH --nice=1000





# python
echo "starting python"
nice python main.py
echo "finished python"



