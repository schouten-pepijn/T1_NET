#!/bin/bash

#SBATCH --job-name=GRUD_no
#SBATCH --mem=18G
#SBATCH --cpus-per-task=4
#SBATCH --time=06-00:00
#SBATCH --partition=luna-long
#SBATCH --nice=1000





# python
echo "starting python"
nice python main.py
echo "finished python"



