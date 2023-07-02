#!/bin/bash
#SBATCH --time=48:30:00
#SBATCH --mem=180gb
#SBATCH --partition=stats-gaann,batch,guest
#SBATCH --job-name=enkf_4
#SBATCH --error=enkf_4.%J.err
#SBATCH --output=enkf_4.%J.out
#SBATCH --nodes=1

pwd
source activate enkf
python MEnKF_50_Reps.py