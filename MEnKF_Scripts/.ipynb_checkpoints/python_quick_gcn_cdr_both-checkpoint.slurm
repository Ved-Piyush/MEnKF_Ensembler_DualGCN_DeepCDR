#!/bin/bash
#SBATCH --time=24:30:00
#SBATCH --mem=700gb
#SBATCH --partition=guest_gpu
#SBATCH --job-name=enkf_4
#SBATCH --error=enkf_4.%J.err
#SBATCH --output=enkf_4.%J.out
#SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --constraint=gpu_32gb
pwd
source activate tensorflow-gpu-2.9-custom
python MEnKF_DualGCN_DeepCDR_Sequential_Injection_Plots.py