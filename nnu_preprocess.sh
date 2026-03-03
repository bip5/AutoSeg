#!/bin/bash
#SBATCH --account=scw1895
#SBATCH --partition=compute
#SBATCH --job-name=prep
#SBATCH --output=logs/%j.nnunet_prep
#SBATCH --error=errors/%j.prep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
####SBATCH --time=1-6:00:00

export nnUNet_raw_data_base="/scratch/a.bip5/AutoSeg/nnUNet_raw_data_base/"
export nnUNet_preprocessed="/scratch/a.bip5/AutoSeg/nnUNet_preprocessed"
export RESULTS_FOLDER="/scratch/a.bip5/AutoSeg/nnUNet_trained_models"
export nnUNet_n_proc_DA=8
# 2. Run
# Use -c to continue from a checkpoint if interrupted
nnUNet_plan_and_preprocess -t 501