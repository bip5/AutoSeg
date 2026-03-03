#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=gpu
#SBATCH --job-name=imp_seg
#SBATCH --output=logs/%j.nnunet
#SBATCH --error=errors/%j.n
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
###SBATCH --mem=32G
####SBATCH --time=1-6:00:00

# -------------------------------------------------------
# PART 0: ENVIRONMENT SETUP
# -------------------------------------------------------
export nnUNet_raw_data_base="/scratch/a.bip5/AutoSeg/nnUNet_raw_data_base/"
export nnUNet_preprocessed="/scratch/a.bip5/AutoSeg/nnUNet_preprocessed"
export RESULTS_FOLDER="/scratch/a.bip5/AutoSeg/nnUNet_trained_models"
# Load modules (Edit as needed for your cluster)
# module load python/3.XX
# module load cuda/11.X
source activate pix2pix 

python -c 'import torch;print(torch.backends.cudnn.version())'
python -c 'import torch;print(torch.__version__)'   
PYTHONUNBUFFERED=1 nnUNet_train 2d nnUNetTrainerV2 501 4 --npz
echo "ALL DONE AT $(date)"