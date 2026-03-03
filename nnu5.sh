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

sleep 100

python -c 'import torch;print(torch.backends.cudnn.version())'
python -c 'import torch;print(torch.__version__)'   
PYTHONUNBUFFERED=1 nnUNet_train 3d_fullres nnUNetTrainerV2 501 4 --npz
echo "ALL DONE AT $(date)"