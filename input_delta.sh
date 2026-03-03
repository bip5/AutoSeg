#!/bin/bash
#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=seg
#SBATCH --output=logs/%A.nnunet.%a
#SBATCH --error=errors/%A.error_log_%a
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=1-4
####SBATCH --time=1-6:00:00



export nnUNet_n_proc_DA=8

python -c 'import torch;print(torch.backends.cudnn.version())'
python -c 'import torch;print(torch.__version__)'   
PYTHONUNBUFFERED=1 nnUNet_train 3d_fullres nnUNetTrainerV2_InputResidualUNet_PerConv 501 $SLURM_ARRAY_TASK_ID --npz
echo "ALL DONE AT $(date)"
