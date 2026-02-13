#!/bin/bash
#SBATCH --account=scw1895
#SBATCH --partition=gpu
#SBATCH --job-name=seg
#SBATCH --output=logs/%A.nnunet.%a
#SBATCH --error=errors/%A.error_log_%a
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --array=0
####SBATCH --time=1-6:00:00



export nnUNet_n_proc_DA=10

python -c 'import torch;print(torch.backends.cudnn.version())'
python -c 'import torch;print(torch.__version__)'   
PYTHONUNBUFFERED=1 nnUNet_train 3d_fullres nnUNetTrainer_Rigidity 501 $SLURM_ARRAY_TASK_ID --npz
echo "ALL DONE AT $(date)"