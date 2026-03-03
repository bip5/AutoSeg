#!/bin/bash

#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=seg
#SBATCH --output=logs/%A.nnunet_%a 
#SBATCH --error=errors/%j.n
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=1-4
###SBATCH --mem=32G
####SBATCH --time=1-6:00:00


# -------------------------------------------------------
# PART 1: PREPROCESSING (CPU HEAVY)
# -------------------------------------------------------
echo "STARTING PREPROCESSING AT $(date)"
# USE ALL CPUS for preprocessing (It's safe here!)
# This ensures it finishes in 10 mins instead of 2 hours.
export nnUNet_def_n_proc=$SLURM_CPUS_ON_NODE
# (Optional) Verify variable is set, else default to 8
if [ -z "$nnUNet_def_n_proc" ]; then export nnUNet_def_n_proc=8; fi
echo "Preprocessing with $nnUNet_def_n_proc threads..."
# Run Preprocessing
# -tl : uses the low-res loader threads setting
# -tf : uses the full-res loader threads setting
nnUNet_plan_and_preprocess -t 501 --verify_dataset_integrity \
  -tl $nnUNet_def_n_proc \
  -tf $nnUNet_def_n_proc
echo "PREPROCESSING DONE AT $(date)"
# -------------------------------------------------------
# PART 2: CLEANUP (RESIDUE REMOVAL)
# -------------------------------------------------------
# Explicitly unset the variable so it doesn't confuse training (just in case)
unset nnUNet_def_n_proc
# Optional: Sleep briefly to let OS reclaim file handles/memory
sleep 10

python -c 'import torch;print(torch.backends.cudnn.version())'
python -c 'import torch;print(torch.__version__)'   
PYTHONUNBUFFERED=1 nnUNet_train 3d_fullres nnUNetTrainer_NoCLRamp 501 $SLURM_ARRAY_TASK_ID --npz
echo "ALL DONE AT $(date)"