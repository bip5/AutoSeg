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
# # PART 0: ENVIRONMENT SETUP
# # -------------------------------------------------------
# export nnUNet_raw_data_base="/scratch/a.bip5/AutoSeg/nnUNet_raw_data_base/"
# export nnUNet_preprocessed="/scratch/a.bip5/AutoSeg/nnUNet_preprocessed"
# export RESULTS_FOLDER="/scratch/a.bip5/AutoSeg/nnUNet_trained_models"
# # Load modules (Edit as needed for your cluster)
# # module load python/3.XX
# # module load cuda/11.X
# source activate pix2pix 
# # -------------------------------------------------------
# # PART 1: PREPROCESSING (CPU HEAVY)
# # -------------------------------------------------------
# echo "STARTING PREPROCESSING AT $(date)"
# # USE ALL CPUS for preprocessing (It's safe here!)
# # This ensures it finishes in 10 mins instead of 2 hours.
# export nnUNet_def_n_proc=$SLURM_CPUS_ON_NODE
# # (Optional) Verify variable is set, else default to 8
# if [ -z "$nnUNet_def_n_proc" ]; then export nnUNet_def_n_proc=8; fi
# echo "Preprocessing with $nnUNet_def_n_proc threads..."
# # Run Preprocessing
# # -tl : uses the low-res loader threads setting
# # -tf : uses the full-res loader threads setting
# nnUNet_plan_and_preprocess -t 501 --verify_dataset_integrity \
  # -tl $nnUNet_def_n_proc \
  # -tf $nnUNet_def_n_proc
# echo "PREPROCESSING DONE AT $(date)"
# # -------------------------------------------------------
# # PART 2: CLEANUP (RESIDUE REMOVAL)
# # -------------------------------------------------------
# # Explicitly unset the variable so it doesn't confuse training (just in case)
# unset nnUNet_def_n_proc
# # Optional: Sleep briefly to let OS reclaim file handles/memory
sleep 120

# # export NCCL_P2P_DISABLE=1
# # export NCCL_IB_DISABLE=1
# # export CUDA_LAUNCH_BLOCKING=1
# # export TORCH_USE_CUDA_DSA=1
# # export OMP_NUM_THREADS=1
# # export MKL_NUM_THREDS=1
# export TORCH_CUDNN_BENCHMARK=0 

python -c 'import torch;print(torch.backends.cudnn.version())'
python -c 'import torch;print(torch.__version__)'   
PYTHONUNBUFFERED=1 nnUNet_train 3d_fullres nnUNetTrainerV2 501 1 --npz
echo "ALL DONE AT $(date)"