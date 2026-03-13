#!/bin/bash
#SBATCH --account=scw1895
#SBATCH --partition=gpu
#SBATCH --job-name=v300
#SBATCH --output=logs/%A.nnunet.%a
#SBATCH --error=errors/%A.error_log_%a
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-6

# ── 300-epoch Fold 0 for all variants ────────────────────────
# Excludes: CLRamp (non-dual) and NoCLRamp (non-dual)
# Submit: sbatch train_all_variants_300ep.sh

TASK="Task501_ISLES2022"
MODEL="3d_fullres"
PLANS="nnUNetPlansv2.1"
FOLD=0

RAW_DATA="/scratch/a.bip5/AutoSeg/nnUNet_raw_data_base/nnUNet_raw_data/${TASK}"
INPUT_TS="${RAW_DATA}/imagesTs"
LABELS_TS="${RAW_DATA}/labelsTs"
TEST_BASE="/scratch/a.bip5/AutoSeg/Test/nnUNet/${MODEL}/${TASK}"

export nnUNet_n_proc_DA=8

# ── Trainer list (indexed by SLURM_ARRAY_TASK_ID) ────────────
# "nnUNetTrainerV2_300epochs"
# "nnUNetTrainer_NoCLRamp_DualMask_300epochs"
# "nnUNetTrainer_Rigidity_300epochs"
# "nnUNetTrainer_SENet_300epochs"
   # "nnUNetTrainer_SegResNet_300epochs"
    # "nnUNetTrainer_SegResNet_IN_300epochs"
	# "nnUNetTrainerV2_InputResidualUNet_PerConv_Unique_300epochs"
	# "nnUNetTrainer_CLRamp_DualMask_300epochs"    
TRAINERS=(           
	"nnUNetTrainer_MaskDenoise_300epochs"
    "nnUNetTrainer_MaskDenoiseRandom_300epochs"    
    "nnUNetTrainerV2_InputResidualUNet_PerStage_Split_300epochs"
    "nnUNetTrainerV2_InputResidualUNet_PerConv_Split_300epochs"
    "nnUNetTrainerV2_InputResidualUNet_PerStage_Unique_300epochs"        
)

SHORT_NAMES=(
    "baseline"
    "ncl_dm"
    "cl_dm"
    "rigid"
    "senet"
    "ir_perstage"
    "ir_perconv"
    "ir_perstage_uniq"
    "ir_perconv_uniq"
    "segresnet"
    "segresnet_in"
    "maskdenoise"
    "maskdenoise_rand"
)

TRAINER="${TRAINERS[$SLURM_ARRAY_TASK_ID]}"
SHORT="${SHORT_NAMES[$SLURM_ARRAY_TASK_ID]}_300ep"

if [ -z "${TRAINER}" ]; then
    echo "ERROR: Invalid array index ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

# ── Auto-commit code state before run ────────────────────────
REPO_DIR="/scratch/a.bip5/AutoSeg/nnUNet"
cd ${REPO_DIR}
git add -A
git commit -m "pre-run ${SHORT} fold${FOLD} $(date '+%Y-%m-%d %H:%M:%S')" || echo "Nothing to commit"
git push origin main || echo "Push failed (non-critical)"
cd -

# ── Environment info ─────────────────────────────────────────
echo "=========================================="
echo "FOLD: ${FOLD}  |  TRAINER: ${TRAINER}"
echo "SHORT: ${SHORT}  |  ARRAY_ID: ${SLURM_ARRAY_TASK_ID}"
echo "=========================================="
python -c 'import torch;print("cuDNN:", torch.backends.cudnn.version())'
python -c 'import torch;print("PyTorch:", torch.__version__)'

# ── 1. Training ──────────────────────────────────────────────
echo ""
echo ">>> TRAINING fold ${FOLD} ..."
PYTHONUNBUFFERED=1 nnUNet_train ${MODEL} ${TRAINER} 501 ${FOLD} --npz

# ── 2. Standard Prediction & Evaluation ──────────────────────
PRED_DIR="${TEST_BASE}/${SHORT}_f${FOLD}"
echo ""
echo ">>> PREDICTING fold ${FOLD} → ${PRED_DIR}"
nnUNet_predict \
    -i ${INPUT_TS} \
    -o ${PRED_DIR} \
    -tr ${TRAINER} \
    -m ${MODEL} \
    -p ${PLANS} \
    -t ${TASK} \
    -f ${FOLD}

echo ">>> EVALUATING fold ${FOLD}"
nnUNet_evaluate_folder \
    -ref  ${LABELS_TS} \
    -pred ${PRED_DIR} \
    -l 1

# ── 3. Collect results ───────────────────────────────────────
echo ""
echo ">>> COLLECTING results for ${SHORT}"
python collect_results.py ${TEST_BASE} --prefix ${SHORT} \
    -o ${TEST_BASE}/${SHORT}_results.txt

echo ""
echo "ALL DONE AT $(date)"
