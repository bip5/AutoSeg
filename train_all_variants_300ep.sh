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

# Define explicit nnU-Net environment path for checkpoint checking
export nnUNet_results="/scratch/a.bip5/AutoSeg/nnUNet_trained_models/nnUNet"


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
TRAINER="${TRAINERS[$SLURM_ARRAY_TASK_ID]}"

if [ -z "${TRAINER}" ]; then
    echo "ERROR: Invalid array index ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

# ── Auto-commit code state before run ────────────────────────
REPO_DIR="/scratch/a.bip5/AutoSeg/nnUNet"
cd ${REPO_DIR}
git add -A
git commit -m "pre-run ${TRAINER} fold${FOLD} $(date '+%Y-%m-%d %H:%M:%S')" || echo "Nothing to commit"
git push origin main || echo "Push failed (non-critical)"
cd -

# ── Environment info ─────────────────────────────────────────
echo "=========================================="
echo "FOLD: ${FOLD}  |  TRAINER: ${TRAINER}"
echo "ARRAY_ID: ${SLURM_ARRAY_TASK_ID}"
echo "=========================================="
python -c 'import torch;print("cuDNN:", torch.backends.cudnn.version())'
python -c 'import torch;print("PyTorch:", torch.__version__)'

# ── 1. Training ──────────────────────────────────────────────
echo ""
MODEL_DIR="${nnUNet_results}/${MODEL}/${TASK}/${TRAINER}__${PLANS}/fold_${FOLD}"
if [ -f "${MODEL_DIR}/model_final_checkpoint.model" ] && [ -z "${FORCE_RETRAIN}" ]; then
    echo ">>> SKIPPING training — model exists: ${MODEL_DIR}"
else
    echo ">>> TRAINING fold ${FOLD} ..."
    PYTHONUNBUFFERED=1 nnUNet_train ${MODEL} ${TRAINER} 501 ${FOLD} --npz
fi

# ── 2. Standard Prediction & Evaluation ──────────────────────
PRED_DIR="${TEST_BASE}/${TRAINER}_f${FOLD}"
echo ""
if [ -f "${PRED_DIR}/summary.json" ] && [ -z "${FORCE_RETRAIN}" ]; then
    echo ">>> SKIPPING prediction & evaluation — results exist: ${PRED_DIR}"
else
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
fi

# ── 3. Collect results ───────────────────────────────────────
echo ""
echo ">>> COLLECTING results for ${TRAINER}"
python collect_results.py ${TEST_BASE} --prefix ${TRAINER} \
    -o ${TEST_BASE}/${TRAINER}_results.txt

# ── 4. Visualise predictions ─────────────────────────────────
VIS_DIR="${PRED_DIR}_vis"
echo ""
echo ">>> VISUALISING predictions for ${TRAINER}"
echo ">>> Outputting visualisations to: ${VIS_DIR}"
python visualise_results.py single \
    --pred-dir ${PRED_DIR} \
    --gt-dir ${LABELS_TS} \
    --img-dir ${INPUT_TS} \
    --out-dir ${VIS_DIR} \
    ${FORCE_RETRAIN:+--overwrite}

echo ""
echo "ALL DONE AT $(date)"
