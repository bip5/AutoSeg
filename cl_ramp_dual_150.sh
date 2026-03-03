#!/bin/bash
#SBATCH --account=scw1895
#SBATCH --partition=accel_ai
#SBATCH --job-name=seg
#SBATCH --output=logs/%A.nnunet.%a
#SBATCH --error=errors/%A.error_log_%a
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
####SBATCH --time=1-6:00:00

# ── Configuration ────────────────────────────────────────────
TASK="Task501_ISLES2022"
TRAINER="nnUNetTrainer_CLRamp_DualMask_150epochs"
MODEL="3d_fullres"
PLANS="nnUNetPlansv2.1"
FOLD=$SLURM_ARRAY_TASK_ID

# Base paths
RAW_DATA="/scratch/a.bip5/AutoSeg/nnUNet_raw_data_base/nnUNet_raw_data/${TASK}"
INPUT_TS="${RAW_DATA}/imagesTs"
LABELS_TS="${RAW_DATA}/labelsTs"
TEST_BASE="/scratch/a.bip5/AutoSeg/Test/nnUNet/${MODEL}/${TASK}"

# Short name for folders (adjust as needed)
SHORT="cl_dm_150"

export nnUNet_n_proc_DA=8

# ── Environment info ─────────────────────────────────────────
echo "=========================================="
echo "FOLD: ${FOLD}  |  TRAINER: ${TRAINER}"
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

# ── 3. Ensemble (fold 4 only) ───────────────────────────────
if [ "${FOLD}" -eq 4 ]; then
    ENSEMBLE_DIR="${TEST_BASE}/${SHORT}_ensemble"
    echo ""
    echo ">>> ENSEMBLE PREDICTION (all folds) → ${ENSEMBLE_DIR}"
    nnUNet_predict \
        -i ${INPUT_TS} \
        -o ${ENSEMBLE_DIR} \
        -tr ${TRAINER} \
        -m ${MODEL} \
        -p ${PLANS} \
        -t ${TASK}

    echo ">>> EVALUATING ensemble"
    nnUNet_evaluate_folder \
        -ref  ${LABELS_TS} \
        -pred ${ENSEMBLE_DIR} \
        -l 1
fi

# ── 4. Noise Mode Prediction & Evaluation (DualMask) ────────
# Mode 1: [noise, img] → clean-path
# Mode 2: [img, noise] → clean-path
# Mode 3: [noise, img] → aug-path
# Mode 4: [img, noise] → aug-path

NOISE_LABELS=("" "augnoise_clean" "refnoise_clean" "augnoise_aug" "refnoise_aug")

for NMODE in 1 2 3 4; do
    NOISE_DIR="${TEST_BASE}/${SHORT}_f${FOLD}_${NOISE_LABELS[$NMODE]}"
    echo ""
    echo ">>> NOISE MODE ${NMODE} → ${NOISE_DIR}"
    python -m nnunet.inference.predict_noise \
        -i ${INPUT_TS} \
        -o ${NOISE_DIR} \
        -tr ${TRAINER} \
        -m ${MODEL} \
        -p ${PLANS} \
        -t ${TASK} \
        --noise_mode ${NMODE} \
        -f ${FOLD}

    echo ">>> EVALUATING noise mode ${NMODE}"
    nnUNet_evaluate_folder \
        -ref  ${LABELS_TS} \
        -pred ${NOISE_DIR} \
        -l 1
done

echo ""
echo "ALL DONE AT $(date)"
