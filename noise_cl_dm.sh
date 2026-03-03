#!/bin/bash
# CL DualMask — Noise analysis for all 5 folds × 4 modes + ensemble
# Run interactively: bash noise_cl_dm.sh

TASK="Task501_ISLES2022"
TRAINER="nnUNetTrainer_CLRamp_DualMask"
MODEL="3d_fullres"
PLANS="nnUNetPlansv2.1"

RAW_DATA="/scratch/a.bip5/AutoSeg/nnUNet_raw_data_base/nnUNet_raw_data/${TASK}"
INPUT_TS="${RAW_DATA}/imagesTs"
LABELS_TS="${RAW_DATA}/labelsTs"
TEST_BASE="/scratch/a.bip5/AutoSeg/Test/nnUNet/${MODEL}/${TASK}"

# ── Auto-commit code state before run ────────────────────────
REPO_DIR="/scratch/a.bip5/AutoSeg/nnUNet"
cd ${REPO_DIR}
git add -A
git commit -m "pre-run CL_DM noise eval $(date '+%Y-%m-%d %H:%M:%S')" || echo "Nothing to commit"
git push origin main || echo "Push failed (non-critical)"
cd -

# ── Per-fold noise runs ──────────────────────────────────────
for FOLD in 0 1 2 3 4; do
    for NMODE in 1 2 3 4 5; do
        OUT_DIR="${TEST_BASE}/cl_dm_f${FOLD}_noise${NMODE}"
        echo ""
        echo ">>> Fold ${FOLD} | Noise mode ${NMODE} → ${OUT_DIR}"

        python -m nnunet.inference.predict_noise \
            -i ${INPUT_TS} \
            -o ${OUT_DIR} \
            -tr ${TRAINER} \
            -m ${MODEL} \
            -p ${PLANS} \
            -t ${TASK} \
            --noise_mode ${NMODE} \
            -f ${FOLD}

        nnUNet_evaluate_folder \
            -ref  ${LABELS_TS} \
            -pred ${OUT_DIR} \
            -l 1
    done
done

# ── Ensemble noise runs ─────────────────────────────────────
for NMODE in 1 2 3 4 5; do
    OUT_DIR="${TEST_BASE}/cl_dm_ens_noise${NMODE}"
    echo ""
    echo ">>> Ensemble | Noise mode ${NMODE} → ${OUT_DIR}"

    python -m nnunet.inference.predict_noise \
        -i ${INPUT_TS} \
        -o ${OUT_DIR} \
        -tr ${TRAINER} \
        -m ${MODEL} \
        -p ${PLANS} \
        -t ${TASK} \
        --noise_mode ${NMODE} \
        -f 0 1 2 3 4

    nnUNet_evaluate_folder \
        -ref  ${LABELS_TS} \
        -pred ${OUT_DIR} \
        -l 1
done

echo ""
echo ">>> Collecting results"
python collect_results.py ${TEST_BASE} --prefix cl_dm -o ${TEST_BASE}/cl_dm_noise_results.txt

echo "ALL DONE AT $(date)"
