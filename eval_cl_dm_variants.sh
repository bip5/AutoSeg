#!/bin/bash
# CL DualMask — Predict, evaluate & noise analysis for fold-0 variants
# Run interactively: bash eval_cl_dm_variants.sh

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
git commit -m "pre-run CL_DM variant eval $(date '+%Y-%m-%d %H:%M:%S')" || echo "Nothing to commit"
git push origin main || echo "Push failed (non-critical)"
cd -

# ── Fold variants: SUFFIX|SHORT_NAME|CHECKPOINT ──────────────
VARIANTS=(
    "0_standard_identical_sgd_i0.1_p5|var_identical_sgd|model_final_checkpoint"
    "0_standard_sgd_i0.1_p5|var_standard_sgd|model_final_checkpoint"
    "0_standard_adamw_i0.1_p5|var_standard_adamw|model_final_checkpoint"
    "0_standard_sgd_i0.1_p5|var_spectrum_sgd|model_best_spectrum"
)

# ── Standard prediction + noise analysis per variant ─────────
for ENTRY in "${VARIANTS[@]}"; do
    IFS='|' read -r FSUFFIX SHORT CHK <<< "${ENTRY}"

    for NMODE in 0 1 2 3 4 5; do
        OUT_DIR="${TEST_BASE}/cl_dm_${SHORT}_noise${NMODE}"
        echo ""
        echo "============================================================"
        echo ">>> ${SHORT} | Noise mode ${NMODE} | chk=${CHK} → ${OUT_DIR}"
        echo "============================================================"

        python -m nnunet.inference.predict_noise \
            -i ${INPUT_TS} \
            -o ${OUT_DIR} \
            -tr ${TRAINER} \
            -m ${MODEL} \
            -p ${PLANS} \
            -t ${TASK} \
            --noise_mode ${NMODE} \
            -f ${FSUFFIX} \
            -chk ${CHK}

        nnUNet_evaluate_folder \
            -ref  ${LABELS_TS} \
            -pred ${OUT_DIR} \
            -l 1
    done
done

# ── Collect variant results only ─────────────────────────────
echo ""
echo ">>> Collecting variant results"
python collect_results.py ${TEST_BASE} --prefix cl_dm_var -o ${TEST_BASE}/cl_dm_variant_results.txt

echo "ALL DONE AT $(date)"
