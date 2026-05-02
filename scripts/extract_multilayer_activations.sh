#!/bin/bash
set -e  # Exit on error

################################################################################
# Extract Multi-Layer Activations from Existing Samples
#
# This script extracts activations from multiple DeepVariant layers for samples
# that already have examples generated.
#
# Layers to extract:
#   - mixed0 (early features, ~28x28, 256 channels)
#   - mixed3 (mid-level, ~14x14, 768 channels)
#   - mixed9 (deepest before classification, ~4x12, 2048 channels)
#   - mixed10 (final layer, ~4x12, 2048 channels)
################################################################################

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(dirname "${SCRIPT_DIR}")"  # Project root
EMBEDDINGS_BASE="${WORK_DIR}/data/1kg_embeddings"

# Samples (the 15 samples from your v3 dataset)
SAMPLES=(
    "HG00731"   # EUR
    "HG01048"   # AMR
    "HG01197"   # AMR
    "HG02922"   # AFR
    "NA18939"   # EAS
    "NA20502"   # EUR
    "NA20847"   # SAS
    "NA19238"   # AFR
    "HG01985"   # AFR
    "HG01565"   # AMR
    "HG00513"   # EAS
    "NA18525"   # EAS
    "NA12878"   # EUR
    "NA20845"   # SAS
    "NA20846"   # SAS
)

# Layers to extract
LAYERS=("mixed0" "mixed3" "mixed9" "mixed10")

echo "=========================================="
echo "Multi-Layer Activation Extraction"
echo "=========================================="
echo "Working directory: ${WORK_DIR}"
echo "Samples: ${#SAMPLES[@]}"
echo "Layers: ${LAYERS[@]}"
echo ""

################################################################################
# Extract activations for each layer
################################################################################
for LAYER in "${LAYERS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Extracting ${LAYER} activations"
    echo "=========================================="

    SAMPLE_COUNT=0
    for SAMPLE in "${SAMPLES[@]}"; do
        ((SAMPLE_COUNT++))
        echo ""
        echo "[${SAMPLE_COUNT}/${#SAMPLES[@]}] Processing ${SAMPLE} (${LAYER})"

        SAMPLE_DIR="${EMBEDDINGS_BASE}/${SAMPLE}"
        INTERMEDIATE_DIR="${SAMPLE_DIR}/intermediate_results_dir"
        EXAMPLES="${INTERMEDIATE_DIR}/make_examples.tfrecord@1.gz"
        OUTFILE="${SAMPLE_DIR}/call_variants_output_${LAYER}.tfrecord.gz"
        ACTIVATION_DIR="${SAMPLE_DIR}/activations_${LAYER}"

        mkdir -p "${ACTIVATION_DIR}"

        # Check if examples exist
        if [ ! -f "${INTERMEDIATE_DIR}/make_examples.tfrecord-00000-of-00001.gz" ]; then
            echo "  ERROR: Examples not found for ${SAMPLE} at ${INTERMEDIATE_DIR}, skipping"
            continue
        fi

        # Check if activations already extracted for this layer
        if [ -n "$(ls -A ${ACTIVATION_DIR} 2>/dev/null)" ]; then
            echo "  ${LAYER} activations already exist, skipping"
            continue
        fi

        echo "  Running call_variants_hooked.py for ${LAYER}..."
        conda run -n dv_interp python3 ${WORK_DIR}/scripts/call_variants_hooked.py \
          --examples "${EXAMPLES}" \
          --checkpoint "${WORK_DIR}/model/wgs" \
          --outfile "${OUTFILE}" \
          --activation_cache_dir "${ACTIVATION_DIR}" \
          --hook_layers "${LAYER}" \
          --batch_size 512 \
          --max_cache_entries 10000 || {
            echo "  ERROR: call_variants_hooked failed for ${SAMPLE} ${LAYER}, skipping"
            continue
        }

        echo "  ✓ ${LAYER} activations extracted for ${SAMPLE}"
    done

    echo ""
    echo "✓ ${LAYER} extraction complete"
done

echo ""
echo "=========================================="
echo "EXTRACTION COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Aggregate activations for each layer:"
for LAYER in "${LAYERS[@]}"; do
    echo "     python3 plotting/aggregate_activations.py \\"
    echo "       --input_dirs ${EMBEDDINGS_BASE}/*/activations_${LAYER} \\"
    echo "       --output_dir data/1kg_combined_cache_${LAYER}"
done
echo ""
echo "  2. Run clustering analysis on each layer to compare"
echo ""
