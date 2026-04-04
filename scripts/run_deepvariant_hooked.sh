#!/bin/bash
# Run DeepVariant with activation hooks.
#
# Steps 1 and 3 (make_examples, postprocess_variants) run inside a
# Singularity container because they depend on C++ extensions.
# Step 2 (call_variants) runs natively using this repo's
# activation_hooks.HookedModel so activations are captured in a single
# forward pass and cached as .npz files on the host.
#
# Prerequisites:
#   - Singularity installed
#   - A local model checkpoint. Download with:
#       mkdir -p model/wgs
#       gcloud storage cp 'gs://deepvariant/models/DeepVariant/1.9.0/checkpoints/wgs/*' model/wgs/
#   - Python deps: tensorflow, numpy, absl-py
#
# Usage:
#   bash scripts/run_deepvariant_hooked.sh [OPTIONS]
#
# Required flags:
#   --ref           Path to reference FASTA
#   --reads         Path to aligned reads BAM
#   --output_dir    Directory for all outputs
#   --checkpoint    Local path to model checkpoint (.ckpt prefix or directory)
#
# Optional flags:
#   --bin_version       DeepVariant version (default: 1.9.0)
#   --model_type        WGS|WES|PACBIO|ONT_R104|HYBRID_PACBIO_ILLUMINA (default: WGS)
#   --regions           Genomic regions to process
#   --num_shards        CPU cores for make_examples (default: 1)
#   --hook_layers       Comma-separated layer names to hook (default: mixed5)
#   --batch_size        Batch size for inference (default: 512)
#   --max_cache_entries Max activations to hold in memory (default: 10000)

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
BIN_VERSION="1.9.0"
MODEL_TYPE="WGS"
REGIONS=""
NUM_SHARDS=1
HOOK_LAYERS="mixed5"
BATCH_SIZE=512
MAX_CACHE_ENTRIES=10000
REF=""
READS=""
OUTPUT_DIR=""
CHECKPOINT=""

# ── Parse flags ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
  --bin_version)
    BIN_VERSION="$2"
    shift 2
    ;;
  --model_type)
    MODEL_TYPE="$2"
    shift 2
    ;;
  --ref)
    REF="$2"
    shift 2
    ;;
  --reads)
    READS="$2"
    shift 2
    ;;
  --output_dir)
    OUTPUT_DIR="$2"
    shift 2
    ;;
  --checkpoint)
    CHECKPOINT="$2"
    shift 2
    ;;
  --regions)
    REGIONS="$2"
    shift 2
    ;;
  --num_shards)
    NUM_SHARDS="$2"
    shift 2
    ;;
  --hook_layers)
    HOOK_LAYERS="$2"
    shift 2
    ;;
  --batch_size)
    BATCH_SIZE="$2"
    shift 2
    ;;
  --max_cache_entries)
    MAX_CACHE_ENTRIES="$2"
    shift 2
    ;;
  *)
    echo "Unknown flag: $1"
    exit 1
    ;;
  esac
done

if [[ -z "${REF}" || -z "${READS}" || -z "${OUTPUT_DIR}" || -z "${CHECKPOINT}" ]]; then
  echo "Error: --ref, --reads, --output_dir, and --checkpoint are required."
  exit 1
fi

# ── Derived paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INTERMEDIATE_DIR="${OUTPUT_DIR}/intermediate_results_dir"
ACTIVATION_CACHE_DIR="${OUTPUT_DIR}/activation_cache"
EXAMPLES="${INTERMEDIATE_DIR}/make_examples.tfrecord@${NUM_SHARDS}.gz"
CVO_OUTPUT="${INTERMEDIATE_DIR}/call_variants_output.tfrecord.gz"
OUTPUT_VCF="${OUTPUT_DIR}/output.vcf.gz"
OUTPUT_GVCF="${OUTPUT_DIR}/output.g.vcf.gz"

MODEL_TYPE_LOWER=$(echo "${MODEL_TYPE}" | tr '[:upper:]' '[:lower:]')
CONTAINER_MODEL="/opt/models/${MODEL_TYPE_LOWER}"

mkdir -p "${INTERMEDIATE_DIR}" "${ACTIVATION_CACHE_DIR}"

echo "========================================"
echo " DeepVariant Hooked Pipeline"
echo "========================================"
echo " Version:      ${BIN_VERSION}"
echo " Model type:   ${MODEL_TYPE}"
echo " Checkpoint:   ${CHECKPOINT}"
echo " Hook layers:  ${HOOK_LAYERS}"
echo " Ref:          ${REF}"
echo " Reads:        ${READS}"
echo " Output dir:   ${OUTPUT_DIR}"
echo "========================================"

# ── Step 1: make_examples (Singularity — requires C++ extensions) ────────────
echo ""
echo "=== Step 1/3: make_examples (singularity) ==="

MAKE_EXAMPLES_CMD=(
  docker run --rm -v /Users:/Users -v /usr/lib/locale/:/usr/lib/locale/
  "google/deepvariant:${BIN_VERSION}"
  /opt/deepvariant/bin/make_examples
  --mode calling
  --ref "${REF}"
  --reads "${READS}"
  --examples "${EXAMPLES}"
  --checkpoint "${CONTAINER_MODEL}"
)
if [[ -n "${REGIONS}" ]]; then
  MAKE_EXAMPLES_CMD+=(--regions "${REGIONS}")
fi

echo "Running: ${MAKE_EXAMPLES_CMD[*]}"
"${MAKE_EXAMPLES_CMD[@]}"

echo "make_examples complete."

# ── Step 2: call_variants with activation hooks (native Python) ──────────────
echo ""
echo "=== Step 2/3: call_variants (hooked — native) ==="

EXAMPLES_PATTERN="${INTERMEDIATE_DIR}/make_examples.tfrecord@${NUM_SHARDS}.gz"

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
  python3 "${REPO_ROOT}/scripts/call_variants_hooked.py" \
  --examples "${EXAMPLES_PATTERN}" \
  --checkpoint "${CHECKPOINT}" \
  --outfile "${CVO_OUTPUT}" \
  --activation_cache_dir "${ACTIVATION_CACHE_DIR}" \
  --hook_layers "${HOOK_LAYERS}" \
  --batch_size "${BATCH_SIZE}" \
  --max_cache_entries "${MAX_CACHE_ENTRIES}"

echo "call_variants (hooked) complete."
echo "Activations cached to: ${ACTIVATION_CACHE_DIR}"

# ── Step 3: postprocess_variants (Singularity — requires C++ extensions) ─────
echo ""
echo "=== Step 3/3: postprocess_variants (singularity) ==="

docker run --rm -v /Users:/Users -v /usr/lib/locale/:/usr/lib/locale/ \
  "google/deepvariant:${BIN_VERSION}" \
  /opt/deepvariant/bin/postprocess_variants \
  --ref "${REF}" \
  --infile "${CVO_OUTPUT}" \
  --outfile "${OUTPUT_VCF}" \
  --gvcf_outfile "${OUTPUT_GVCF}"

echo "postprocess_variants complete."

echo ""
echo "========================================"
echo " Pipeline complete!"
echo "========================================"
echo " VCF:          ${OUTPUT_VCF}"
echo " gVCF:         ${OUTPUT_GVCF}"
echo " Activations:  ${ACTIVATION_CACHE_DIR}/"
echo "========================================"
