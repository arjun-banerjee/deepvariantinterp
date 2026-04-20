#!/bin/bash
# Truth set benchmark: run standard and hooked DeepVariant on GIAB quick-start
# data, evaluate both with hap.py, and compare results.
#
# This uses the NA12878 quick-start bundle (chr20, 10kb region) from Google
# Cloud Storage. The GIAB truth set is NIST v3.3.2 restricted to that region.
#
# Expected outcome: both pipelines produce identical (or near-identical)
# hap.py metrics — 100% recall/precision on ~50 variants — proving the
# activation-hooking pathway does not degrade variant call accuracy.
#
# Prerequisites:
#   - Docker installed and running
#   - For hooked pipeline: Python 3 with tensorflow, numpy, absl-py
#   - For hooked pipeline: a local model checkpoint (--checkpoint flag)
#
# Usage:
#   bash scripts/run_truth_set_benchmark.sh --checkpoint /path/to/model/wgs
#
# Optional flags:
#   --base_dir          Working directory (default: ./truth_set_benchmark)
#   --input_dir         Path to existing quickstart-testdata directory (skips download)
#   --bin_version       DeepVariant Docker tag (default: 1.9.0)
#   --skip_standard     Skip the standard pipeline run
#   --skip_hooked       Skip the hooked pipeline run

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
BIN_VERSION="1.9.0"
HAPPY_VERSION="v0.3.12"
BASE_DIR="${PWD}/truth_set_benchmark"
CHECKPOINT=""
INPUT_DIR=""
PYTHON=""
SKIP_STANDARD=false
SKIP_HOOKED=false

REGION="chr20:10,000,000-10,010,000"
REGION_HAPPY="chr20:10000000-10010000"

# ── Parse flags ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base_dir)
      BASE_DIR="$2"
      shift 2
      ;;
    --bin_version)
      BIN_VERSION="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --input_dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON="$2"
      shift 2
      ;;
    --skip_standard)
      SKIP_STANDARD=true
      shift
      ;;
    --skip_hooked)
      SKIP_HOOKED=true
      shift
      ;;
    *)
      echo "Error: unknown flag $1" >&2
      exit 1
      ;;
  esac
done

# Auto-detect conda Python if not explicitly set.
if [[ -z "${PYTHON}" ]]; then
  if command -v conda &>/dev/null && conda run -n deepvariant python -c "" 2>/dev/null; then
    PYTHON="conda run --no-capture-output -n deepvariant python"
  else
    PYTHON="python3"
  fi
fi

if [[ "${SKIP_HOOKED}" == "false" && -z "${CHECKPOINT}" ]]; then
  echo "Error: --checkpoint is required unless --skip_hooked is set." >&2
  echo "  Download a checkpoint with:" >&2
  echo "    mkdir -p model/wgs" >&2
  echo "    gcloud storage cp 'gs://deepvariant/models/DeepVariant/${BIN_VERSION}/checkpoints/wgs/*' model/wgs/" >&2
  exit 1
fi

# ── Derived paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

STANDARD_OUTPUT_DIR="${BASE_DIR}/output_standard"
HOOKED_OUTPUT_DIR="${BASE_DIR}/output_hooked"
HAPPY_STANDARD_DIR="${BASE_DIR}/happy_standard"
HAPPY_HOOKED_DIR="${BASE_DIR}/happy_hooked"

REF_FASTA="ucsc.hg19.chr20.unittest.fasta"
BAM_FILE="NA12878_S1.chr20.10_10p1mb.bam"
TRUTH_VCF="test_nist.b37_chr20_100kbp_at_10mb.vcf.gz"
TRUTH_BED="test_nist.b37_chr20_100kbp_at_10mb.bed"

DATA_URL="https://storage.googleapis.com/deepvariant/quickstart-testdata"

# ── Locate or download test data ────────────────────────────────────────────
# If --input_dir was not provided, search well-known locations before downloading.
if [[ -z "${INPUT_DIR}" ]]; then
  SEARCH_PATHS=(
    "${REPO_ROOT}/quickstart-testdata"
    "${REPO_ROOT}/../quickstart-testdata"
    "${PWD}/quickstart-testdata"
    "${HOME}/quickstart-testdata"
  )
  for candidate in "${SEARCH_PATHS[@]}"; do
    if [[ -f "${candidate}/${BAM_FILE}" && -f "${candidate}/${REF_FASTA}" ]]; then
      INPUT_DIR="$(cd "${candidate}" && pwd)"
      break
    fi
  done
fi

if [[ -n "${INPUT_DIR}" && -f "${INPUT_DIR}/${BAM_FILE}" ]]; then
  echo ""
  echo "=== Using existing test data at ${INPUT_DIR} ==="
else
  INPUT_DIR="${BASE_DIR}/input"
  echo ""
  echo "=== Downloading quick-start test data to ${INPUT_DIR} ==="
  mkdir -p "${INPUT_DIR}"

  FILES=(
    "${BAM_FILE}"
    "${BAM_FILE}.bai"
    "${TRUTH_BED}"
    "${TRUTH_VCF}"
    "${TRUTH_VCF}.tbi"
    "${REF_FASTA}"
    "${REF_FASTA}.fai"
    "${REF_FASTA}.gz"
    "${REF_FASTA}.gz.fai"
    "${REF_FASTA}.gz.gzi"
  )

  for f in "${FILES[@]}"; do
    if [[ ! -f "${INPUT_DIR}/${f}" ]]; then
      echo "  Downloading ${f}..."
      curl -sS -o "${INPUT_DIR}/${f}" "${DATA_URL}/${f}"
    else
      echo "  Already exists: ${f}"
    fi
  done
  echo "Download complete."
fi

echo "========================================"
echo " DeepVariant Truth Set Benchmark"
echo "========================================"
echo " Input data:     ${INPUT_DIR}"
echo " Base dir:       ${BASE_DIR}"
echo " DV version:     ${BIN_VERSION}"
echo " Checkpoint:     ${CHECKPOINT:-'(standard only)'}"
echo " Skip standard:  ${SKIP_STANDARD}"
echo " Skip hooked:    ${SKIP_HOOKED}"
echo " Region:         ${REGION}"
echo "========================================"

# ── Run standard DeepVariant ─────────────────────────────────────────────────
if [[ "${SKIP_STANDARD}" == "false" ]]; then
  echo ""
  echo "=========================================="
  echo " Running STANDARD DeepVariant pipeline"
  echo "=========================================="
  mkdir -p "${STANDARD_OUTPUT_DIR}"

  docker run --rm \
    --platform linux/amd64 \
    -v "${INPUT_DIR}":"/input" \
    -v "${STANDARD_OUTPUT_DIR}":"/output" \
    "google/deepvariant:${BIN_VERSION}" \
    /opt/deepvariant/bin/run_deepvariant \
    --model_type=WGS \
    --ref="/input/${REF_FASTA}" \
    --reads="/input/${BAM_FILE}" \
    --regions "${REGION}" \
    --output_vcf="/output/output.vcf.gz" \
    --output_gvcf="/output/output.g.vcf.gz" \
    --intermediate_results_dir "/output/intermediate_results_dir" \
    --num_shards=1

  echo "Standard pipeline complete."
else
  echo ""
  echo "=== Skipping standard pipeline (--skip_standard) ==="
fi

# ── Run hooked DeepVariant ───────────────────────────────────────────────────
if [[ "${SKIP_HOOKED}" == "false" ]]; then
  echo ""
  echo "=========================================="
  echo " Running HOOKED DeepVariant pipeline"
  echo "=========================================="

  bash "${SCRIPT_DIR}/run_deepvariant_hooked.sh" \
    --ref "${INPUT_DIR}/${REF_FASTA}" \
    --reads "${INPUT_DIR}/${BAM_FILE}" \
    --output_dir "${HOOKED_OUTPUT_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --regions "${REGION}" \
    --num_shards 1 \
    --bin_version "${BIN_VERSION}" \
    --python "${PYTHON}"

  echo "Hooked pipeline complete."
else
  echo ""
  echo "=== Skipping hooked pipeline (--skip_hooked) ==="
fi

# ── Evaluate with hap.py ────────────────────────────────────────────────────
run_happy() {
  local vcf_dir="$1"
  local happy_dir="$2"
  local label="$3"

  echo ""
  echo "=== Evaluating ${label} pipeline with hap.py ==="
  mkdir -p "${happy_dir}"

  docker run --rm \
    --platform linux/amd64 \
    -v "${INPUT_DIR}":"/input" \
    -v "${vcf_dir}":"/vcf" \
    -v "${happy_dir}":"/happy" \
    "jmcdani20/hap.py:${HAPPY_VERSION}" /opt/hap.py/bin/hap.py \
    "/input/${TRUTH_VCF}" \
    "/vcf/output.vcf.gz" \
    -f "/input/${TRUTH_BED}" \
    -r "/input/${REF_FASTA}" \
    -o "/happy/happy.output" \
    --engine=vcfeval \
    --pass-only \
    -l "${REGION_HAPPY}"

  echo "${label} evaluation complete."
}

if [[ "${SKIP_STANDARD}" == "false" ]]; then
  run_happy "${STANDARD_OUTPUT_DIR}" "${HAPPY_STANDARD_DIR}" "STANDARD"
fi

if [[ "${SKIP_HOOKED}" == "false" ]]; then
  run_happy "${HOOKED_OUTPUT_DIR}" "${HAPPY_HOOKED_DIR}" "HOOKED"
fi

# ── Print results ────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                    TRUTH SET BENCHMARK RESULTS                     ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

print_summary() {
  local happy_dir="$1"
  local label="$2"
  local summary="${happy_dir}/happy.output.summary.csv"

  if [[ ! -f "${summary}" ]]; then
    echo "  [${label}] No results found at ${summary}"
    return
  fi

  echo ""
  echo "── ${label} ──"
  echo ""
  # Print CSV as a readable table using column if available, otherwise cat.
  if command -v column &>/dev/null; then
    column -t -s, < "${summary}"
  else
    cat "${summary}"
  fi
}

if [[ "${SKIP_STANDARD}" == "false" ]]; then
  print_summary "${HAPPY_STANDARD_DIR}" "STANDARD"
fi

if [[ "${SKIP_HOOKED}" == "false" ]]; then
  print_summary "${HAPPY_HOOKED_DIR}" "HOOKED"
fi

# ── Compare pipelines ───────────────────────────────────────────────────────
if [[ "${SKIP_STANDARD}" == "false" && "${SKIP_HOOKED}" == "false" ]]; then
  echo ""
  echo "── COMPARISON ──"
  echo ""

  STANDARD_CSV="${HAPPY_STANDARD_DIR}/happy.output.summary.csv"
  HOOKED_CSV="${HAPPY_HOOKED_DIR}/happy.output.summary.csv"

  if [[ -f "${STANDARD_CSV}" && -f "${HOOKED_CSV}" ]]; then
    # Extract a metric from the PASS row for a given variant type.
    # Finds the column index from the header row, then reads the value.
    extract_metric() {
      local csv="$1"
      local variant_type="$2"
      local metric_name="$3"
      awk -F, -v vt="${variant_type}" -v mn="${metric_name}" '
        NR==1 { for (i=1; i<=NF; i++) if ($i == mn) col=i }
        $1 == vt && $2 == "PASS" && col { print $col }
      ' "${csv}"
    }

    STD_SNP_F1=$(extract_metric "${STANDARD_CSV}" "SNP" "METRIC.F1_Score")
    STD_INDEL_F1=$(extract_metric "${STANDARD_CSV}" "INDEL" "METRIC.F1_Score")
    HOOK_SNP_F1=$(extract_metric "${HOOKED_CSV}" "SNP" "METRIC.F1_Score")
    HOOK_INDEL_F1=$(extract_metric "${HOOKED_CSV}" "INDEL" "METRIC.F1_Score")

    STD_SNP_RECALL=$(extract_metric "${STANDARD_CSV}" "SNP" "METRIC.Recall")
    STD_INDEL_RECALL=$(extract_metric "${STANDARD_CSV}" "INDEL" "METRIC.Recall")
    HOOK_SNP_RECALL=$(extract_metric "${HOOKED_CSV}" "SNP" "METRIC.Recall")
    HOOK_INDEL_RECALL=$(extract_metric "${HOOKED_CSV}" "INDEL" "METRIC.Recall")

    STD_SNP_PREC=$(extract_metric "${STANDARD_CSV}" "SNP" "METRIC.Precision")
    STD_INDEL_PREC=$(extract_metric "${STANDARD_CSV}" "INDEL" "METRIC.Precision")
    HOOK_SNP_PREC=$(extract_metric "${HOOKED_CSV}" "SNP" "METRIC.Precision")
    HOOK_INDEL_PREC=$(extract_metric "${HOOKED_CSV}" "INDEL" "METRIC.Precision")

    printf "  %-18s  %-15s  %-15s\n" "" "Standard" "Hooked"
    printf "  %-18s  %-15s  %-15s\n" "SNP Recall" "${STD_SNP_RECALL:-N/A}" "${HOOK_SNP_RECALL:-N/A}"
    printf "  %-18s  %-15s  %-15s\n" "SNP Precision" "${STD_SNP_PREC:-N/A}" "${HOOK_SNP_PREC:-N/A}"
    printf "  %-18s  %-15s  %-15s\n" "SNP F1" "${STD_SNP_F1:-N/A}" "${HOOK_SNP_F1:-N/A}"
    printf "  %-18s  %-15s  %-15s\n" "INDEL Recall" "${STD_INDEL_RECALL:-N/A}" "${HOOK_INDEL_RECALL:-N/A}"
    printf "  %-18s  %-15s  %-15s\n" "INDEL Precision" "${STD_INDEL_PREC:-N/A}" "${HOOK_INDEL_PREC:-N/A}"
    printf "  %-18s  %-15s  %-15s\n" "INDEL F1" "${STD_INDEL_F1:-N/A}" "${HOOK_INDEL_F1:-N/A}"

    # Check for divergence. Use awk for float comparison.
    THRESHOLD="0.001"
    PASS=true

    check_match() {
      local label="$1"
      local std_val="$2"
      local hook_val="$3"
      if [[ -z "${std_val}" || -z "${hook_val}" ]]; then
        echo "  WARNING: Could not compare ${label} (missing value)."
        PASS=false
        return
      fi
      local diff
      diff=$(awk "BEGIN { d = ${std_val} - ${hook_val}; print (d < 0 ? -d : d) }")
      local ok
      ok=$(awk "BEGIN { print (${diff} <= ${THRESHOLD}) ? 1 : 0 }")
      if [[ "${ok}" != "1" ]]; then
        echo "  FAIL: ${label} differs by ${diff} (threshold: ${THRESHOLD})"
        PASS=false
      fi
    }

    check_match "SNP F1" "${STD_SNP_F1}" "${HOOK_SNP_F1}"
    check_match "INDEL F1" "${STD_INDEL_F1}" "${HOOK_INDEL_F1}"

    echo ""
    if [[ "${PASS}" == "true" ]]; then
      echo "  RESULT: PASS — hooked pipeline matches standard pipeline."
    else
      echo "  RESULT: FAIL — hooked pipeline diverges from standard pipeline."
      exit 1
    fi
  else
    echo "  Cannot compare: one or both summary CSVs are missing."
    exit 1
  fi
fi

echo ""
echo "========================================"
echo " Benchmark complete."
echo " Results stored in: ${BASE_DIR}"
echo "========================================"
