#!/bin/bash
set -e  # Exit on error

################################################################################
# 1000 Genomes Population Clustering Pipeline - ALL SAMPLES
#
# This script is portable and sets up everything needed automatically.
# It processes ALL 1000 Genomes samples from GCS to extract DeepVariant
# activations and perform population clustering analysis.
#
# Prerequisites (must be pre-installed):
#   - tensorflow, ml_collections, etils, gsutil, docker/singularity
#
# Steps:
#   0. Setup environment and download reference genome
#   1. Discover all BAM files in GCS bucket
#   2. Download BAM files for chr20 slice (10M-10.1M) from GCS
#   3. Reheader BAMs from "20" to "chr20" for UCSC reference compatibility
#   4. Run DeepVariant make_examples
#   5. Run call_variants_hooked.py to extract activations
#   6. Aggregate activations from all samples
#   7. Run clustering analysis with UMAP and K-means
################################################################################

# Configuration
LAYER="mixed5"                         # DeepVariant layer to extract (mixed0-mixed10)
REGION_GCS="20:10000000-10100000"      # Region for GCS download (no "chr" prefix)
REGION_UCSC="chr20:10000000-10100000"  # Region for DeepVariant (with "chr" prefix)
GCS_BUCKET="gs://brain-genomics-public/research/cohort/1KGP/grch37_bams/"

# Directories (relative to script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(dirname "${SCRIPT_DIR}")"  # Project root
BAM_DIR="${WORK_DIR}/data/1kg_bams"
OUTPUT_BASE="${WORK_DIR}/data/1kg_embeddings"
COMBINED_CACHE="${WORK_DIR}/data/1kg_combined_cache_${LAYER}"
RESULTS_DIR="${WORK_DIR}/data/results"
REF_DIR="${WORK_DIR}/data/reference"
REF="${REF_DIR}/chr20_ucsc.fasta"
SAMPLE_LIST_FILE="${WORK_DIR}/data/all_1kg_samples.txt"

# Create directories
mkdir -p "${BAM_DIR}"
mkdir -p "${OUTPUT_BASE}"
mkdir -p "${COMBINED_CACHE}"
mkdir -p "${RESULTS_DIR}"
mkdir -p "${REF_DIR}"

echo "=========================================="
echo "1000 Genomes Population Clustering Pipeline"
echo "Processing ALL samples from GCS"
echo "=========================================="
echo "Working directory: ${WORK_DIR}"
echo "GCS Bucket: ${GCS_BUCKET}"
echo "Region (GCS): ${REGION_GCS}"
echo "Region (UCSC): ${REGION_UCSC}"
echo ""

################################################################################
# Step -1: Setup and validation
################################################################################
echo "=========================================="
echo "STEP -1: Setting up environment"
echo "=========================================="

# Check for Python packages (install if missing)
echo "Checking Python dependencies..."
python3 -c "import numpy" 2>/dev/null || { echo "Installing numpy..."; pip install numpy; }
python3 -c "import pandas" 2>/dev/null || { echo "Installing pandas..."; pip install pandas; }
python3 -c "import sklearn" 2>/dev/null || { echo "Installing scikit-learn..."; pip install scikit-learn; }
python3 -c "import umap" 2>/dev/null || { echo "Installing umap-learn..."; pip install umap-learn; }
python3 -c "import plotly" 2>/dev/null || { echo "Installing plotly..."; pip install plotly; }
python3 -c "import matplotlib" 2>/dev/null || { echo "Installing matplotlib..."; pip install matplotlib; }
python3 -c "import absl" 2>/dev/null || { echo "Installing absl-py..."; pip install absl-py; }
python3 -c "from google.protobuf import text_format" 2>/dev/null || { echo "Installing protobuf..."; pip install protobuf; }

# Check for samtools
if ! command -v samtools &> /dev/null; then
    echo "ERROR: samtools not found. Please install samtools:"
    echo "  conda install -c bioconda samtools"
    echo "  OR: brew install samtools (macOS)"
    echo "  OR: apt-get install samtools (Ubuntu/Debian)"
    exit 1
fi
echo "✓ samtools found: $(samtools --version | head -n1)"

# Check for Singularity or Docker
USE_SINGULARITY=false
if command -v singularity &> /dev/null; then
    echo "✓ Using Singularity for DeepVariant"
    USE_SINGULARITY=true
elif command -v docker &> /dev/null; then
    echo "✓ Using Docker for DeepVariant"
else
    echo "ERROR: Neither Singularity nor Docker found. Please install one:"
    echo "  https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Download reference genome if needed
if [ ! -f "${REF}" ]; then
    echo "Downloading chr20 reference genome from UCSC..."
    wget -O "${REF_DIR}/chr20.fa.gz" 'http://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/chr20.fa.gz'
    gunzip "${REF_DIR}/chr20.fa.gz"
    mv "${REF_DIR}/chr20.fa" "${REF}"
    echo "✓ Reference genome downloaded to ${REF}"
else
    echo "✓ Reference genome found: ${REF}"
fi

# Index reference if needed
if [ ! -f "${REF}.fai" ]; then
    echo "Indexing reference genome..."
    samtools faidx "${REF}"
    echo "✓ Reference genome indexed"
else
    echo "✓ Reference genome index found"
fi

echo ""
echo "✓ Environment setup complete"
echo "Reference: ${REF}"
echo ""

################################################################################
# Step 0: Discover all samples in GCS bucket
################################################################################
echo "=========================================="
echo "STEP 0: Discovering samples in GCS bucket"
echo "=========================================="

echo "Listing all BAM files from GCS..."
gsutil ls "${GCS_BUCKET}" | grep "\.bam$" | grep -v "\.bai$" > "${SAMPLE_LIST_FILE}"

TOTAL_BAMS=$(wc -l < "${SAMPLE_LIST_FILE}")
echo "Found ${TOTAL_BAMS} BAM files"

# Extract sample IDs from BAM filenames
# Typical format: HG00096.mapped.ILLUMINA.bwa.GBR.low_coverage.20120522.bam
# We extract the first part before the first dot
echo "Extracting sample IDs..."
SAMPLES=()
while IFS= read -r bam_path; do
    basename=$(basename "${bam_path}")
    sample_id=$(echo "${basename}" | cut -d'.' -f1)
    SAMPLES+=("${sample_id}")
done < "${SAMPLE_LIST_FILE}"

echo "Total samples to process: ${#SAMPLES[@]}"
echo ""

# Save sample list for reference
echo "# All 1KG sample IDs" > "${WORK_DIR}/data/all_sample_ids.txt"
printf '%s\n' "${SAMPLES[@]}" >> "${WORK_DIR}/data/all_sample_ids.txt"

################################################################################
# Step 1: Download and prepare BAM files
################################################################################
echo "=========================================="
echo "STEP 1: Downloading and preparing BAM files"
echo "=========================================="

SAMPLE_COUNT=0
for SAMPLE in "${SAMPLES[@]}"; do
    ((SAMPLE_COUNT++))
    echo ""
    echo "[${SAMPLE_COUNT}/${#SAMPLES[@]}] Processing sample: ${SAMPLE}"

    # Find full BAM filename from our list
    FULL_BAM=$(grep "/${SAMPLE}\." "${SAMPLE_LIST_FILE}" | head -n 1)

    if [ -z "${FULL_BAM}" ]; then
        echo "  ERROR: Could not find BAM for ${SAMPLE}"
        continue
    fi

    BASENAME=$(basename "${FULL_BAM}")
    echo "  Found: ${BASENAME}"

    # Download slice
    SLICE_BAM="${BAM_DIR}/${SAMPLE}.chr20_slice.bam"
    if [ -f "${SLICE_BAM}" ]; then
        echo "  Slice already exists, skipping download"
    else
        echo "  Downloading slice for region ${REGION_GCS}..."
        samtools view -b -h "${FULL_BAM}" "${REGION_GCS}" > "${SLICE_BAM}" || {
            echo "  ERROR: Failed to download ${SAMPLE}, skipping"
            continue
        }
        echo "  Downloaded to ${SLICE_BAM}"
    fi

    # Reheader: change "20" to "chr20" for UCSC compatibility
    REHEADERED_BAM="${BAM_DIR}/${SAMPLE}.chr20_slice_reheadered.bam"
    if [ -f "${REHEADERED_BAM}" ]; then
        echo "  Reheadered BAM already exists, skipping"
    else
        echo "  Reheadering BAM (20 -> chr20)..."
        samtools view -H "${SLICE_BAM}" | sed 's/SN:20/SN:chr20/g' > "${BAM_DIR}/${SAMPLE}.header.txt"
        samtools reheader "${BAM_DIR}/${SAMPLE}.header.txt" "${SLICE_BAM}" > "${REHEADERED_BAM}"
        samtools index "${REHEADERED_BAM}"
        rm "${BAM_DIR}/${SAMPLE}.header.txt"
        echo "  Created ${REHEADERED_BAM}"
    fi
done

echo ""
echo "Step 1 complete: All BAM files prepared"

################################################################################
# Step 2: Run make_examples for each sample
################################################################################
echo ""
echo "=========================================="
echo "STEP 2: Running make_examples"
echo "=========================================="

SAMPLE_COUNT=0
for SAMPLE in "${SAMPLES[@]}"; do
    ((SAMPLE_COUNT++))
    echo ""
    echo "[${SAMPLE_COUNT}/${#SAMPLES[@]}] Processing sample: ${SAMPLE}"

    SAMPLE_DIR="${OUTPUT_BASE}/${SAMPLE}"
    mkdir -p "${SAMPLE_DIR}"

    REHEADERED_BAM="${BAM_DIR}/${SAMPLE}.chr20_slice_reheadered.bam"
    EXAMPLES_OUT="${SAMPLE_DIR}/make_examples.tfrecord@1.gz"

    if [ -f "${SAMPLE_DIR}/make_examples.tfrecord-00000-of-00001.gz" ]; then
        echo "  Examples already exist, skipping"
        continue
    fi

    if [ ! -f "${REHEADERED_BAM}" ]; then
        echo "  ERROR: Reheadered BAM not found: ${REHEADERED_BAM}"
        continue
    fi

    echo "  Running make_examples..."
    if [ "${USE_SINGULARITY}" = true ]; then
        singularity exec docker://google/deepvariant:1.9.0 \
          /opt/deepvariant/bin/make_examples \
            --mode calling \
            --ref "${REF}" \
            --reads "${REHEADERED_BAM}" \
            --regions "${REGION_UCSC}" \
            --examples "${EXAMPLES_OUT}" \
            --channel_list "read_base,base_quality,mapping_quality,strand,read_supports_variant,base_differs_from_ref,insert_size" || {
            echo "  ERROR: make_examples failed for ${SAMPLE}, skipping"
            continue
        }
    else
        docker run \
          -v "${WORK_DIR}:${WORK_DIR}" \
          google/deepvariant:1.9.0 \
          /opt/deepvariant/bin/make_examples \
            --mode calling \
            --ref "${REF}" \
            --reads "${REHEADERED_BAM}" \
            --regions "${REGION_UCSC}" \
            --examples "${EXAMPLES_OUT}" \
            --channel_list "read_base,base_quality,mapping_quality,strand,read_supports_variant,base_differs_from_ref,insert_size" || {
            echo "  ERROR: make_examples failed for ${SAMPLE}, skipping"
            continue
        }
    fi

    echo "  make_examples complete for ${SAMPLE}"
done

echo ""
echo "Step 2 complete: All examples generated"

################################################################################
# Step 3: Run call_variants_hooked.py to extract activations
################################################################################
echo ""
echo "=========================================="
echo "STEP 3: Extracting activations with call_variants_hooked.py"
echo "=========================================="

SAMPLE_COUNT=0
for SAMPLE in "${SAMPLES[@]}"; do
    ((SAMPLE_COUNT++))
    echo ""
    echo "[${SAMPLE_COUNT}/${#SAMPLES[@]}] Processing sample: ${SAMPLE}"

    SAMPLE_DIR="${OUTPUT_BASE}/${SAMPLE}"
    EXAMPLES="${SAMPLE_DIR}/make_examples.tfrecord@1.gz"
    OUTFILE="${SAMPLE_DIR}/call_variants_output.tfrecord.gz"
    ACTIVATION_DIR="${SAMPLE_DIR}/activations"

    mkdir -p "${ACTIVATION_DIR}"

    if [ ! -f "${SAMPLE_DIR}/make_examples.tfrecord-00000-of-00001.gz" ]; then
        echo "  ERROR: Examples not found for ${SAMPLE}"
        continue
    fi

    # Check if activations already extracted
    if [ -n "$(ls -A ${ACTIVATION_DIR} 2>/dev/null)" ]; then
        echo "  Activations already exist, skipping"
        continue
    fi

    echo "  Running call_variants_hooked.py..."
    python3 scripts/call_variants_hooked.py \
      --examples "${EXAMPLES}" \
      --checkpoint "model/wgs" \
      --outfile "${OUTFILE}" \
      --activation_cache_dir "${ACTIVATION_DIR}" \
      --hook_layers "${LAYER}" \
      --batch_size 512 \
      --max_cache_entries 10000 || {
        echo "  ERROR: call_variants_hooked failed for ${SAMPLE}, skipping"
        continue
    }

    echo "  Activations extracted for ${SAMPLE}"
done

echo ""
echo "Step 3 complete: All activations extracted"

################################################################################
# Step 4: Aggregate activations from all samples
################################################################################
echo ""
echo "=========================================="
echo "STEP 4: Aggregating activations"
echo "=========================================="

# Build input_dirs argument
INPUT_DIRS=()
PROCESSED_SAMPLES=()
for SAMPLE in "${SAMPLES[@]}"; do
    ACTIVATION_DIR="${OUTPUT_BASE}/${SAMPLE}/activations"
    if [ -d "${ACTIVATION_DIR}" ] && [ -n "$(ls -A ${ACTIVATION_DIR} 2>/dev/null)" ]; then
        INPUT_DIRS+=("${ACTIVATION_DIR}")
        PROCESSED_SAMPLES+=("${SAMPLE}")
    else
        echo "WARNING: No activations found for ${SAMPLE}"
    fi
done

if [ ${#INPUT_DIRS[@]} -eq 0 ]; then
    echo "ERROR: No activation directories found"
    exit 1
fi

echo "Aggregating ${#INPUT_DIRS[@]} sample activation directories..."
python3 plotting/aggregate_activations.py \
  --input_dirs "${INPUT_DIRS[@]}" \
  --output_dir "${COMBINED_CACHE}"

echo "Step 4 complete: Activations aggregated to ${COMBINED_CACHE}"

# Save list of successfully processed samples
echo "# Successfully processed samples" > "${WORK_DIR}/data/processed_sample_ids.txt"
printf '%s\n' "${PROCESSED_SAMPLES[@]}" >> "${WORK_DIR}/data/processed_sample_ids.txt"
echo "Processed ${#PROCESSED_SAMPLES[@]} samples successfully"

################################################################################
# Step 5: Create metadata CSV from 1000 Genomes panel
################################################################################
echo ""
echo "=========================================="
echo "STEP 5: Creating metadata CSV"
echo "=========================================="

# Download 1000 Genomes sample panel if not exists
PANEL_FILE="${WORK_DIR}/data/integrated_call_samples_v3.20130502.ALL.panel"
if [ ! -f "${PANEL_FILE}" ]; then
    echo "Downloading 1000 Genomes panel file..."
    wget -O "${PANEL_FILE}" \
      "ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel"
fi

# Create metadata CSV mapping file_index to sample_id and super_population
METADATA_CSV="${WORK_DIR}/data/1kg_file_mapping_all.csv"
echo "Creating metadata CSV from panel file..."
echo "file_index,sample_id,super_population" > "${METADATA_CSV}"

FILE_INDEX=0
for SAMPLE in "${PROCESSED_SAMPLES[@]}"; do
    # Look up super population from panel file
    SUPER_POP=$(awk -v sample="${SAMPLE}" '$1 == sample {print $3}' "${PANEL_FILE}")

    if [ -z "${SUPER_POP}" ]; then
        echo "  WARNING: No super-population found for ${SAMPLE}, using UNKNOWN"
        SUPER_POP="UNKNOWN"
    fi

    echo "${FILE_INDEX},${SAMPLE},${SUPER_POP}" >> "${METADATA_CSV}"
    ((FILE_INDEX++))
done

echo "Created metadata CSV with ${FILE_INDEX} samples"

################################################################################
# Step 6: Run clustering analysis
################################################################################
echo ""
echo "=========================================="
echo "STEP 6: Running clustering analysis"
echo "=========================================="

OUTPUT_PNG="${RESULTS_DIR}/1kg_all_samples_clustering_kmeans_n100.png"

echo "Running cluster_activations.py on ${#PROCESSED_SAMPLES[@]} samples..."
python3 plotting/cluster_activations.py \
  --cache_dir "${COMBINED_CACHE}" \
  --output "${OUTPUT_PNG}" \
  --layer "${LAYER}" \
  --population_metadata "${METADATA_CSV}" \
  --color_by_population \
  --use_pca \
  --pca_components 200 \
  --umap_intermediate_dim 50 \
  --umap_n_neighbors 100 \
  --n_clusters 5 \
  --random_state 42

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE!"
echo "=========================================="
echo "Total samples in GCS: ${#SAMPLES[@]}"
echo "Successfully processed: ${#PROCESSED_SAMPLES[@]}"
echo ""
echo "Results:"
echo "  2D Visualization: ${OUTPUT_PNG}"
echo "  3D Visualization: ${RESULTS_DIR}/1kg_all_samples_clustering_kmeans_n100_3d.html"
echo "  Embeddings: ${RESULTS_DIR}/1kg_all_samples_clustering_kmeans_n100_results.npz"
echo "  Metrics: ${RESULTS_DIR}/1kg_all_samples_clustering_kmeans_n100_metrics.json"
echo ""
echo "Data locations:"
echo "  Combined cache: ${COMBINED_CACHE}"
echo "  Individual sample data: ${OUTPUT_BASE}/<SAMPLE_ID>/"
echo "  Sample list: ${WORK_DIR}/data/all_sample_ids.txt"
echo "  Processed samples: ${WORK_DIR}/data/processed_sample_ids.txt"
echo "  Metadata: ${METADATA_CSV}"
echo ""
