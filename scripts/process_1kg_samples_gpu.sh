#!/bin/bash
set -e  # Exit on error

################################################################################
# 1000 Genomes Population Clustering Pipeline - ALL SAMPLES
#
# This script is portable and sets up everything needed automatically.
# It processes ALL 1000 Genomes samples from EBI FTP to extract DeepVariant
# activations and perform population clustering analysis.
#
# Prerequisites (must be pre-installed):
#   - tensorflow, ml_collections, etils, docker/singularity
#
# Steps:
#   0. Setup environment and download reference genome
#   1. Discover all BAM files from EBI FTP index
#   2. Download BAM slices for chr20 (10M-10.1M) using samtools + HTTP
#   3. Reheader BAMs from "20" to "chr20" for UCSC reference compatibility
#   4. Run DeepVariant make_examples
#   5. Run call_variants_hooked.py to extract activations
#   6. Aggregate activations from all samples
#   7. Run clustering analysis with UMAP and K-means
################################################################################

# Configuration
REGION_GRCh37="20:10000000-10100000"      # Region for download (GRCh37 format)
REGION_UCSC="chr20:10000000-10100000"     # Region for DeepVariant (UCSC format)
EBI_BASE_URL="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3"
EBI_INDEX_URL="${EBI_BASE_URL}/20130502.phase3.low_coverage.alignment.index"

# Directories (relative to script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(dirname "${SCRIPT_DIR}")"  # Project root
BAM_DIR="${WORK_DIR}/data/1kg_bams"
OUTPUT_BASE="${WORK_DIR}/data/1kg_embeddings"
COMBINED_CACHE="${WORK_DIR}/data/1kg_combined_cache_all"
RESULTS_DIR="${WORK_DIR}/data/results"
REF_DIR="${WORK_DIR}/data/reference"
REF="${REF_DIR}/chr20_ucsc.fasta"
SAMPLE_LIST_FILE="${WORK_DIR}/data/all_1kg_samples.txt"
METADATA_CSV="${WORK_DIR}/data/1kg_file_mapping_all.csv"

# Create directories
mkdir -p "${BAM_DIR}"
mkdir -p "${OUTPUT_BASE}"
mkdir -p "${COMBINED_CACHE}"
mkdir -p "${RESULTS_DIR}"
mkdir -p "${REF_DIR}"

echo "=========================================="
echo "1000 Genomes Population Clustering Pipeline"
echo "Processing ALL samples from EBI FTP"
echo "=========================================="
echo "Working directory: ${WORK_DIR}"
echo "EBI FTP Base: ${EBI_BASE_URL}"
echo "Region (GRCh37): ${REGION_GRCh37}"
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

# Check for curl
if ! command -v curl &> /dev/null; then
    echo "ERROR: curl not found. Please install curl"
    exit 1
fi
echo "✓ curl found"

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
    curl -o "${REF_DIR}/chr20.fa.gz" 'http://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/chr20.fa.gz'
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
# Step 0: Discover all samples from EBI FTP index
################################################################################
echo "=========================================="
echo "STEP 0: Discovering samples from EBI FTP"
echo "=========================================="

echo "Downloading alignment index from EBI..."
curl -s "${EBI_INDEX_URL}" | grep "\.mapped\.ILLUMINA\.bwa\." | awk '{print $1}' > "${SAMPLE_LIST_FILE}"

TOTAL_BAMS=$(wc -l < "${SAMPLE_LIST_FILE}")
echo "Found ${TOTAL_BAMS} mapped BAM files"

# Extract sample IDs and create sample array
echo "Extracting sample IDs..."
SAMPLES=()
while IFS= read -r bam_path; do
    # Path format: data/HG00096/alignment/HG00096.mapped.ILLUMINA.bwa.GBR.low_coverage.20120522.bam
    filename=$(basename "${bam_path}")
    sample_id=$(echo "${filename}" | cut -d'.' -f1)
    SAMPLES+=("${sample_id}|${bam_path}")
done < "${SAMPLE_LIST_FILE}"

echo "Total samples to process: ${#SAMPLES[@]}"
echo ""

# Create sample ID list for reference
echo "# All 1KG sample IDs ($(date))" > "${WORK_DIR}/data/all_sample_ids.txt"
for entry in "${SAMPLES[@]}"; do
    sample_id=$(echo "${entry}" | cut -d'|' -f1)
    echo "${sample_id}" >> "${WORK_DIR}/data/all_sample_ids.txt"
done

################################################################################
# Step 1: Download and prepare BAM files
################################################################################
echo "=========================================="
echo "STEP 1: Downloading and preparing BAM slices"
echo "=========================================="
echo "This will download ONLY the chr20:10M-10.1M region (~500 KB per sample)"
echo "Total data transfer: ~1.2 GB for all ${#SAMPLES[@]} samples"
echo ""

SAMPLE_COUNT=0
PROCESSED_SAMPLES=()

for entry in "${SAMPLES[@]}"; do
    ((SAMPLE_COUNT++))

    # Parse entry: sample_id|bam_path
    SAMPLE=$(echo "${entry}" | cut -d'|' -f1)
    BAM_PATH=$(echo "${entry}" | cut -d'|' -f2)

    echo ""
    echo "[${SAMPLE_COUNT}/${#SAMPLES[@]}] Processing sample: ${SAMPLE}"

    # Build full URL
    FULL_BAM_URL="${EBI_BASE_URL}/${BAM_PATH}"
    echo "  URL: ${FULL_BAM_URL}"

    # Download slice using samtools (only downloads the region, not the full BAM!)
    SLICE_BAM="${BAM_DIR}/${SAMPLE}.chr20_slice.bam"
    if [ -f "${SLICE_BAM}" ]; then
        echo "  Slice already exists, skipping download"
    else
        echo "  Downloading slice for region ${REGION_GRCh37}..."
        if samtools view -b -h "${FULL_BAM_URL}" "${REGION_GRCh37}" > "${SLICE_BAM}" 2>/dev/null; then
            echo "  Downloaded slice ($(du -h "${SLICE_BAM}" | cut -f1))"
        else
            echo "  ERROR: Failed to download ${SAMPLE}, skipping"
            rm -f "${SLICE_BAM}"
            continue
        fi
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

    PROCESSED_SAMPLES+=("${SAMPLE}")
done

echo ""
echo "Step 1 complete: All BAM files prepared"
echo "Processed ${#PROCESSED_SAMPLES[@]} samples successfully"
echo ""

################################################################################
# Step 2: Run make_examples for each sample
################################################################################
echo "=========================================="
echo "STEP 2: Running make_examples"
echo "=========================================="

SAMPLE_COUNT=0
for SAMPLE in "${PROCESSED_SAMPLES[@]}"; do
    ((SAMPLE_COUNT++))
    echo ""
    echo "[${SAMPLE_COUNT}/${#PROCESSED_SAMPLES[@]}] Processing sample: ${SAMPLE}"

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
for SAMPLE in "${PROCESSED_SAMPLES[@]}"; do
    ((SAMPLE_COUNT++))
    echo ""
    echo "[${SAMPLE_COUNT}/${#PROCESSED_SAMPLES[@]}] Processing sample: ${SAMPLE}"

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
      --hook_layers "mixed5" \
      --batch_size 512 \
      --max_cache_entries 10000 || {
        echo "  ERROR: call_variants_hooked failed for ${SAMPLE}, skipping"
        continue
    }

    echo "  Activations extracted to ${ACTIVATION_DIR}"
done

echo ""
echo "Step 3 complete: All activations extracted"

################################################################################
# Step 4: Aggregate activations from all samples
################################################################################
echo ""
echo "=========================================="
echo "STEP 4: Aggregating activations from all samples"
echo "=========================================="

echo "Combining all activation files into ${COMBINED_CACHE}..."
mkdir -p "${COMBINED_CACHE}"

FILE_INDEX=0
echo "file_index,sample_id,super_population" > "${METADATA_CSV}"

for SAMPLE in "${PROCESSED_SAMPLES[@]}"; do
    ACTIVATION_DIR="${OUTPUT_BASE}/${SAMPLE}/activations"

    if [ ! -d "${ACTIVATION_DIR}" ] || [ -z "$(ls -A ${ACTIVATION_DIR} 2>/dev/null)" ]; then
        echo "  WARNING: No activations found for ${SAMPLE}, skipping"
        continue
    fi

    # Copy activation files to combined cache
    for npz_file in "${ACTIVATION_DIR}"/*.npz; do
        if [ -f "${npz_file}" ]; then
            cp "${npz_file}" "${COMBINED_CACHE}/activations_${SAMPLE}.npz"
            echo "${FILE_INDEX},${SAMPLE},UNKNOWN" >> "${METADATA_CSV}"
            ((FILE_INDEX++))
            echo "  Copied ${SAMPLE} (file_index=${FILE_INDEX})"
            break  # Only copy first .npz file per sample
        fi
    done
done

echo ""
echo "Step 4 complete: Aggregated ${FILE_INDEX} samples"
echo "Metadata saved to: ${METADATA_CSV}"
echo "NOTE: super_population is set to UNKNOWN - update this file with actual population labels"
echo ""

################################################################################
# Step 5: Run clustering analysis
################################################################################
echo "=========================================="
echo "STEP 5: Running clustering analysis"
echo "=========================================="

OUTPUT_PNG="${RESULTS_DIR}/1kg_all_samples_clustering_kmeans_n100.png"

echo "Running cluster_activations.py on ${FILE_INDEX} samples..."
python3 plotting/cluster_activations.py \
  --cache_dir "${COMBINED_CACHE}" \
  --output "${OUTPUT_PNG}" \
  --layer mixed5 \
  --population_metadata "${METADATA_CSV}" \
  --color_by_population \
  --use_pca \
  --pca_components 200 \
  --umap_intermediate_dim 20 \
  --umap_n_neighbors 100 \
  --n_clusters 5 \
  --umap_3d \
  --random_state 42

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE!"
echo "=========================================="
echo "Results:"
echo "  - Clustering visualization: ${OUTPUT_PNG}"
echo "  - Combined activations: ${COMBINED_CACHE}/"
echo "  - Sample metadata: ${METADATA_CSV}"
echo "  - Total samples processed: ${FILE_INDEX}"
echo "=========================================="
