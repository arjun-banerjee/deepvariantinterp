# Chr20 Activation Analysis

End-to-end pipeline for mechanistic interpretability of DeepVariant's InceptionV3 model on HG002 chr20 (GRCh38, 30x 2x250bp).

## Pipeline order

1. `collect_activations.py` — run make_examples TFRecords through the model, hook all 11 mixed layers, write per-batch `.npz` files
2. `extract_variant_positions.py` — decode variant protobufs from TFRecords → `variants_metadata.csv`
3. `parse_gencode_gtf.py` — parse GENCODE v45 GTF → `gencode_features_chr20.tsv.gz`
4. `parse_repeatmasker.py` — parse UCSC rmsk + simpleRepeat → `repeats_chr20.tsv.gz`
5. `annotate_activations.py` — annotate each variant with functional class, ENCODE cCREs, repeat class → `annotated_metadata.csv`
6. `plotting/umap_activations.py` — per-layer: flatten → StandardScaler → PCA(200) → KMeans(k=15) → UMAP(2D)
7. `plot_annotated_umap.py` — 4-panel annotated plots + AMI scores

## Data dependencies (not tracked in git)
- `data/annotations/gencode_features_chr20.tsv.gz`
- `data/annotations/repeats_chr20.tsv.gz`
- `data/annotations/encode_ccres_hg38_chr20.bed`
- `output/intermediate_hg002/make_examples.tfrecord-*-of-*.gz`
- Activation NPZ files (749 files, ~150 GB compressed on external SSD)

## Key findings (HG002 chr20, 191,657 variants)
- Repeat element class (SINE/LINE/SSR/DNA/LTR) is the dominant clustering signal across all 11 layers (AMI 0.09–0.14)
- Peak alignment at mixed7 (AMI=0.141), monotonic decline through mixed8–10
- Gene model and regulatory annotations show near-zero AMI (~0.01) throughout
- Late layers (mixed8–10) compress into a continuous manifold as the network converges on the genotype classification axis
