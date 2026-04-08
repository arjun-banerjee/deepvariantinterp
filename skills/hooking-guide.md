# Activation Hooking Guide

DeepVariant's standard pipeline discards intermediate layer activations after calling variants. This repo replaces Step 2 (`call_variants`) with a hooked version that captures those activations and saves them to disk alongside the normal VCF output.

## How It Works

The standard pipeline has three steps:

```
make_examples → call_variants → postprocess_variants
```

The hooked pipeline replaces Step 2 with a native Python script:

```
make_examples (Docker)
  ↓
call_variants_hooked.py  ← captures activations here
  ↓
postprocess_variants (Docker)
```

During inference, `call_variants_hooked.py` builds a secondary Keras model that outputs both predictions and intermediate layer tensors in a single forward pass. Activations are compressed and written to disk as `.npz` files.

## Running the Full Pipeline

```bash
cd ~/Documents/Code/deepvariantinterp
export INPUT_DIR="${PWD}/quickstart-testdata"

bash scripts/run_deepvariant_hooked.sh \
  --ref "${INPUT_DIR}/ucsc.hg19.chr20.unittest.fasta" \
  --reads "${INPUT_DIR}/NA12878_S1.chr20.10_10p1mb.bam" \
  --output_dir "${PWD}/quickstart-output" \
  --checkpoint "${PWD}/model/wgs/deepvariant.wgs.ckpt" \
  --regions "chr20:10,000,000-10,010,000" \
  --hook_layers mixed5
```

### All Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--ref` | required | Reference FASTA |
| `--reads` | required | Aligned reads BAM |
| `--output_dir` | required | Directory for all outputs |
| `--checkpoint` | required | Path to model checkpoint |
| `--hook_layers` | `mixed5` | Comma-separated layer names to capture |
| `--regions` | (all) | Genomic region(s) to process |
| `--num_shards` | `1` | CPU shards for make_examples |
| `--batch_size` | `512` | Inference batch size |
| `--max_cache_entries` | `10000` | Max activations held in RAM before flushing |
| `--model_type` | `WGS` | WGS, WES, PACBIO, ONT_R104, HYBRID_PACBIO_ILLUMINA |
| `--bin_version` | `1.9.0` | DeepVariant Docker image version |

## Output Files

```
quickstart-output/
├── output.vcf.gz                          # Final variant calls (standard DeepVariant output)
├── output.vcf.gz.tbi                      # VCF index
├── activation_cache/
│   ├── manifest.json                      # Cache metadata
│   └── activations_00000000.npz           # Captured activations (one file per flush)
└── intermediate_results_dir/
    ├── make_examples.tfrecord-*.gz        # Pileup images
    └── call_variants_output-*.tfrecord.gz # Raw variant probabilities
```

### manifest.json

```json
{
  "total_entries": 84,
  "flushed_up_to": 84,
  "in_memory_entries": 0,
  "layer_names": ["mixed5"],
  "timestamp": 1774816288.84
}
```

### activations_*.npz

Each `.npz` file contains one array per hooked layer:

```python
import numpy as np

data = np.load("activation_cache/activations_00000000.npz")
print(list(data.keys()))        # ['mixed5']
print(data['mixed5'].shape)     # (84, 4, 12, 768)
                                # (n_examples, H, W, channels)
```

The shape is `(batch, spatial_H, spatial_W, channels)`. For `mixed5` with the default WGS model the spatial dimensions are 4×12 and there are 768 channels.

## Which Layers Can Be Hooked

The model uses an InceptionV3 backbone. The most useful layers to hook are the 11 "mixed" blocks:

| Layer | Depth | Spatial dims | Channels | Notes |
|-------|-------|-------------|----------|-------|
| `mixed0` | shallow | ~28×28 | 256 | Early features |
| `mixed1` | | ~28×28 | 288 | |
| `mixed2` | | ~28×28 | 288 | |
| `mixed3` | | ~14×14 | 768 | After pooling |
| `mixed4` | | ~14×14 | 768 | |
| `mixed5` | mid | ~4×12 | 768 | **Default** |
| `mixed6` | | ~4×12 | 768 | |
| `mixed7` | | ~4×12 | 768 | |
| `mixed8` | | ~4×12 | 1280 | After pooling |
| `mixed9` | deep | ~4×12 | 2048 | |
| `mixed10` | deepest | ~4×12 | 2048 | Final mixed layer |

Hook multiple layers at once:

```bash
--hook_layers mixed3,mixed5,mixed7,mixed10
```

To get the full list of hookable layers programmatically:

```python
from deepvariant import activation_hooks
from deepvariant import keras_modeling

model = keras_modeling.inceptionv3_with_pretrained_weights(...)
print(activation_hooks.list_hookable_layers(model))
```

## Visualizing Activations with UMAP

Install plotting deps first (must use conda — pip fails to build `llvmlite` from source):

```bash
conda install -n deepVariant -c conda-forge umap-learn matplotlib scikit-learn -y
```

```bash
python plotting/umap_activations.py \
  --cache_dir quickstart-output/activation_cache \
  --layer mixed5 \
  --output umap_mixed5.png
```

| Flag | Default | Description |
|------|---------|-------------|
| `--cache_dir` | required | Path to activation_cache directory |
| `--layer` | first found | Layer name to visualize |
| `--output` | `umap.png` | Output plot path |
| `--n_neighbors` | `15` | UMAP connectivity parameter |
| `--min_dist` | `0.1` | UMAP minimum distance |
| `--max_samples` | `5000` | Subsample if more examples than this |
| `--title` | auto | Plot title |

Produces a scatter plot where each point is one pileup image (variant candidate), colored by batch file index. Similar activations cluster together — useful for identifying outliers or clusters of variant types.

## Loading Activations for Custom Analysis

```python
import numpy as np
from pathlib import Path

cache_dir = Path("quickstart-output/activation_cache")

# Load all batches
all_activations = []
for npz_path in sorted(cache_dir.glob("activations_*.npz")):
    data = np.load(npz_path)
    all_activations.append(data['mixed5'])

activations = np.concatenate(all_activations, axis=0)
# shape: (total_examples, 4, 12, 768)

# Flatten spatial dims for sklearn/umap
flat = activations.reshape(len(activations), -1)
# shape: (total_examples, 36864)
```

## Architecture Reference

```
Input pileup image (100 × 221 × 7 channels)
         ↓
   InceptionV3 backbone
     mixed0  (shallow features)
     mixed1
     mixed2
     mixed3
     mixed4
     mixed5  ← default hook point
     mixed6
     mixed7
     mixed8
     mixed9
     mixed10 (deepest features)
         ↓
   Global Average Pooling
         ↓
   Dropout(0.2)
         ↓
   Dense(3)  →  [hom-ref, het, hom-alt] probabilities
```

The 7 input channels encode: read base, base quality, mapping quality, strand, read supports variant, base differs from ref, and insert size.