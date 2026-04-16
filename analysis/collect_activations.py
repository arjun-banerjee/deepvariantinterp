#!/usr/bin/env python3
"""Collect activations from all 11 InceptionV3 mixed layers of the DeepVariant WGS model.

Reads pileup image TFRecords produced by make_examples, feeds them through
the InceptionV3 model via HookedModel (single forward pass capturing all 11
mixed layers), and writes per-batch .npz activation files to output/activations/.

Each NPZ contains 11 keys (mixed0..mixed10). Analyze each layer independently
using umap_activations.py --layer mixed5 (or any other mixed layer).

Usage:
    /opt/homebrew/Cellar/deepvariant/1.9.0/libexec/venv/bin/python3 \
        collect_activations.py
"""

import os, sys, glob

# Fork source on sys.path so `from deepvariant import ...` works
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
FORK_DIR = os.path.join(REPO_DIR, 'deepvariantinterp')
sys.path.insert(0, FORK_DIR)

import json
import numpy as np
import tensorflow as tf

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT   = os.path.join(FORK_DIR, 'model', 'wgs', 'deepvariant.wgs.ckpt')
_EXAMPLES_GLOB = os.path.join(REPO_DIR, 'output', 'intermediate_hg002',
                               'make_examples.tfrecord-*-of-*.gz')
EXAMPLES     = sorted(glob.glob(_EXAMPLES_GLOB))
if not EXAMPLES:
    raise FileNotFoundError(f'No TFRecord shards found matching: {_EXAMPLES_GLOB}')
print(f'Found {len(EXAMPLES)} TFRecord shard(s).')
OUTPUT_DIR   = os.path.join(REPO_DIR, 'output', 'activations')
BATCH_SIZE   = 256

# ── Layers to hook ────────────────────────────────────────────────────────────
from deepvariant import activation_hooks
LAYERS = list(activation_hooks.INCEPTION_MIXED_LAYERS)  # mixed0..mixed10
print(f'Hooking {len(LAYERS)} layers: {LAYERS}')

# ── Load model ────────────────────────────────────────────────────────────────
from deepvariant import keras_modeling

ckpt_dir = os.path.dirname(CHECKPOINT)
with open(os.path.join(ckpt_dir, 'example_info.json')) as f:
    info = json.load(f)
example_shape = info['shape']   # [100, 221, 7]
print(f'Example shape: {example_shape}')

print(f'Building InceptionV3 ({example_shape})...')
model = keras_modeling.inceptionv3(example_shape)

print(f'Loading weights from {CHECKPOINT}...')
model.load_weights(CHECKPOINT)
print('Weights loaded.')

# ── Build HookedModel (all 11 mixed layers, single forward pass) ──────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
hooked = activation_hooks.HookedModel(
    model=model,
    layer_names=LAYERS,
    cache_dir=OUTPUT_DIR,
    max_cache_entries=50,   # keep only 50 batches in memory; evict + flush aggressively
    single_pass=True,
)
print(f'\nHookedModel ready. Capturing: {LAYERS}')
print(f'Output: {OUTPUT_DIR}\n')

# ── TFRecord dataset ──────────────────────────────────────────────────────────
feature_spec = {
    'image/encoded':              tf.io.FixedLenFeature((), tf.string),
    'variant/encoded':            tf.io.FixedLenFeature((), tf.string),
    'alt_allele_indices/encoded': tf.io.FixedLenFeature((), tf.string),
}

def parse_fn(serialized):
    feats = tf.io.parse_single_example(serialized, feature_spec)
    image = tf.io.decode_raw(feats['image/encoded'], tf.uint8)
    image = tf.reshape(image, example_shape)
    image = (tf.cast(image, tf.float32) - 128.0) / 128.0
    return image, feats['variant/encoded']

dataset = (
    tf.data.TFRecordDataset(EXAMPLES, compression_type='GZIP')
    .map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ── Inference loop ────────────────────────────────────────────────────────────
total = 0
for batch_idx, (images, variants) in enumerate(dataset):
    preds, acts = hooked.predict_and_cache(images)

    if batch_idx == 0:
        print('Layer shapes (first batch):')
        for layer_name, arr in acts.items():
            flat_dim = arr.shape[1] * arr.shape[2] * arr.shape[3] if arr.ndim == 4 else arr.shape[1]
            print(f'  {layer_name}: {arr.shape}  →  flatten: {flat_dim} dims')

    if batch_idx % 50 == 0:
        print(f'Batch {batch_idx:4d} | n={images.shape[0]} | total={total + images.shape[0]}')
    total += images.shape[0]

hooked.finalize()
print(f'\nDone. {total} examples processed → {OUTPUT_DIR}')

# ── Sanity check ──────────────────────────────────────────────────────────────
npz_files = sorted(f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npz'))
if npz_files:
    sample = np.load(os.path.join(OUTPUT_DIR, npz_files[0]))
    print(f'\nSanity check — {npz_files[0]}:')
    for k, v in sample.items():
        print(f'  {k}: shape={v.shape}  dtype={v.dtype}')
