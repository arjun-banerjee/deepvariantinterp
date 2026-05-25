#!/usr/bin/env python3
"""Extract genomic positions from make_examples TFRecords.

Decodes the variant/encoded protobuf in each TFRecord example to recover
(chrom, pos, ref, alt) for every candidate variant. Output CSV links each
activation row (indexed by npz_file_idx + example_idx) to its position.

Usage:
    /opt/homebrew/Cellar/deepvariant/1.9.0/libexec/venv/bin/python3 \
        extract_variant_positions.py
"""

import os
import sys
import glob

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
FORK_DIR = os.path.join(REPO_DIR, 'deepvariantinterp')
sys.path.insert(0, FORK_DIR)

import tensorflow as tf
import pandas as pd
from third_party.nucleus.protos import variants_pb2

# ── Config ────────────────────────────────────────────────────────────────────
TFRECORD_GLOB = os.path.join(
    REPO_DIR, 'output', 'intermediate_hg002',
    'make_examples.tfrecord-*-of-*.gz'
)
OUTPUT_CSV = os.path.join(REPO_DIR, 'output', 'activations', 'variants_metadata.csv')
BATCH_SIZE = 256   # must match collect_activations.py BATCH_SIZE

# ── Feature spec ──────────────────────────────────────────────────────────────
feature_spec = {
    'image/encoded':              tf.io.FixedLenFeature((), tf.string),
    'variant/encoded':            tf.io.FixedLenFeature((), tf.string),
    'alt_allele_indices/encoded': tf.io.FixedLenFeature((), tf.string),
}

# ── Parse TFRecords ───────────────────────────────────────────────────────────
tfrecord_files = sorted(glob.glob(TFRECORD_GLOB))
if not tfrecord_files:
    # Fallback: single-shard pattern without -of-NNNNN suffix
    alt = os.path.join(REPO_DIR, 'output', 'intermediate',
                       'make_examples.tfrecord-00000-of-00001.gz')
    if os.path.exists(alt):
        tfrecord_files = [alt]

if not tfrecord_files:
    print(f'ERROR: no TFRecord files matched {TFRECORD_GLOB}')
    sys.exit(1)

print(f'Found {len(tfrecord_files)} TFRecord shard(s)')

rows = []
global_idx = 0

for shard_path in tfrecord_files:
    dataset = tf.data.TFRecordDataset(shard_path, compression_type='GZIP')
    for raw_record in dataset:
        feats = tf.io.parse_single_example(raw_record, feature_spec)
        v = variants_pb2.Variant()
        v.ParseFromString(feats['variant/encoded'].numpy())

        npz_file_idx = global_idx // BATCH_SIZE
        example_idx  = global_idx % BATCH_SIZE

        rows.append({
            'global_idx':   global_idx,
            'npz_file_idx': npz_file_idx,
            'example_idx':  example_idx,
            'chrom':        v.reference_name,
            'pos':          v.start + 1,      # 0-based → 1-based (VCF convention)
            'ref':          v.reference_bases,
            'alt':          ','.join(v.alternate_bases),
        })
        global_idx += 1

    print(f'  {os.path.basename(shard_path)}: {global_idx} examples so far')

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f'\nWrote {len(df)} rows → {OUTPUT_CSV}')
print(df[['chrom', 'pos', 'ref', 'alt']].head(10).to_string())
