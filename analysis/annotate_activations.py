#!/usr/bin/env python3
"""Annotate variant positions with GENCODE functional class, ENCODE cCREs, and repeat elements.

Reads variants_metadata.csv (output of extract_variant_positions.py) and
annotates each row with:
  functional_class  — highest-priority gene model feature (CDS/UTR/promoter/intron/intergenic)
  gene_name         — overlapping gene (if any)
  encode_regulatory — ENCODE cCRE class (PLS/pELS/dELS/CTCF-only/none)
  repeat_class      — repeat element class (SINE/LINE/SSR/DNA/LTR/Satellite/Low_complexity/none)

Usage:
    /opt/homebrew/Cellar/deepvariant/1.9.0/libexec/venv/bin/python3 \
        annotate_activations.py
"""

import os
import gzip
import csv
import sys

import pandas as pd
from intervaltree import IntervalTree

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURES_TSV  = os.path.join(REPO_DIR, 'data', 'annotations', 'gencode_features_chr20.tsv.gz')
CCRES_BED     = os.path.join(REPO_DIR, 'data', 'annotations', 'encode_ccres_hg38_chr20.bed')
REPEATS_TSV   = os.path.join(REPO_DIR, 'data', 'annotations', 'repeats_chr20.tsv.gz')
VARIANTS_CSV  = os.path.join(REPO_DIR, 'output', 'activations', 'variants_metadata.csv')
OUTPUT_CSV    = os.path.join(REPO_DIR, 'output', 'activations', 'annotated_metadata.csv')

# Gene model priority: lower index = higher priority
GENE_PRIORITY = ['CDS', '5UTR', '3UTR', 'promoter', 'pseudogene', 'exon_nc', 'intron', 'intergenic']

# Repeat class priority: lower index = higher priority
REPEAT_PRIORITY = ['SINE', 'LINE', 'SSR', 'DNA', 'LTR', 'Satellite', 'Low_complexity', 'other_repeat']

# ── Build gene model IntervalTree ─────────────────────────────────────────────
print('Building gene model interval tree...')
gene_tree = IntervalTree()

if not os.path.exists(FEATURES_TSV):
    print(f'ERROR: {FEATURES_TSV} not found — run parse_gencode_gtf.py first')
    sys.exit(1)

gene_body_tree = IntervalTree()
gene_extents = {}

with gzip.open(FEATURES_TSV, 'rt') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        s = int(row['start0'])
        e = int(row['end0'])
        if s >= e:
            continue
        ft    = row['feature_type']
        gname = row['gene_name']
        gene_tree[s:e] = (ft, gname)

        # Track gene extents for intron assignment
        if ft in ('CDS', '5UTR', '3UTR', 'exon', 'exon_nc', 'promoter'):
            if gname not in gene_extents:
                gene_extents[gname] = [s, e]
            else:
                gene_extents[gname][0] = min(gene_extents[gname][0], s)
                gene_extents[gname][1] = max(gene_extents[gname][1], e)

for gname, (s, e) in gene_extents.items():
    if s < e:
        gene_body_tree[s:e] = gname

print(f'  Gene model tree: {len(gene_tree)} intervals')
print(f'  Gene body tree:  {len(gene_body_tree)} genes')

# ── Build ENCODE cCRE IntervalTree ────────────────────────────────────────────
print('Building ENCODE cCRE interval tree...')
ccre_tree = IntervalTree()

if not os.path.exists(CCRES_BED):
    print(f'  WARNING: {CCRES_BED} not found — encode_regulatory will be "none"')
else:
    with open(CCRES_BED) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            cols = line.rstrip('\n').split('\t')
            if len(cols) < 4:
                continue
            chrom, start_s, end_s = cols[0], cols[1], cols[2]
            ccre_raw = cols[5] if len(cols) > 5 else cols[3]
            ccre_class = ccre_raw.split(',')[0]
            s = int(start_s)
            e = int(end_s)
            if s < e:
                ccre_tree[s:e] = ccre_class
    print(f'  cCRE tree: {len(ccre_tree)} intervals')

# ── Build Repeat IntervalTree ─────────────────────────────────────────────────
print('Building repeat element interval tree...')
repeat_tree = IntervalTree()

if not os.path.exists(REPEATS_TSV):
    print(f'  WARNING: {REPEATS_TSV} not found — run parse_repeatmasker.py first')
    print(f'  repeat_class will be "none"')
else:
    with gzip.open(REPEATS_TSV, 'rt') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            s = int(row['start0'])
            e = int(row['end0'])
            if s >= e:
                continue
            repeat_tree[s:e] = row['repeat_class']
    print(f'  Repeat tree: {len(repeat_tree)} intervals')

# ── Annotate variants ─────────────────────────────────────────────────────────
print(f'Annotating {VARIANTS_CSV} ...')

if not os.path.exists(VARIANTS_CSV):
    print(f'ERROR: {VARIANTS_CSV} not found — run extract_variant_positions.py first')
    sys.exit(1)

df = pd.read_csv(VARIANTS_CSV)

def annotate_position(pos_1based):
    """Return (functional_class, gene_name, encode_regulatory, repeat_class)."""
    pos0 = pos_1based - 1

    # Gene model
    hits = gene_tree[pos0]
    functional_class = 'intergenic'
    gene_name = ''

    if hits:
        best_priority = len(GENE_PRIORITY) - 1
        for interval in hits:
            ft, gname = interval.data
            if ft in GENE_PRIORITY:
                p = GENE_PRIORITY.index(ft)
                if p < best_priority:
                    best_priority = p
                    functional_class = ft
                    gene_name = gname
        if functional_class == 'intergenic':
            body_hits = gene_body_tree[pos0]
            if body_hits:
                functional_class = 'intron'
                gene_name = next(iter(body_hits)).data
    else:
        body_hits = gene_body_tree[pos0]
        if body_hits:
            functional_class = 'intron'
            gene_name = next(iter(body_hits)).data

    # ENCODE cCRE
    ccre_hits = ccre_tree[pos0]
    encode_regulatory = next(iter(ccre_hits)).data if ccre_hits else 'none'

    # Repeat class — highest priority repeat overlapping position
    rep_hits = repeat_tree[pos0]
    if rep_hits:
        best_rep = len(REPEAT_PRIORITY)
        rep_class = 'other_repeat'
        for interval in rep_hits:
            rc = interval.data
            if rc in REPEAT_PRIORITY:
                p = REPEAT_PRIORITY.index(rc)
                if p < best_rep:
                    best_rep = p
                    rep_class = rc
    else:
        rep_class = 'none'

    return functional_class, gene_name, encode_regulatory, rep_class

print(f'  Annotating {len(df)} variants ...')
results = df['pos'].apply(annotate_position)
df['functional_class']  = [r[0] for r in results]
df['gene_name']         = [r[1] for r in results]
df['encode_regulatory'] = [r[2] for r in results]
df['repeat_class']      = [r[3] for r in results]

df.to_csv(OUTPUT_CSV, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
from collections import Counter
fc_counts = Counter(df['functional_class'])
er_counts = Counter(df['encode_regulatory'])
rc_counts = Counter(df['repeat_class'])

print(f'\nFunctional class ({len(df)} variants):')
for cls in GENE_PRIORITY:
    n = fc_counts.get(cls, 0)
    pct = 100 * n / max(len(df), 1)
    print(f'  {cls:12s}: {n:5d}  ({pct:4.1f}%)')

print(f'\nENCODE regulatory:')
for cls, n in er_counts.most_common():
    print(f'  {cls:25s}: {n}')

print(f'\nRepeat class:')
for cls in REPEAT_PRIORITY + ['none']:
    n = rc_counts.get(cls, 0)
    pct = 100 * n / max(len(df), 1)
    print(f'  {cls:15s}: {n:5d}  ({pct:4.1f}%)')

print(f'\nSaved → {OUTPUT_CSV}')
