#!/usr/bin/env python3
"""Parse UCSC RepeatMasker + SimpleRepeat tables → flat repeat TSV (one-time, cached).

Sources:
  rmsk_chr20.txt.gz       — RepeatMasker (SINEs, LINEs, DNA transposons, LTR, etc.)
  simpleRepeat_chr20.txt.gz — Tandem Repeats Finder (SSRs/microsatellites)

Output: data/annotations/repeats_chr20.tsv.gz
Columns: chrom, start0, end0, repeat_class, repeat_family, repeat_name

repeat_class normalized values:
  SINE, LINE, DNA, LTR, Satellite, Low_complexity, SSR, other_repeat

Usage:
    /opt/homebrew/Cellar/deepvariant/1.9.0/libexec/venv/bin/python3 \
        parse_repeatmasker.py
"""

import os
import gzip
import csv
from collections import Counter

REPO_DIR     = os.path.dirname(os.path.abspath(__file__))
ANNOT_DIR    = os.path.join(REPO_DIR, 'data', 'annotations')
RMSK_FILE    = os.path.join(ANNOT_DIR, 'rmsk_chr20.txt.gz')
SIMPLE_FILE  = os.path.join(ANNOT_DIR, 'simpleRepeat_chr20.txt.gz')
OUTPUT_TSV   = os.path.join(ANNOT_DIR, 'repeats_chr20.tsv.gz')

# repClass normalization: map raw UCSC repClass → our canonical label
# rmsk repClass values: SINE, LINE, LTR, DNA, Simple_repeat, Low_complexity,
#                       Satellite, RNA, RC, rRNA, tRNA, snRNA, scRNA, Unknown, Other
CLASS_MAP = {
    'SINE':           'SINE',
    'LINE':           'LINE',
    'LTR':            'LTR',
    'DNA':            'DNA',
    'Satellite':      'Satellite',
    'Low_complexity': 'Low_complexity',
    'Simple_repeat':  'SSR',   # rmsk simple repeats (period > 12 mostly)
    'RNA':            'other_repeat',
    'RC':             'other_repeat',
    'rRNA':           'other_repeat',
    'tRNA':           'other_repeat',
    'snRNA':          'other_repeat',
    'scRNA':          'other_repeat',
    'srpRNA':         'other_repeat',
    'Unknown':        'other_repeat',
    'Other':          'other_repeat',
}

rows = []

# ── Parse rmsk ────────────────────────────────────────────────────────────────
# Columns: bin swScore milliDiv milliDel milliIns genoName genoStart genoEnd
#          genoLeft strand repName repClass repFamily repStart repEnd repLeft id
print(f'Parsing {RMSK_FILE} ...')
n_rmsk = 0
with gzip.open(RMSK_FILE, 'rt') as f:
    for line in f:
        cols = line.rstrip('\n').split('\t')
        if len(cols) < 13:
            continue
        chrom     = cols[5]
        start0    = int(cols[6])   # already 0-based
        end0      = int(cols[7])
        rep_name  = cols[10]
        rep_class = cols[11]
        rep_family = cols[12]
        if chrom != 'chr20':
            continue
        norm_class = CLASS_MAP.get(rep_class, 'other_repeat')
        rows.append((chrom, start0, end0, norm_class, rep_family, rep_name))
        n_rmsk += 1

print(f'  {n_rmsk} rmsk elements on chr20')

# ── Parse simpleRepeat ────────────────────────────────────────────────────────
# Columns: bin chrom chromStart chromEnd ... sequence
# We only need chrom, chromStart, chromEnd
print(f'Parsing {SIMPLE_FILE} ...')
n_simple = 0
with gzip.open(SIMPLE_FILE, 'rt') as f:
    for line in f:
        cols = line.rstrip('\n').split('\t')
        if len(cols) < 4:
            continue
        chrom  = cols[1]
        start0 = int(cols[2])
        end0   = int(cols[3])
        if chrom != 'chr20':
            continue
        # simpleRepeat sequence in col 16 if available (period ≤12 microsatellites)
        rep_name = cols[16] if len(cols) > 16 else 'simple_repeat'
        rows.append((chrom, start0, end0, 'SSR', 'Simple_repeat', rep_name))
        n_simple += 1

print(f'  {n_simple} simpleRepeat elements on chr20')

# ── Write TSV.gz ──────────────────────────────────────────────────────────────
header = ['chrom', 'start0', 'end0', 'repeat_class', 'repeat_family', 'repeat_name']
with gzip.open(OUTPUT_TSV, 'wt', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(header)
    writer.writerows(rows)

class_counts = Counter(r[3] for r in rows)
print(f'\nRepeat class distribution:')
for k, v in sorted(class_counts.items(), key=lambda x: -x[1]):
    print(f'  {k:15s}: {v:6d}')
print(f'\nTotal: {len(rows)} repeat elements → {OUTPUT_TSV}')
