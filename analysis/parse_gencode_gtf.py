#!/usr/bin/env python3
"""Parse GENCODE v45 GTF → flat feature TSV (one-time, cached).

Reads gencode.v45.chr20.gtf.gz (chr20 only), extracts exon/CDS/UTR/transcript
features, classifies UTRs as 5UTR/3UTR, and computes promoter intervals from
transcript TSS positions. Saves a flat TSV.gz for fast reloading by
annotate_activations.py.

Usage:
    /opt/homebrew/Cellar/deepvariant/1.9.0/libexec/venv/bin/python3 \
        parse_gencode_gtf.py
"""

import os
import gzip
import csv

REPO_DIR    = os.path.dirname(os.path.abspath(__file__))
GTF_FILE    = os.path.join(REPO_DIR, 'data', 'annotations', 'gencode.v45.chr20.gtf.gz')
OUTPUT_TSV  = os.path.join(REPO_DIR, 'data', 'annotations', 'gencode_features_chr20.tsv.gz')
PROMOTER_BP = 2000    # bp upstream of TSS to call "promoter"
PROMOTER_DN =  200    # bp downstream of TSS included in promoter window

def parse_attrs(attr_string):
    """Parse GTF attribute string into a dict."""
    attrs = {}
    for part in attr_string.strip().rstrip(';').split(';'):
        part = part.strip()
        if not part:
            continue
        try:
            key, val = part.split(' ', 1)
            attrs[key] = val.strip('"')
        except ValueError:
            pass
    return attrs

rows = []

print(f'Parsing {GTF_FILE} ...')
with gzip.open(GTF_FILE, 'rt') as f:
    for line in f:
        if line.startswith('#'):
            continue
        cols = line.rstrip('\n').split('\t')
        if len(cols) < 9:
            continue
        chrom, source, feature, start_s, end_s, score, strand, frame, attrs_s = cols
        start0 = int(start_s) - 1   # GTF is 1-based inclusive → convert to 0-based
        end0   = int(end_s)         # end is already 0-based exclusive after -1+1
        attrs  = parse_attrs(attrs_s)

        gene_name  = attrs.get('gene_name', attrs.get('gene_id', ''))
        gene_type  = attrs.get('gene_type', attrs.get('gene_biotype', ''))
        transcript_id = attrs.get('transcript_id', '')

        if feature == 'CDS':
            rows.append((chrom, start0, end0, 'CDS', gene_name, gene_type, strand))

        elif feature == 'UTR':
            # GENCODE uses "UTR" for both 5' and 3' — classify by strand + transcript context
            # We tag as "UTR" for now; annotate_activations will handle priority
            rows.append((chrom, start0, end0, 'UTR', gene_name, gene_type, strand))

        elif feature == 'exon':
            rows.append((chrom, start0, end0, 'exon', gene_name, gene_type, strand))

        elif feature == 'transcript':
            # Compute promoter window: TSS ± PROMOTER_BP (upstream) + PROMOTER_DN (downstream)
            if strand == '+':
                tss = start0
                prom_start = max(0, tss - PROMOTER_BP)
                prom_end   = tss + PROMOTER_DN
            else:
                tss = end0
                prom_start = max(0, tss - PROMOTER_DN)
                prom_end   = tss + PROMOTER_BP
            rows.append((chrom, prom_start, prom_end, 'promoter', gene_name, gene_type, strand))

print(f'  Parsed {len(rows)} raw feature rows')

# ── Resolve UTR subtype (5UTR vs 3UTR) ───────────────────────────────────────
# Strategy: collect all CDS intervals per gene, then for each UTR check if it
# is upstream (5') or downstream (3') of the CDS block on that strand.
# Build a per-gene CDS range lookup.
cds_by_gene = {}
for chrom, s, e, ftype, gname, gtype, strand in rows:
    if ftype == 'CDS' and gname:
        if gname not in cds_by_gene:
            cds_by_gene[gname] = {'min': s, 'max': e, 'strand': strand}
        else:
            cds_by_gene[gname]['min'] = min(cds_by_gene[gname]['min'], s)
            cds_by_gene[gname]['max'] = max(cds_by_gene[gname]['max'], e)

resolved_rows = []
for chrom, s, e, ftype, gname, gtype, strand in rows:
    if ftype == 'UTR' and gname in cds_by_gene:
        cds = cds_by_gene[gname]
        if strand == '+':
            utr_type = '5UTR' if e <= cds['min'] else '3UTR'
        else:
            utr_type = '5UTR' if s >= cds['max'] else '3UTR'
        resolved_rows.append((chrom, s, e, utr_type, gname, gtype, strand))
    elif ftype == 'UTR':
        # Non-coding gene with UTR annotation — treat as exon_nc
        resolved_rows.append((chrom, s, e, 'exon_nc', gname, gtype, strand))
    else:
        resolved_rows.append((chrom, s, e, ftype, gname, gtype, strand))

# Rename 'exon' for non-coding genes to 'exon_nc'
final_rows = []
protein_coding_genes = {gname for _, _, _, ft, gname, gtype, _ in resolved_rows
                        if ft == 'CDS'}
for chrom, s, e, ftype, gname, gtype, strand in resolved_rows:
    if ftype == 'exon' and gname not in protein_coding_genes:
        ftype = 'exon_nc'
    final_rows.append((chrom, s, e, ftype, gname, gtype, strand))

# ── Add pseudogene gene bodies ────────────────────────────────────────────────
# Extract gene-level intervals for all pseudogene gene_types. These are
# added as feature_type='pseudogene' so annotate_activations.py can build
# a separate IntervalTree or include them in the gene model priority hierarchy.
PSEUDOGENE_TYPES = {
    'processed_pseudogene',
    'unprocessed_pseudogene',
    'transcribed_processed_pseudogene',
    'transcribed_unprocessed_pseudogene',
    'polymorphic_pseudogene',
    'unitary_pseudogene',
    'pseudogene',
}

print('\nExtracting pseudogene gene bodies ...')
pseudogene_rows = []
with gzip.open(GTF_FILE, 'rt') as f:
    for line in f:
        if line.startswith('#'):
            continue
        cols = line.rstrip('\n').split('\t')
        if len(cols) < 9:
            continue
        chrom, source, feature, start_s, end_s, score, strand, frame, attrs_s = cols
        if feature != 'gene':
            continue
        attrs = parse_attrs(attrs_s)
        gene_type = attrs.get('gene_type', attrs.get('gene_biotype', ''))
        if gene_type not in PSEUDOGENE_TYPES:
            continue
        start0 = int(start_s) - 1
        end0   = int(end_s)
        gene_name = attrs.get('gene_name', attrs.get('gene_id', ''))
        pseudogene_rows.append((chrom, start0, end0, 'pseudogene', gene_name, gene_type, strand))

print(f'  Found {len(pseudogene_rows)} pseudogene gene bodies on chr20')
final_rows.extend(pseudogene_rows)

# ── Write TSV.gz ──────────────────────────────────────────────────────────────
header = ['chrom', 'start0', 'end0', 'feature_type', 'gene_name', 'gene_type', 'strand']
with gzip.open(OUTPUT_TSV, 'wt', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(header)
    writer.writerows(final_rows)

from collections import Counter
type_counts = Counter(r[3] for r in final_rows)
print(f'\nFeature counts:')
for k, v in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f'  {k:12s}: {v:6d}')
print(f'\nSaved {len(final_rows)} rows → {OUTPUT_TSV}')
