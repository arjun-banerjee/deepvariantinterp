#!/usr/bin/env python3
"""Superpopulation-stratified quality analysis of DeepVariant variant calls.

Parses VCF outputs from per-sample DeepVariant runs, groups variants by
superpopulation, and produces visualizations showing how quality scores
(QUAL), variant type distributions, and genotype confidence vary across
superpopulations.

Usage:
    python plotting/quality_by_superpopulation.py \
      --output_dir 1kg_quality_output \
      --mapping bams/1kg_file_mapping_v3.csv \
      --output 1kg_quality_output/quality_analysis

    # Or point at the existing hooked output:
    python plotting/quality_by_superpopulation.py \
      --output_dir 1kg_hooked_output \
      --mapping bams/1kg_file_mapping_v3.csv \
      --output 1kg_hooked_output/quality_analysis
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--output_dir',
        required=True,
        help=(
            'Root output directory containing per-sample subdirectories '
            '(e.g. 1kg_hooked_output with samples/<SAMPLE>/output.vcf.gz).'
        ),
    )
    p.add_argument(
        '--mapping',
        required=True,
        help=(
            'CSV mapping sample_id to super_population. '
            'Must contain columns: sample_id, super_population. '
            'Optionally file_index for ordering.'
        ),
    )
    p.add_argument(
        '--output', '-o',
        default='quality_analysis',
        help='Output prefix for plots and data (e.g. path/quality_analysis).',
    )
    p.add_argument(
        '--min_qual',
        type=float,
        default=0.0,
        help='Minimum QUAL to include (default: 0, include all).',
    )
    p.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='Figure DPI.',
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# VCF parsing
# ---------------------------------------------------------------------------

def parse_vcf(vcf_path: str, sample_id: str) -> pd.DataFrame:
    """Parse a VCF(.gz) file and extract variant quality information.

    Returns a DataFrame with columns:
        chrom, pos, ref, alt, qual, filter, variant_type, genotype, sample_id
    """
    records = []

    opener = gzip.open if vcf_path.endswith('.gz') else open
    with opener(vcf_path, 'rt') as fh:
        for line in fh:
            if line.startswith('#'):
                continue

            fields = line.rstrip('\n').split('\t')
            if len(fields) < 10:
                continue

            chrom = fields[0]
            pos = int(fields[1])
            ref = fields[3]
            alts = fields[4].split(',')
            qual_str = fields[5]
            filt = fields[6]

            qual = float(qual_str) if qual_str != '.' else 0.0

            # Parse genotype from the first sample column
            fmt = fields[8].split(':')
            sample_vals = fields[9].split(':')
            fmt_dict = dict(zip(fmt, sample_vals))

            gt_raw = fmt_dict.get('GT', './.')
            gt_alleles = gt_raw.replace('|', '/').split('/')
            try:
                gt_ints = [int(a) for a in gt_alleles if a != '.']
            except ValueError:
                gt_ints = []

            if len(gt_ints) == 2:
                if gt_ints[0] == 0 and gt_ints[1] == 0:
                    genotype = 'hom-ref'
                elif gt_ints[0] == gt_ints[1]:
                    genotype = 'hom-alt'
                else:
                    genotype = 'het'
            else:
                genotype = 'unknown'

            # Genotype quality
            gq = float(fmt_dict.get('GQ', '0'))

            # Depth
            dp = int(fmt_dict.get('DP', '0'))

            for alt in alts:
                if alt == '.' or alt == '*':
                    continue

                if len(ref) == 1 and len(alt) == 1:
                    vtype = 'SNP'
                elif len(ref) != len(alt):
                    vtype = 'INDEL'
                else:
                    vtype = 'MNP'

                records.append({
                    'chrom': chrom,
                    'pos': pos,
                    'ref': ref,
                    'alt': alt,
                    'qual': qual,
                    'filter': filt,
                    'variant_type': vtype,
                    'genotype': genotype,
                    'gq': gq,
                    'dp': dp,
                    'sample_id': sample_id,
                })

    return pd.DataFrame(records)


def discover_vcfs(output_dir: str) -> Dict[str, str]:
    """Find per-sample VCF files under output_dir/samples/<sample>/output.vcf.gz."""
    samples_dir = os.path.join(output_dir, 'samples')
    vcfs = {}

    if not os.path.isdir(samples_dir):
        print(f'WARNING: {samples_dir} not found, scanning output_dir directly',
              file=sys.stderr)
        samples_dir = output_dir

    for entry in sorted(os.listdir(samples_dir)):
        sample_path = os.path.join(samples_dir, entry)
        if not os.path.isdir(sample_path):
            continue
        for vcf_name in ('output.vcf.gz', 'output.vcf'):
            vcf_path = os.path.join(sample_path, vcf_name)
            if os.path.isfile(vcf_path):
                vcfs[entry] = vcf_path
                break

    return vcfs


def load_all_variants(
    vcfs: Dict[str, str],
    mapping_df: pd.DataFrame,
    min_qual: float = 0.0,
) -> pd.DataFrame:
    """Parse all VCFs and join with superpopulation labels."""
    frames = []
    sample_to_pop = dict(
        zip(mapping_df['sample_id'], mapping_df['super_population'])
    )

    for sample_id, vcf_path in sorted(vcfs.items()):
        pop = sample_to_pop.get(sample_id)
        if pop is None:
            print(f'  WARNING: {sample_id} not in mapping CSV, skipping')
            continue

        print(f'  Parsing {sample_id} ({pop}): {vcf_path}')
        df = parse_vcf(vcf_path, sample_id)
        df['super_population'] = pop
        frames.append(df)

    if not frames:
        print('ERROR: No VCFs parsed successfully.', file=sys.stderr)
        raise SystemExit(1)

    combined = pd.concat(frames, ignore_index=True)

    if min_qual > 0:
        before = len(combined)
        combined = combined[combined['qual'] >= min_qual]
        print(f'  Filtered QUAL >= {min_qual}: {before} -> {len(combined)} variants')

    return combined


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_statistics(df: pd.DataFrame) -> Dict:
    """Compute summary statistics grouped by superpopulation."""
    stats = {}

    for pop in sorted(df['super_population'].unique()):
        pop_df = df[df['super_population'] == pop]
        pass_df = pop_df[pop_df['filter'] == 'PASS']

        stats[pop] = {
            'n_samples': pop_df['sample_id'].nunique(),
            'samples': sorted(pop_df['sample_id'].unique().tolist()),
            'n_variants_total': len(pop_df),
            'n_variants_pass': len(pass_df),
            'pass_rate': len(pass_df) / max(len(pop_df), 1),
            'qual': {
                'mean': float(pop_df['qual'].mean()),
                'median': float(pop_df['qual'].median()),
                'std': float(pop_df['qual'].std()),
                'q25': float(pop_df['qual'].quantile(0.25)),
                'q75': float(pop_df['qual'].quantile(0.75)),
            },
            'qual_pass': {
                'mean': float(pass_df['qual'].mean()) if len(pass_df) else 0,
                'median': float(pass_df['qual'].median()) if len(pass_df) else 0,
            },
            'gq': {
                'mean': float(pop_df['gq'].mean()),
                'median': float(pop_df['gq'].median()),
            },
            'variant_types': pop_df['variant_type'].value_counts().to_dict(),
            'genotypes': pop_df['genotype'].value_counts().to_dict(),
            'variants_per_sample': float(len(pop_df) / max(pop_df['sample_id'].nunique(), 1)),
        }

    # Kruskal-Wallis test across superpopulations
    from scipy import stats as sp_stats
    groups = [
        df[df['super_population'] == pop]['qual'].values
        for pop in sorted(df['super_population'].unique())
    ]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) >= 2:
        kw_stat, kw_p = sp_stats.kruskal(*groups)
        stats['_kruskal_wallis'] = {
            'statistic': float(kw_stat),
            'p_value': float(kw_p),
            'n_groups': len(groups),
        }

    # Per-sample stats for within-population variance
    per_sample = []
    for _, row in df.groupby('sample_id').agg(
        qual_mean=('qual', 'mean'),
        qual_median=('qual', 'median'),
        n_variants=('qual', 'count'),
        n_pass=('filter', lambda x: (x == 'PASS').sum()),
        super_population=('super_population', 'first'),
    ).reset_index().iterrows():
        per_sample.append(row.to_dict())
    stats['_per_sample'] = per_sample

    return stats


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_quality_analysis(
    df: pd.DataFrame,
    stats: Dict,
    output_prefix: str,
    dpi: int = 150,
) -> List[str]:
    """Generate comprehensive quality analysis plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    POP_ORDER = ['AFR', 'AMR', 'EAS', 'EUR', 'SAS']
    POP_COLORS = {
        'AFR': '#E74C3C',
        'AMR': '#F39C12',
        'EAS': '#27AE60',
        'EUR': '#3498DB',
        'SAS': '#8E44AD',
    }
    available_pops = [p for p in POP_ORDER if p in df['super_population'].unique()]

    os.makedirs(os.path.dirname(output_prefix) or '.', exist_ok=True)
    saved_files = []

    # ── Figure 1: Main quality overview (2x2 grid) ────────────────────────────
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: QUAL distribution by superpopulation (violin + box)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_data = [df[df['super_population'] == p]['qual'].values for p in available_pops]
    parts = ax1.violinplot(
        plot_data,
        positions=range(len(available_pops)),
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(POP_COLORS[available_pops[i]])
        pc.set_alpha(0.4)

    bp = ax1.boxplot(
        plot_data,
        positions=range(len(available_pops)),
        widths=0.15,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='black', linewidth=2),
    )
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(POP_COLORS[available_pops[i]])
        patch.set_alpha(0.7)

    ax1.set_xticks(range(len(available_pops)))
    ax1.set_xticklabels(available_pops, fontsize=11, fontweight='bold')
    ax1.set_ylabel('QUAL Score', fontsize=12)
    ax1.set_title('Variant Quality Score Distribution', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Panel B: Variant count by type and superpopulation
    ax2 = fig.add_subplot(gs[0, 1])
    vtypes = ['SNP', 'INDEL', 'MNP']
    vtype_colors = ['#3498DB', '#9B59B6', '#E67E22']
    x = np.arange(len(available_pops))
    width = 0.25
    for vi, (vt, vc) in enumerate(zip(vtypes, vtype_colors)):
        counts = [
            len(df[(df['super_population'] == p) & (df['variant_type'] == vt)])
            for p in available_pops
        ]
        if sum(counts) > 0:
            bars = ax2.bar(x + vi * width - width, counts, width,
                           label=vt, color=vc, alpha=0.8, edgecolor='white')
            for bar, c in zip(bars, counts):
                if c > 0:
                    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                             str(c), ha='center', va='bottom', fontsize=7)

    ax2.set_xticks(x)
    ax2.set_xticklabels(available_pops, fontsize=11, fontweight='bold')
    ax2.set_ylabel('Variant Count', fontsize=12)
    ax2.set_title('Variant Type Distribution', fontsize=13, fontweight='bold')
    ax2.legend(frameon=True, fancybox=True)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Panel C: Mean QUAL per sample, colored by superpopulation
    ax3 = fig.add_subplot(gs[1, 0])
    sample_stats = df.groupby(['sample_id', 'super_population']).agg(
        mean_qual=('qual', 'mean'),
        n_variants=('qual', 'count'),
    ).reset_index().sort_values('super_population')

    for pop in available_pops:
        mask = sample_stats['super_population'] == pop
        pop_data = sample_stats[mask]
        ax3.scatter(
            pop_data['sample_id'], pop_data['mean_qual'],
            s=pop_data['n_variants'] * 2,
            c=POP_COLORS[pop], label=pop, alpha=0.8,
            edgecolors='white', linewidth=0.5, zorder=3,
        )

    ax3.set_ylabel('Mean QUAL Score', fontsize=12)
    ax3.set_title('Per-Sample Mean Quality (size = variant count)',
                  fontsize=13, fontweight='bold')
    ax3.legend(title='Superpopulation', frameon=True, fancybox=True,
               loc='upper right', fontsize=8)
    ax3.tick_params(axis='x', rotation=45, labelsize=8)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    # Panel D: Genotype quality (GQ) by superpopulation
    ax4 = fig.add_subplot(gs[1, 1])
    gq_data = [df[df['super_population'] == p]['gq'].values for p in available_pops]
    bp2 = ax4.boxplot(
        gq_data,
        positions=range(len(available_pops)),
        widths=0.4,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='black', linewidth=2),
    )
    for i, patch in enumerate(bp2['boxes']):
        patch.set_facecolor(POP_COLORS[available_pops[i]])
        patch.set_alpha(0.7)

    ax4.set_xticks(range(len(available_pops)))
    ax4.set_xticklabels(available_pops, fontsize=11, fontweight='bold')
    ax4.set_ylabel('Genotype Quality (GQ)', fontsize=12)
    ax4.set_title('Genotype Quality Distribution', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')

    # Suptitle with KW test result
    kw = stats.get('_kruskal_wallis', {})
    if kw:
        sig = 'significant' if kw['p_value'] < 0.05 else 'not significant'
        fig.suptitle(
            f'DeepVariant Quality Scores by Superpopulation\n'
            f'Kruskal-Wallis H={kw["statistic"]:.2f}, p={kw["p_value"]:.4f} ({sig})',
            fontsize=15, fontweight='bold', y=1.02,
        )
    else:
        fig.suptitle(
            'DeepVariant Quality Scores by Superpopulation',
            fontsize=15, fontweight='bold', y=1.02,
        )

    path1 = f'{output_prefix}_overview.png'
    fig.savefig(path1, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(path1)
    print(f'  Saved: {path1}')

    # ── Figure 2: QUAL by variant type × superpopulation ──────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

    for ax, vt in zip(axes2, ['SNP', 'INDEL']):
        vt_df = df[df['variant_type'] == vt]
        if len(vt_df) == 0:
            ax.text(0.5, 0.5, f'No {vt} variants', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{vt} Quality', fontsize=13, fontweight='bold')
            continue

        plot_data_vt = [
            vt_df[vt_df['super_population'] == p]['qual'].values
            for p in available_pops
        ]
        has_data = [len(d) > 0 for d in plot_data_vt]

        if any(has_data):
            bp_vt = ax.boxplot(
                [d if len(d) > 0 else [0] for d in plot_data_vt],
                positions=range(len(available_pops)),
                widths=0.5,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color='black', linewidth=2),
            )
            for i, patch in enumerate(bp_vt['boxes']):
                patch.set_facecolor(POP_COLORS[available_pops[i]])
                patch.set_alpha(0.7 if has_data[i] else 0.1)

        ax.set_xticks(range(len(available_pops)))
        ax.set_xticklabels(available_pops, fontsize=11, fontweight='bold')
        ax.set_ylabel('QUAL Score', fontsize=12)
        ax.set_title(f'{vt} Quality by Superpopulation',
                      fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        for i, p in enumerate(available_pops):
            n = len(vt_df[vt_df['super_population'] == p])
            ax.text(i, ax.get_ylim()[0], f'n={n}', ha='center', va='top',
                    fontsize=8, fontstyle='italic')

    fig2.suptitle('Quality Score by Variant Type and Superpopulation',
                  fontsize=14, fontweight='bold')
    fig2.tight_layout()

    path2 = f'{output_prefix}_by_variant_type.png'
    fig2.savefig(path2, dpi=dpi, bbox_inches='tight')
    plt.close(fig2)
    saved_files.append(path2)
    print(f'  Saved: {path2}')

    # ── Figure 3: PASS rate and variant count per sample ──────────────────────
    fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 6))

    sample_summary = df.groupby(['sample_id', 'super_population']).agg(
        n_total=('qual', 'count'),
        n_pass=('filter', lambda x: (x == 'PASS').sum()),
        mean_qual=('qual', 'mean'),
    ).reset_index()
    sample_summary['pass_rate'] = sample_summary['n_pass'] / sample_summary['n_total']
    sample_summary = sample_summary.sort_values(['super_population', 'sample_id'])

    # PASS rate per sample
    for pop in available_pops:
        mask = sample_summary['super_population'] == pop
        pop_data = sample_summary[mask]
        ax5.bar(
            pop_data['sample_id'], pop_data['pass_rate'],
            color=POP_COLORS[pop], alpha=0.8, label=pop,
            edgecolor='white', linewidth=0.5,
        )

    ax5.set_ylabel('PASS Rate', fontsize=12)
    ax5.set_title('Filter PASS Rate by Sample', fontsize=13, fontweight='bold')
    ax5.tick_params(axis='x', rotation=45, labelsize=8)
    ax5.set_ylim(0, 1.05)
    ax5.axhline(y=sample_summary['pass_rate'].mean(), color='gray',
                linestyle='--', alpha=0.5, label='mean')
    ax5.legend(title='Superpop', fontsize=8, loc='lower right')
    ax5.grid(axis='y', alpha=0.3, linestyle='--')

    # Variant count per sample
    for pop in available_pops:
        mask = sample_summary['super_population'] == pop
        pop_data = sample_summary[mask]
        ax6.bar(
            pop_data['sample_id'], pop_data['n_total'],
            color=POP_COLORS[pop], alpha=0.8, label=pop,
            edgecolor='white', linewidth=0.5,
        )

    ax6.set_ylabel('Variant Count', fontsize=12)
    ax6.set_title('Variant Count by Sample', fontsize=13, fontweight='bold')
    ax6.tick_params(axis='x', rotation=45, labelsize=8)
    ax6.legend(title='Superpop', fontsize=8, loc='upper right')
    ax6.grid(axis='y', alpha=0.3, linestyle='--')

    fig3.suptitle('Per-Sample Quality Metrics', fontsize=14, fontweight='bold')
    fig3.tight_layout()

    path3 = f'{output_prefix}_per_sample.png'
    fig3.savefig(path3, dpi=dpi, bbox_inches='tight')
    plt.close(fig3)
    saved_files.append(path3)
    print(f'  Saved: {path3}')

    # ── Figure 4: Summary statistics heatmap ──────────────────────────────────
    fig4, ax7 = plt.subplots(figsize=(10, 6))

    metric_names = [
        'Mean QUAL', 'Median QUAL', 'Mean GQ', 'PASS Rate',
        'Variants/Sample', 'SNP %', 'INDEL %', 'Het %',
    ]
    heatmap_data = []
    for pop in available_pops:
        s = stats.get(pop, {})
        n_total = s.get('n_variants_total', 1)
        vtypes = s.get('variant_types', {})
        gts = s.get('genotypes', {})
        row = [
            s.get('qual', {}).get('mean', 0),
            s.get('qual', {}).get('median', 0),
            s.get('gq', {}).get('mean', 0),
            s.get('pass_rate', 0) * 100,
            s.get('variants_per_sample', 0),
            vtypes.get('SNP', 0) / max(n_total, 1) * 100,
            vtypes.get('INDEL', 0) / max(n_total, 1) * 100,
            gts.get('het', 0) / max(n_total, 1) * 100,
        ]
        heatmap_data.append(row)

    heatmap_arr = np.array(heatmap_data)
    # Normalize each column to [0,1] for color scaling
    with np.errstate(divide='ignore', invalid='ignore'):
        col_min = heatmap_arr.min(axis=0)
        col_max = heatmap_arr.max(axis=0)
        col_range = col_max - col_min
        col_range[col_range == 0] = 1
        heatmap_norm = (heatmap_arr - col_min) / col_range

    im = ax7.imshow(heatmap_norm.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)

    ax7.set_xticks(range(len(available_pops)))
    ax7.set_xticklabels(available_pops, fontsize=12, fontweight='bold')
    ax7.set_yticks(range(len(metric_names)))
    ax7.set_yticklabels(metric_names, fontsize=10)

    for i in range(len(available_pops)):
        for j in range(len(metric_names)):
            val = heatmap_arr[i, j]
            fmt = f'{val:.1f}' if val >= 1 else f'{val:.2f}'
            color = 'white' if heatmap_norm.T[j, i] > 0.6 else 'black'
            ax7.text(i, j, fmt, ha='center', va='center',
                     fontsize=9, color=color, fontweight='bold')

    ax7.set_title('Quality Metrics Summary by Superpopulation',
                  fontsize=14, fontweight='bold', pad=15)
    fig4.colorbar(im, ax=ax7, label='Relative Scale (per metric)', shrink=0.8)
    fig4.tight_layout()

    path4 = f'{output_prefix}_heatmap.png'
    fig4.savefig(path4, dpi=dpi, bbox_inches='tight')
    plt.close(fig4)
    saved_files.append(path4)
    print(f'  Saved: {path4}')

    return saved_files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print('=' * 60)
    print('DeepVariant Superpopulation Quality Analysis')
    print('=' * 60)

    # Load mapping
    print(f'\nLoading population mapping: {args.mapping}')
    mapping_df = pd.read_csv(args.mapping)
    if 'sample_id' not in mapping_df.columns:
        if 'file_index' in mapping_df.columns:
            print('  NOTE: mapping has file_index but no sample_id column name, '
                  'attempting auto-detect...')
    print(f'  {len(mapping_df)} samples in mapping')
    print(f'  Superpopulations: {sorted(mapping_df["super_population"].unique())}')

    # Discover VCFs
    print(f'\nDiscovering VCFs in: {args.output_dir}')
    vcfs = discover_vcfs(args.output_dir)
    print(f'  Found {len(vcfs)} sample VCFs:')
    for sid, path in sorted(vcfs.items()):
        pop = mapping_df.loc[
            mapping_df['sample_id'] == sid, 'super_population'
        ].values
        pop_str = pop[0] if len(pop) > 0 else '???'
        print(f'    {sid} ({pop_str}): {os.path.basename(path)}')

    # Parse all VCFs
    print(f'\nParsing VCF files...')
    df = load_all_variants(vcfs, mapping_df, args.min_qual)

    print(f'\n  Total variants: {len(df)}')
    print(f'  Samples: {df["sample_id"].nunique()}')
    print(f'  Superpopulations: {sorted(df["super_population"].unique())}')
    print(f'  Variant types: {dict(df["variant_type"].value_counts())}')
    print(f'  Genotypes: {dict(df["genotype"].value_counts())}')

    # Compute statistics
    print(f'\n{"="*60}')
    print('Computing statistics...')
    print(f'{"="*60}')
    all_stats = compute_statistics(df)

    for pop in sorted(df['super_population'].unique()):
        s = all_stats[pop]
        print(f'\n  {pop} ({s["n_samples"]} samples, '
              f'{s["n_variants_total"]} variants):')
        print(f'    QUAL: mean={s["qual"]["mean"]:.1f}, '
              f'median={s["qual"]["median"]:.1f}, '
              f'std={s["qual"]["std"]:.1f}')
        print(f'    GQ:   mean={s["gq"]["mean"]:.1f}, '
              f'median={s["gq"]["median"]:.1f}')
        print(f'    PASS rate: {s["pass_rate"]:.1%}')
        print(f'    Types: {s["variant_types"]}')

    kw = all_stats.get('_kruskal_wallis', {})
    if kw:
        print(f'\n  Kruskal-Wallis test (QUAL across superpopulations):')
        print(f'    H = {kw["statistic"]:.4f}')
        print(f'    p = {kw["p_value"]:.6f}')
        if kw['p_value'] < 0.05:
            print(f'    --> Significant difference detected (p < 0.05)')
        else:
            print(f'    --> No significant difference (p >= 0.05)')

    # Generate plots
    print(f'\n{"="*60}')
    print('Generating plots...')
    print(f'{"="*60}')
    saved = plot_quality_analysis(df, all_stats, args.output, args.dpi)

    # Save raw data and stats
    csv_path = f'{args.output}_variants.csv'
    df.to_csv(csv_path, index=False)
    print(f'  Saved variant data: {csv_path}')

    stats_path = f'{args.output}_stats.json'
    serializable = {
        k: v for k, v in all_stats.items()
        if k != '_per_sample'
    }
    if '_per_sample' in all_stats:
        serializable['per_sample'] = all_stats['_per_sample']
    with open(stats_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f'  Saved statistics: {stats_path}')

    # Summary
    print(f'\n{"="*60}')
    print('Analysis Complete!')
    print(f'{"="*60}')
    print(f'  Plots:')
    for s in saved:
        print(f'    {s}')
    print(f'  Data: {csv_path}')
    print(f'  Stats: {stats_path}')
    print()


if __name__ == '__main__':
    main()
