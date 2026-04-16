#!/usr/bin/env python3
"""Plot UMAP embeddings colored by functional, regulatory, and repeat annotations.

Loads a pre-computed embedding NPZ (from umap_activations.py) and annotated
metadata CSV (from annotate_activations.py), then produces a 4-panel figure:
  Panel 1: KMeans cluster labels
  Panel 2: Gene model annotation (CDS/UTR/promoter/intron/intergenic)
  Panel 3: ENCODE cCRE regulatory class
  Panel 4: Repeat element class (SINE/LINE/SSR/DNA/etc.)

Also computes Adjusted Mutual Information (AMI) between cluster labels and
each annotation track.

Usage:
    # Auto-detect layer from embedding file
    /opt/homebrew/Cellar/deepvariant/1.9.0/libexec/venv/bin/python3 \
        plot_annotated_umap.py --layer mixed5

    # Explicit embedding path
    /opt/homebrew/Cellar/deepvariant/1.9.0/libexec/venv/bin/python3 \
        plot_annotated_umap.py \
        --embedding output/activations/umap_mixed5_embedding.npz \
        --output output/activations/umap_mixed5_annotated.png
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layer', default='mixed5',
                   help='Layer name — used to auto-locate embedding NPZ (default: mixed5).')
    p.add_argument('--embedding', default=None,
                   help='Explicit path to embedding NPZ (overrides --layer auto-detect).')
    p.add_argument('--meta', default=None,
                   help='Path to annotated_metadata.csv (default: output/activations/annotated_metadata.csv).')
    p.add_argument('--output', '-o', default=None,
                   help='Output PNG path (default: output/activations/umap_{layer}_annotated.png).')
    p.add_argument('--dpi', type=int, default=150)
    p.add_argument('--s', type=float, default=8.0, help='Scatter point size.')
    p.add_argument('--alpha', type=float, default=0.6, help='Scatter alpha.')
    return p.parse_args()


# ── Color palettes ─────────────────────────────────────────────────────────────
BACKGROUND_LABELS = {'none', 'unknown'}

FC_COLORS = {
    'CDS':        '#E69F00',
    '5UTR':       '#56B4E9',
    '3UTR':       '#009E73',
    'promoter':   '#F0E442',
    'pseudogene': '#8B4513',
    'exon_nc':    '#0072B2',
    'intron':     '#D55E00',
    'intergenic': '#CC79A7',
    'unknown':    '#999999',
}

ER_COLORS = {
    'PLS':              '#FF0000',
    'pELS':             '#FFA500',
    'dELS':             '#FFCD00',
    'CTCF-only':        '#00B0F0',
    'DNase-H3K4me3':    '#C0C0C0',
    'none':             '#EEEEEE',
    'unknown':          '#999999',
}

RC_COLORS = {
    'SINE':         '#1f77b4',
    'LINE':         '#ff7f0e',
    'SSR':          '#2ca02c',
    'DNA':          '#d62728',
    'LTR':          '#9467bd',
    'Satellite':    '#8c564b',
    'Low_complexity': '#e377c2',
    'other_repeat': '#bcbd22',
    'none':         '#EEEEEE',
}


def _scatter_categorical(ax, coords, labels, color_map, title, s=8, alpha=0.6):
    unique = sorted(set(labels), key=lambda x: list(color_map.keys()).index(x)
                    if x in color_map else 999)
    # Pass 1: background labels (none/unknown) at low alpha, no legend entry
    for label in unique:
        if label not in BACKGROUND_LABELS:
            continue
        mask = np.array([l == label for l in labels])
        c = color_map.get(label, '#cccccc')
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=c, s=s, alpha=0.15, edgecolors='none', rasterized=True)
    # Pass 2: named categories at full alpha with legend entries
    named = [l for l in unique if l not in BACKGROUND_LABELS]
    for label in named:
        mask = np.array([l == label for l in labels])
        c = color_map.get(label, '#999999')
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=c, s=s, alpha=alpha, edgecolors='none',
                   label=f'{label} ({mask.sum()})', rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('UMAP-1', fontsize=8)
    ax.set_ylabel('UMAP-2', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=7, loc='best', framealpha=0.6, markerscale=2,
              ncol=1 if len(named) <= 8 else 2)


def _scatter_cluster(ax, coords, cluster_labels, title, s=8, alpha=0.6):
    n_clusters = int(cluster_labels.max()) + 1
    cmap = plt.get_cmap('tab20' if n_clusters > 10 else 'tab10')
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=cluster_labels, cmap=cmap,
                    vmin=0, vmax=max(9, n_clusters - 1),
                    s=s, alpha=alpha, edgecolors='none', rasterized=True)
    plt.colorbar(sc, ax=ax, label='cluster', shrink=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('UMAP-1', fontsize=8)
    ax.set_ylabel('UMAP-2', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def main():
    args = _parse_args()

    # Resolve paths
    act_dir = os.path.join(REPO_DIR, 'output', 'activations')
    emb_path = args.embedding or os.path.join(act_dir, f'umap_{args.layer}_embedding.npz')
    meta_path = args.meta or os.path.join(act_dir, 'annotated_metadata.csv')
    out_path = args.output or os.path.join(act_dir, f'umap_{args.layer}_annotated.png')

    if not os.path.exists(emb_path):
        print(f'ERROR: embedding not found: {emb_path}')
        print(f'  Run: python deepvariantinterp/plotting/umap_activations.py --layer {args.layer}')
        sys.exit(1)
    if not os.path.exists(meta_path):
        print(f'ERROR: metadata not found: {meta_path}')
        print('  Run: python annotate_activations.py')
        sys.exit(1)

    # Load embedding
    emb_data = np.load(emb_path, allow_pickle=True)
    coords = emb_data['embedding']          # (N, 2)
    cluster_labels = emb_data['cluster_labels'] if 'cluster_labels' in emb_data.files else None
    layer_name = str(emb_data['layer']) if 'layer' in emb_data.files else args.layer
    print(f'Loaded embedding: {coords.shape} for layer={layer_name}')

    # Load metadata and join on global_indices (handles subsampling correctly)
    df_full = pd.read_csv(meta_path)
    print(f'Loaded metadata: {len(df_full)} variants')

    if 'global_indices' in emb_data.files:
        global_indices = emb_data['global_indices']   # (N,) int array
        df = df_full.iloc[global_indices].reset_index(drop=True)
        print(f'Joined {len(df)} embedding rows to metadata via global_indices')
    else:
        # Legacy fallback: positional alignment (only correct when no subsampling)
        n = min(len(coords), len(df_full))
        if n < len(coords) or n < len(df_full):
            print(f'WARNING: positional align to {n} rows (embedding={len(coords)}, meta={len(df_full)})')
        df = df_full.iloc[:n].reset_index(drop=True)
        if cluster_labels is not None:
            cluster_labels = cluster_labels[:n]
        coords = coords[:n]

    n = len(df)
    fc_labels = df['functional_class'].fillna('unknown').tolist()
    er_labels = df['encode_regulatory'].fillna('none').tolist()
    rc_labels = df['repeat_class'].fillna('none').tolist() if 'repeat_class' in df.columns else ['none'] * n

    # Build figure
    has_clusters = cluster_labels is not None
    n_panels = 4 if has_clusters else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5))
    fig.suptitle(f'DeepVariant InceptionV3 — {layer_name} UMAP (n={n})', fontsize=12)

    col = 0
    if has_clusters:
        _scatter_cluster(axes[col], coords, cluster_labels,
                         f'KMeans clusters (k={int(cluster_labels.max())+1})',
                         s=args.s, alpha=args.alpha)
        col += 1

    _scatter_categorical(axes[col], coords, fc_labels, FC_COLORS,
                         'Gene model annotation', s=args.s, alpha=args.alpha)
    _scatter_categorical(axes[col+1], coords, er_labels, ER_COLORS,
                         'ENCODE cCRE regulatory', s=args.s, alpha=args.alpha)
    _scatter_categorical(axes[col+2], coords, rc_labels, RC_COLORS,
                         'Repeat element class', s=args.s, alpha=args.alpha)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')

    # AMI scores
    if has_clusters:
        from sklearn.metrics import adjusted_mutual_info_score
        ami_fc = adjusted_mutual_info_score(fc_labels, cluster_labels.tolist())
        ami_er = adjusted_mutual_info_score(er_labels, cluster_labels.tolist())
        ami_rc = adjusted_mutual_info_score(rc_labels, cluster_labels.tolist())
        print(f'\nAdjusted Mutual Information — {layer_name}:')
        print(f'  clusters vs functional_class:  {ami_fc:.4f}')
        print(f'  clusters vs encode_regulatory: {ami_er:.4f}')
        print(f'  clusters vs repeat_class:      {ami_rc:.4f}')


if __name__ == '__main__':
    main()
