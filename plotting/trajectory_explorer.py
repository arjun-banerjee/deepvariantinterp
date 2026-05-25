#!/usr/bin/env python3
"""Trajectory Explorer — interactive Dash app for miscall trajectory analysis.

Shows a 1×N layer strip (mixed0–10 + 3 max-pool layers inserted at the correct
architectural positions). Every column thumbnail shows all 5 miscalls at once.
Click any column to expand it in the full detail panel below.

Usage:
    python plotting/trajectory_explorer.py
    # open http://127.0.0.1:8051
"""
from __future__ import annotations

import io
import base64
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import (
    Dash, Input, Output, State, callback, clientside_callback,
    dcc, html, ctx, no_update, ALL,
)

from analysis_report import generate_report as _generate_report

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO = Path(__file__).resolve().parent.parent
INCEPTION_ALL_DIR = REPO / 'hg002_chr20_inception_analysis' / 'all_variant_umap'
MAXPOOL_ALL_DIR = (
    REPO / 'hg002_chr20_maxpool_encode_analysis' / 'analysis' / 'all_variant_umap'
)
# Reference CSV used to build the consistent keep_eids subsample
_REF_CSV = INCEPTION_ALL_DIR / 'umap_coords_mixed5_all.csv'
_SAMPLE_TARGET = 45_000  # approximate total variants to show (always includes all FP/FN)
_SAMPLE_SEED   = 42

# ── Layer sequence: max-pool layers inserted at architectural positions ────────
#
#   max_pooling2d_1 sits between mixed2 and mixed3
#   max_pooling2d_2 sits between mixed4 and mixed5
#   max_pooling2d_3 sits between mixed7 and mixed8
#   (max_pooling2d_0 is before mixed0 — not shown; mixed0 already captures post-MP0)

_ORDERED_COLS: list[tuple[str, str]] = [
    ('mixed0',           'inception'),
    ('mixed1',           'inception'),
    ('mixed2',           'inception'),
    ('max_pooling2d_1',  'maxpool'),
    ('mixed3',           'inception'),
    ('mixed4',           'inception'),
    ('max_pooling2d_2',  'maxpool'),
    ('mixed5',           'inception'),
    ('mixed6',           'inception'),
    ('mixed7',           'inception'),
    ('max_pooling2d_3',  'maxpool'),
    ('mixed8',           'inception'),
    ('mixed9',           'inception'),
    ('mixed10',          'inception'),
]

# ── Per-variant info lookup (populated by load_data) ─────────────────────────

_eid_info: dict[int, dict] = {}  # eid → {status, chrom, start, ref, alt, pred_gt, truth_gt}

# ── Colors — exact palette from umap_cluster_explorer.py ─────────────────────

STATUS_COLORS = {
    'TP':      '#22c55e',
    'FP':      '#ef4444',
    'FN':      '#f97316',
    'RefCall': '#3b82f6',
}
GT_COLORS = {
    '0/0': '#3b82f6',
    '0/1': '#f59e0b',
    '1/1': '#22c55e',
    '1/2': '#c084fc',
    './.': '#6b7280',
}
MIXED_COLOR = '#94a3b8'   # slate for "Mixed" flip assignment
MP_COLOR    = '#a78bfa'   # violet for max-pool column labels / borders
FLIP_COLOR  = '#f59e0b'   # amber for flip transitions

_TRACK_COLORS = [
    '#facc15', '#c084fc', '#fb923c', '#34d399', '#60a5fa',
    '#f472b6', '#a3e635', '#38bdf8', '#fbbf24', '#e879f9',
]

_ASSIGN_COLORS: dict[str, str] = {
    'TP':      '#22c55e',
    'FP':      '#ef4444',
    'FN':      '#f97316',
    'RefCall': '#3b82f6',
    'Mixed':   '#94a3b8',
}

VARIANTS_VCF = (
    REPO / 'hg002_chr20_maxpool_encode_analysis' / 'analysis' / 'output.vcf.gz'
)

S = dict  # short alias for style dicts

DARK_BG  = '#0d1117'
SURFACE  = '#161b22'
BORDER   = '#30363d'
TEXT_DIM = '#8b949e'
TEXT     = '#e6edf3'
ACCENT   = '#58a6ff'


# ── Data loading ──────────────────────────────────────────────────────────────

def _short(layer: str) -> str:
    if layer.startswith('max_pooling2d_'):
        return 'MP' + layer[-1]
    return 'M' + layer.replace('mixed', '')


def _load_vcf_scores() -> dict:
    """Parse VCF for GQ, DP, VAF per variant. Returns (chrom, pos0, ref, alt) → dict."""
    import gzip
    scores: dict = {}
    if not VARIANTS_VCF.exists():
        print(f'  VCF not found at {VARIANTS_VCF} — skipping quality scores')
        return scores
    with gzip.open(str(VARIANTS_VCF), 'rt') as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 10:
                continue
            chrom, pos_s, ref, alt = parts[0], parts[1], parts[3], parts[4]
            fmt_keys = parts[8].split(':')
            fmt_vals = parts[9].split(':')
            fdict = dict(zip(fmt_keys, fmt_vals))
            try:
                gq = int(fdict.get('GQ', 0))
            except (ValueError, TypeError):
                gq = 0
            try:
                dp = int(fdict.get('DP', 0))
            except (ValueError, TypeError):
                dp = 0
            try:
                vaf_s = fdict.get('VAF', '0').split(',')[0]
                vaf = float(vaf_s) if vaf_s not in ('.', '') else 0.0
            except (ValueError, TypeError):
                vaf = 0.0
            key = (chrom, int(pos_s) - 1, ref, alt)  # convert to 0-based
            scores[key] = {'GQ': gq, 'DP': dp, 'VAF': vaf}
    print(f'  VCF scores: {len(scores)} entries')
    return scores


def compute_flip_assignments(
    layers_data: dict,
    columns: list[tuple[str, str]],
    tracked_eids: list[int],
    k: int = 10,
    majority_thresh: float = 0.7,
) -> dict[int, list[str | None]]:
    """For each tracked eid, return its nearest-k cluster assignment at every layer.

    Assignment = majority status label if that class has ≥ majority_thresh of the k
    nearest neighbors, else 'Mixed'.  None = variant not in this layer.
    """
    result: dict[int, list] = {}
    for eid in tracked_eids:
        assignments: list = []
        for layer, _ in columns:
            ld = layers_data.get(layer)
            if ld is None or eid not in ld.get('eid_idx', {}):
                assignments.append(None)
                continue
            row_i = ld['eid_idx'][eid]
            bg = ld['bg']
            vx = float(bg['x'].iat[row_i])
            vy = float(bg['y'].iat[row_i])
            bx = bg['x'].values
            by = bg['y'].values
            bs = bg['status'].values
            sq = (bx - vx) ** 2 + (by - vy) ** 2
            kk = min(k + 1, len(sq))
            idx = np.argpartition(sq, kk - 1)[:kk]
            idx = idx[np.argsort(sq[idx])]
            idx = [i for i in idx if i != row_i][:k]
            counts = Counter(bs[idx].tolist())
            total = len(idx)
            if total == 0:
                assignments.append(None)
                continue
            majority = max(counts, key=counts.get)
            if counts[majority] / total >= majority_thresh:
                assignments.append(majority)
            else:
                assignments.append('Mixed')
        result[eid] = assignments
    return result


def _build_keep_eids() -> set[int]:
    """Determine which example_idx values to keep across all layers.

    Always includes every FP/FN variant; randomly samples from the rest to
    reach approximately _SAMPLE_TARGET total.
    """
    df = pd.read_csv(_REF_CSV, usecols=['example_idx', 'call_status'])
    miscall_mask = df['call_status'].isin(['FP', 'FN'])
    keep = set(df.loc[miscall_mask, 'example_idx'].tolist())
    bg_eids = df.loc[~miscall_mask, 'example_idx'].values
    n_bg = max(0, _SAMPLE_TARGET - len(keep))
    rng = np.random.default_rng(_SAMPLE_SEED)
    sampled = rng.choice(bg_eids, size=min(n_bg, len(bg_eids)), replace=False)
    keep.update(sampled.tolist())
    print(f'  keep_eids: {len(keep)} total ({len(df[miscall_mask])} miscalls + {len(sampled)} bg sample)')
    return keep


def load_data() -> tuple[dict, list[tuple[str, str]]]:
    """Return (layers_data, available_columns).

    layers_data[layer] = {
        'bg':      DataFrame(eid, x, y, status),
        'eid_idx': dict(eid → row_position_in_bg),
    }
    Also populates module-level _eid_info: eid → metadata dict.
    """
    global _eid_info
    keep_eids = _build_keep_eids()

    # Build per-variant metadata lookup from the reference CSV (all 111k rows)
    ref = pd.read_csv(_REF_CSV)
    _eid_info = (
        ref.drop_duplicates('example_idx')
        .set_index('example_idx')
        [['call_status', 'chrom_hg38', 'start_hg38', 'ref', 'alt', 'pred_gt', 'truth_gt']]
        .rename(columns={'call_status': 'status', 'chrom_hg38': 'chrom', 'start_hg38': 'start'})
        .to_dict('index')
    )
    print(f'  eid_info: {len(_eid_info)} entries')

    vcf_scores = _load_vcf_scores()

    # Build flat score arrays indexed by eid for fast vectorized join
    _eid_gq  = {eid: info.get('GQ',  np.nan) for eid, info in _eid_info.items()}
    _eid_dp  = {eid: info.get('DP',  np.nan) for eid, info in _eid_info.items()}
    _eid_vaf = {eid: info.get('VAF', np.nan) for eid, info in _eid_info.items()}
    _eid_pgt = {eid: info.get('pred_gt', '') for eid, info in _eid_info.items()}

    # Augment _eid_info with VCF quality scores
    for eid, info in _eid_info.items():
        key = (info.get('chrom'), info.get('start'), info.get('ref'), info.get('alt'))
        if key in vcf_scores:
            info.update(vcf_scores[key])
            _eid_gq[eid]  = vcf_scores[key]['GQ']
            _eid_dp[eid]  = vcf_scores[key]['DP']
            _eid_vaf[eid] = vcf_scores[key]['VAF']

    layers_data: dict = {}
    _COLS_NEEDED = ['example_idx', 'UMAP1', 'UMAP2', 'call_status', 'pred_gt']

    def _load_layer_csv(csv_path: Path) -> dict | None:
        if not csv_path.exists():
            print(f'  SKIP: no CSV at {csv_path}')
            return None
        df = pd.read_csv(csv_path, usecols=_COLS_NEEDED)
        df = df[df['example_idx'].isin(keep_eids)].reset_index(drop=True)
        bg = df.rename(columns={
            'example_idx': 'eid', 'UMAP1': 'x', 'UMAP2': 'y',
            'call_status': 'status',
        })[['eid', 'x', 'y', 'status', 'pred_gt']]
        bg = bg.copy()
        bg['GQ']  = bg['eid'].map(_eid_gq).astype(float)
        bg['DP']  = bg['eid'].map(_eid_dp).astype(float)
        bg['VAF'] = bg['eid'].map(_eid_vaf).astype(float)
        eid_idx = {int(eid): i for i, eid in enumerate(bg['eid'])}
        return {'bg': bg, 'eid_idx': eid_idx}

    # Inception layers — all-variant CSV
    for layer in [l for l, t in _ORDERED_COLS if t == 'inception']:
        result = _load_layer_csv(INCEPTION_ALL_DIR / f'umap_coords_{layer}_all.csv')
        if result:
            layers_data[layer] = result
            print(f'  {layer}: {len(result["bg"])} variants')

    # Max-pool layers — all-variant CSV
    for layer in [l for l, t in _ORDERED_COLS if t == 'maxpool']:
        result = _load_layer_csv(MAXPOOL_ALL_DIR / f'umap_coords_{layer}_all.csv')
        if result:
            layers_data[layer] = result
            print(f'  {layer}: {len(result["bg"])} variants')

    columns = [(l, t) for l, t in _ORDERED_COLS if l in layers_data]
    return layers_data, columns



# ── Thumbnail generation ──────────────────────────────────────────────────────

def make_thumbnail(layer_data: dict, layer_type: str) -> str:
    """Static matplotlib PNG of background variant density."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    bg = layer_data['bg']

    fig, ax = plt.subplots(figsize=(2.2, 1.9))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    bc = MP_COLOR if layer_type == 'maxpool' else BORDER
    for sp in ax.spines.values():
        sp.set_edgecolor(bc)
        sp.set_linewidth(1.2 if layer_type == 'maxpool' else 0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    # Background (sampled)
    bg_s = bg.sample(min(3000, len(bg)), random_state=0)
    for status, color in STATUS_COLORS.items():
        mask = bg_s['status'] == status
        ax.scatter(bg_s.loc[mask, 'x'], bg_s.loc[mask, 'y'],
                   s=0.4, c=color, alpha=0.22, linewidths=0, rasterized=True)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight',
                facecolor=DARK_BG, pad_inches=0.03)
    plt.close(fig)
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()


def make_variant_thumbnail(
    layer_data: dict,
    layer_type: str,
    variant_eid: int,
    track_color: str = '#facc15',
) -> str:
    """Like make_thumbnail but overlays the variant's UMAP position as a star."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    bg = layer_data['bg']
    eid_idx = layer_data.get('eid_idx', {})

    fig, ax = plt.subplots(figsize=(2.0, 1.7))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    bc = MP_COLOR if layer_type == 'maxpool' else BORDER
    for sp in ax.spines.values():
        sp.set_edgecolor(bc)
        sp.set_linewidth(1.2 if layer_type == 'maxpool' else 0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    bg_s = bg.sample(min(1500, len(bg)), random_state=0)
    for status, color in STATUS_COLORS.items():
        mask = bg_s['status'] == status
        ax.scatter(bg_s.loc[mask, 'x'], bg_s.loc[mask, 'y'],
                   s=0.5, c=color, alpha=0.22, linewidths=0, rasterized=True)

    if variant_eid in eid_idx:
        row_i = eid_idx[variant_eid]
        vx = float(bg['x'].iat[row_i])
        vy = float(bg['y'].iat[row_i])
        ax.scatter([vx], [vy], s=130, c=track_color, marker='*',
                   linewidths=0.8, edgecolors='white', zorder=10)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=88, bbox_inches='tight',
                facecolor=DARK_BG, pad_inches=0.03)
    plt.close(fig)
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()


def _label_row(label: str, color: str, description: str) -> html.Div:
    """One row in the labeling reference legend."""
    return html.Div([
        html.Span(label, style=S(
            color=color, fontWeight='700', fontSize='11px',
            background=color + '22', borderRadius='3px',
            padding='1px 6px', minWidth='90px', display='inline-block',
            flexShrink='0',
        )),
        html.Span(description, style=S(
            color=TEXT_DIM, fontSize='11px', lineHeight='1.5',
        )),
    ], style=S(display='flex', alignItems='flex-start', gap='12px'))


def build_bottom_section() -> html.Div:
    """Bottom section: tracked-variants table (dynamic) + annotation reference image."""
    th_style = S(
        padding='6px 12px', textAlign='left',
        fontSize='10px', textTransform='uppercase',
        letterSpacing='0.06em', color=TEXT_DIM,
        borderBottom=f'1px solid {BORDER}',
        fontWeight='600',
    )
    return html.Div([
        html.Div('Tracked Variants', style=S(
            fontSize='13px', fontWeight='700', color=ACCENT,
            marginBottom='10px', paddingTop='4px',
        )),
        html.Div(
            'Click any point in the detail map above to track it here.',
            style=S(fontSize='11px', color=TEXT_DIM, marginBottom='12px'),
        ),

        # ── Flip trajectory (nearest-10 neighbor assignment × layer) ──────
        html.Div('Layer Trajectory  (nearest-10 cluster assignment)', style=S(
            fontSize='12px', fontWeight='600', color=ACCENT,
            marginTop='8px', marginBottom='8px',
        )),
        html.Div(id='flip-trajectory', style=S(overflowX='auto', marginBottom='24px')),

        # ── Tracked variants metadata table ───────────────────────────────
        html.Div('Variant Metadata', style=S(
            fontSize='12px', fontWeight='600', color=ACCENT,
            marginBottom='8px',
        )),
        html.Table([
            html.Thead(html.Tr([
                html.Th('#',         style=th_style),
                html.Th('eid',       style=th_style),
                html.Th('Status',    style=th_style),
                html.Th('Chrom',     style=th_style),
                html.Th('Position',  style=th_style),
                html.Th('Ref→Alt',   style=th_style),
                html.Th('Pred GT',   style=th_style),
                html.Th('Truth GT',  style=th_style),
                html.Th('GQ',        style=th_style),
                html.Th('DP',        style=th_style),
                html.Th('VAF',       style=th_style),
            ])),
            html.Tbody(id='tracked-table-body'),
        ], style=S(
            width='100%', borderCollapse='collapse',
            background=SURFACE, borderRadius='4px',
            border=f'1px solid {BORDER}',
        )),

        html.Div([
            html.Div('Labeling reference', style=S(
                fontSize='12px', fontWeight='600', color=ACCENT,
                marginBottom='10px', marginTop='32px',
            )),
            html.Div([
                _label_row('TP',      STATUS_COLORS['TP'],
                    'True positive — variant correctly called. k-NN neighborhood is ≥70% TP.'),
                _label_row('FP',      STATUS_COLORS['FP'],
                    'False positive — variant called but absent in truth. Neighborhood ≥70% FP.'),
                _label_row('FN',      STATUS_COLORS['FN'],
                    'False negative — variant missed by caller. Neighborhood ≥70% FN.'),
                _label_row('RefCall', STATUS_COLORS['RefCall'],
                    'Reference call — site called as homozygous reference. Neighborhood ≥70% RefCall.'),
                _label_row('Mixed',   MIXED_COLOR,
                    'No single class dominates the k-NN neighborhood (all classes < 70%). '
                    'Variant sits at a cluster boundary — network representation is ambiguous at this layer.'),
                html.Div(style=S(height='10px')),
                _label_row('yellow separator', '#f59e0b',
                    'Flip — k-NN assignment changed between two consecutive mixed (inception) layers. '
                    'Max-pool transitions are excluded from flip detection.'),
                _label_row('red separator',    '#ef4444',
                    'Final incorrect flip — the last assignment change for a miscall (FP or FN). '
                    'This is the transition where the network commits to the wrong answer. '
                    'If the red line is not at the last mixed layer, it also implies the network '
                    'had remaining layers in which it could have corrected itself but never flipped back — '
                    'the wrong assignment was stable from that point onward.'),
            ], style=S(
                display='flex', flexDirection='column', gap='6px',
                maxWidth='700px',
            )),
        ]),

        html.Hr(style=S(border='none', borderTop=f'1px solid {BORDER}',
                        margin='36px 0 24px')),
        html.A(
            '↗ View flip analysis report',
            href='/assets/flip_analysis_report.html',
            target='_blank',
            style=S(
                color=ACCENT, fontSize='13px', fontWeight='600',
                textDecoration='none',
            ),
        ),
        html.Div(
            'Flip count · commitment layer · GQ correlation · recovery rate · stabilization by layer',
            style=S(fontSize='11px', color=TEXT_DIM, marginTop='4px'),
        ),
    ], style=S(padding='20px 20px 48px'))


def build_variant_trajectory_strip(
    variant_eid: int,
    layers_data: dict,
    columns: list[tuple[str, str]],
    track_color: str = '#facc15',
    show_maxpool: bool = False,
) -> html.Div:
    """Horizontal row: all UMAP layers with variant marked + flip separators.

    Flips are only detected between consecutive mixed (inception) layers.
    Separator colors:
      yellow = k-NN assignment changed between two mixed layers
      red    = the final such change for miscalls (FP/FN)
    """
    info = _eid_info.get(variant_eid, {})
    status = info.get('status', '?')
    is_miscall = status in ('FP', 'FN')

    visible_cols = [
        (l, t) for l, t in columns
        if t == 'inception' or show_maxpool
    ]

    assns: list = compute_flip_assignments(
        layers_data, visible_cols, [variant_eid]
    ).get(variant_eid, [None] * len(visible_cols))

    # Flips only count between consecutive mixed-to-mixed transitions
    flip_at: set[int] = set()
    last_flip_ci: int | None = None
    prev: str | None = None
    prev_type: str | None = None
    for ci, (layer, layer_type) in enumerate(visible_cols):
        a = assns[ci]
        if a is not None:
            if (prev is not None and a != prev
                    and layer_type == 'inception'
                    and prev_type == 'inception'):
                flip_at.add(ci)
                last_flip_ci = ci
            prev = a
            prev_type = layer_type

    cells: list = []
    for ci, (layer, layer_type) in enumerate(visible_cols):
        # Vertical separator before this column
        if ci > 0:
            if is_miscall and ci == last_flip_ci:
                sep_color, sep_w = '#ef4444', '4px'  # red: final miscall flip
            elif ci in flip_at:
                sep_color, sep_w = '#f59e0b', '3px'  # yellow: mixed-to-mixed flip
            else:
                sep_color, sep_w = BORDER, '1px'
            cells.append(html.Div(style=S(
                width=sep_w, flexShrink='0',
                backgroundColor=sep_color, alignSelf='stretch',
                margin='0 1px',
            )))

        ld = layers_data.get(layer)
        thumb_src = (
            make_variant_thumbnail(ld, layer_type, variant_eid, track_color)
            if ld else ''
        )
        assignment = assns[ci] or '–'
        a_color = _ASSIGN_COLORS.get(assignment, TEXT_DIM)
        is_mp = layer_type == 'maxpool'

        cells.append(html.Div([
            html.Div(_short(layer), style=S(
                textAlign='center', fontSize='9px',
                fontWeight='700' if is_mp else '400',
                color=MP_COLOR if is_mp else TEXT_DIM,
                padding='2px 0',
            )),
            html.Img(src=thumb_src, style=S(width='100%', display='block')),
            html.Div(
                html.Span(assignment, style=S(
                    fontSize='8px', fontWeight='600',
                    color=a_color, background=a_color + '28',
                    borderRadius='2px', padding='1px 4px',
                )),
                style=S(textAlign='center', marginTop='2px'),
            ),
        ], style=S(flex='1', minWidth='0')))

    chrom = info.get('chrom', '?')
    start = info.get('start', '?')
    loc = f'{chrom}:{start:,}' if isinstance(start, (int, float)) else f'eid:{variant_eid}'
    status_color = STATUS_COLORS.get(status, TEXT)

    return html.Div([
        html.Div([
            html.Span(f'eid {variant_eid}',
                      style=S(color=track_color, fontWeight='700', fontSize='11px')),
            html.Span(f'  {loc}', style=S(color=TEXT_DIM, fontSize='10px')),
            html.Span(f'  {status}',
                      style=S(color=status_color, fontSize='10px', fontWeight='600')),
            html.Span('  ·  yellow = flip  ·  red = final miscall flip',
                      style=S(color=TEXT_DIM, fontSize='9px', marginLeft='8px')),
        ], style=S(marginBottom='5px', paddingRight='28px')),
        html.Div(cells, style=S(
            display='flex', alignItems='stretch', overflowX='hidden',
        )),
    ], style=S(
        padding='8px 16px 10px',
        backgroundColor=SURFACE,
        borderBottom=f'1px solid {BORDER}',
        flexShrink='0',
    ))


# ── Detail figure ─────────────────────────────────────────────────────────────

def build_detail_figure(
    layers_data: dict,
    col_idx: int,
    columns: list[tuple[str, str]],
    vis_statuses: list[str],
    tracked_eids: list[int],
    pt_size: float = 4,
    color_by: str = 'status',
) -> go.Figure:
    layer, layer_type = columns[col_idx]
    ld  = layers_data[layer]
    bg  = ld['bg']
    eid_idx = ld.get('eid_idx', {})
    disp = layer.replace('max_pooling2d_', 'MaxPool ').replace('mixed', 'Mixed ')
    title_color = MP_COLOR if layer_type == 'maxpool' else ACCENT

    fig = go.Figure()
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=TEXT, size=12,
                  family='-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif'),
        autosize=True,
        margin=dict(l=52, r=20, t=60, b=44),
        legend=dict(
            bgcolor=SURFACE, bordercolor=BORDER, borderwidth=1,
            font=dict(size=12),
            itemclick='toggle', itemdoubleclick='toggleothers',
            tracegroupgap=8,
        ),
        title=dict(
            text=f'<b>{disp}</b>',
            font=dict(size=16, color=title_color),
            x=0.01, xanchor='left',
        ),
        xaxis=dict(
            gridcolor=BORDER, zerolinecolor=BORDER, zeroline=False,
            title=dict(text='UMAP 1', font=dict(color=TEXT_DIM)),
            tickfont=dict(color=TEXT_DIM),
        ),
        yaxis=dict(
            gridcolor=BORDER, zerolinecolor=BORDER, zeroline=False,
            title=dict(text='UMAP 2', font=dict(color=TEXT_DIM)),
            tickfont=dict(color=TEXT_DIM),
            scaleanchor='x', scaleratio=1,
        ),
        hovermode='closest',
        clickmode='event',
    )

    # ── Background traces ───────────────────────────────────────────────────
    _CONTINUOUS = {'GQ', 'DP', 'VAF'}

    if color_by in _CONTINUOUS:
        vals = bg[color_by].fillna(0).tolist()
        cbar_title = {'GQ': 'Genotype Quality', 'DP': 'Read Depth', 'VAF': 'Variant AF'}[color_by]
        fig.add_trace(go.Scatter(
            x=bg['x'].tolist(),
            y=bg['y'].tolist(),
            customdata=bg['eid'].tolist(),
            mode='markers',
            marker=dict(
                size=pt_size, opacity=0.7,
                color=vals,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title=dict(text=cbar_title, font=dict(color=TEXT_DIM, size=11)),
                    tickfont=dict(color=TEXT_DIM), thickness=14, len=0.65,
                    bgcolor=DARK_BG, bordercolor=BORDER,
                ),
            ),
            name=color_by,
            hovertemplate=f'eid %{{customdata}}<br>{color_by}: %{{marker.color:.1f}}<extra></extra>',
        ))
    elif color_by == 'pred_gt':
        for i, (gt, color) in enumerate(GT_COLORS.items()):
            mask = bg['pred_gt'] == gt
            pts  = bg[mask]
            if pts.empty:
                continue
            fig.add_trace(go.Scatter(
                x=pts['x'].tolist(), y=pts['y'].tolist(),
                customdata=pts['eid'].tolist(),
                mode='markers',
                marker=dict(size=pt_size, color=color, opacity=0.5),
                name=gt,
                legendgroup=f'gt_{gt}',
                legendgrouptitle=dict(
                    text='Pred Genotype',
                    font=dict(color=TEXT_DIM, size=10),
                ) if i == 0 else dict(text=''),
                hovertemplate='eid %{customdata}<br>GT: ' + gt + '<extra></extra>',
            ))
    else:  # status (default)
        for i, (status, color) in enumerate(STATUS_COLORS.items()):
            mask = bg['status'] == status
            pts  = bg[mask]
            if pts.empty:
                continue
            fig.add_trace(go.Scatter(
                x=pts['x'].tolist(),
                y=pts['y'].tolist(),
                customdata=pts['eid'].tolist(),
                mode='markers',
                marker=dict(size=pt_size, color=color, opacity=0.5),
                name=status,
                legendgroup=f'bg_{status}',
                legendgrouptitle=dict(
                    text='Background',
                    font=dict(color=TEXT_DIM, size=10),
                ) if i == 0 else dict(text=''),
                visible=True if status in vis_statuses else 'legendonly',
                hovertemplate='eid %{customdata}<br>%{x:.3f}, %{y:.3f}<extra>' + status + '</extra>',
            ))

    # ── Tracked variant overlays ────────────────────────────────────────────
    for vi, eid in enumerate(tracked_eids):
        if eid not in eid_idx:
            continue
        row_i  = eid_idx[eid]
        vx     = float(bg['x'].iat[row_i])
        vy     = float(bg['y'].iat[row_i])
        info   = _eid_info.get(eid, {})
        status = info.get('status', '?')
        chrom  = info.get('chrom', '?')
        start  = info.get('start', '?')
        ref    = info.get('ref', '?')
        alt    = info.get('alt', '?')
        pred   = info.get('pred_gt', '?')
        truth  = info.get('truth_gt', '?')
        tcolor = _TRACK_COLORS[vi % len(_TRACK_COLORS)]
        label  = f'#{vi + 1}'

        fig.add_trace(go.Scatter(
            x=[vx], y=[vy], mode='markers',
            marker=dict(symbol='circle', size=28, color=tcolor, opacity=0.18,
                        line=dict(width=0)),
            showlegend=False, hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            x=[vx], y=[vy],
            mode='markers+text',
            marker=dict(symbol='star', size=14, color=tcolor,
                        line=dict(width=1, color='white')),
            text=[label],
            textposition='top right',
            textfont=dict(color=tcolor, size=11),
            name=f'{label} {status} eid:{eid}',
            legendgroup=f'tr_{eid}',
            legendgrouptitle=dict(
                text='Tracked',
                font=dict(color=TEXT_DIM, size=10),
            ) if vi == 0 else dict(text=''),
            hovertemplate=(
                f'<b>{label}</b> eid:{eid}<br>'
                f'{chrom}:{start}<br>'
                f'{ref}→{alt}  {status}<br>'
                f'pred {pred} / truth {truth}'
                '<extra></extra>'
            ),
        ))

    return fig


# ── App ───────────────────────────────────────────────────────────────────────

def build_app(
    layers_data: dict,
    columns: list[tuple[str, str]],
) -> Dash:
    print('Generating thumbnails…')
    thumbs: list[str] = []
    for ci, (layer, layer_type) in enumerate(columns):
        thumbs.append(make_thumbnail(layers_data[layer], layer_type))
        print(f'  {ci + 1}/{len(columns)}  {layer}')

    app = Dash(__name__, title='Trajectory Explorer',
               suppress_callback_exceptions=True,
               assets_folder=str(Path(__file__).parent / 'assets'))


    lbl_hdr = S(
        fontSize='9px', textTransform='uppercase', letterSpacing='0.08em',
        color=TEXT_DIM, marginBottom='8px', marginTop='16px', fontWeight='600',
    )

    # ── Sidebar ────────────────────────────────────────────────────────────
    sidebar = html.Div([
        html.Div('Background', style=lbl_hdr),
        dcc.Checklist(
            id='status-checklist',
            options=[{
                'label': html.Span([
                    html.Span('■ ', style=S(color=c, fontSize='13px')),
                    html.Span(s,   style=S(color=TEXT, fontSize='12px')),
                ]),
                'value': s,
            } for s, c in STATUS_COLORS.items()],
            value=list(STATUS_COLORS.keys()),
            labelStyle=S(display='flex', alignItems='center',
                         marginBottom='8px', cursor='pointer', gap='4px'),
            inputStyle=S(cursor='pointer', accentColor=ACCENT),
        ),

        html.Hr(style=S(border='none', borderTop=f'1px solid {BORDER}',
                        margin='16px 0 12px')),
        html.Div('Tracked', style=lbl_hdr),
        html.Div('Click any point in the map to track it.',
                 style=S(fontSize='10px', color=TEXT_DIM, marginBottom='8px')),
        html.Div(id='tracked-list', style=S(
            overflowY='auto', maxHeight='260px',
        )),
        html.Button('Clear all', id='clear-tracked',
                    n_clicks=0,
                    style=S(
                        marginTop='8px', background='none',
                        border=f'1px solid {BORDER}', color=TEXT_DIM,
                        borderRadius='4px', padding='3px 10px',
                        fontSize='11px', cursor='pointer',
                    )),

        html.Hr(style=S(border='none', borderTop=f'1px solid {BORDER}',
                        margin='14px 0')),
        html.Div('Color by', style=lbl_hdr),
        dcc.Dropdown(
            id='color-by-dropdown',
            options=[
                {'label': 'Call Status',      'value': 'status'},
                {'label': 'Pred Genotype',    'value': 'pred_gt'},
                {'label': 'GQ (quality)',     'value': 'GQ'},
                {'label': 'DP (read depth)',  'value': 'DP'},
                {'label': 'VAF (allele freq)', 'value': 'VAF'},
            ],
            value='status',
            clearable=False,
            style=S(
                background=SURFACE, color=TEXT, borderColor=BORDER,
                fontSize='12px',
            ),
        ),

        html.Div('Point size', style=lbl_hdr),
        dcc.Slider(
            id='pt-size-slider',
            min=1, max=12, step=0.5, value=4,
            marks={1: '1', 4: '4', 8: '8', 12: '12'},
            tooltip=dict(placement='bottom', always_visible=False),
        ),

        html.Div('Layer types', style=lbl_hdr),
        html.Div([
            html.Span('■ ', style=S(color=TEXT_DIM, fontSize='11px')),
            html.Span('Inception (mixed)', style=S(color=TEXT_DIM, fontSize='11px')),
        ], style=S(display='flex', alignItems='center', marginBottom='6px')),
        html.Div([
            html.Span('■ ', style=S(color=MP_COLOR, fontSize='11px')),
            html.Span('Max-pooling', style=S(color=MP_COLOR, fontSize='11px')),
        ], style=S(display='flex', alignItems='center', marginBottom='6px')),

        html.Hr(style=S(border='none', borderTop=f'1px solid {BORDER}',
                        margin='14px 0 10px')),
        dcc.Checklist(
            id='show-maxpool',
            options=[{
                'label': html.Span('Show max-pool layers',
                                   style=S(color=TEXT, fontSize='12px')),
                'value': 'show',
            }],
            value=[],  # hidden by default
            labelStyle=S(display='flex', alignItems='center',
                         cursor='pointer', gap='4px'),
            inputStyle=S(cursor='pointer', accentColor=MP_COLOR),
        ),
    ], style=S(
        width='210px', flexShrink='0',
        background=SURFACE,
        borderRight=f'1px solid {BORDER}',
        padding='14px 12px',
        overflowY='auto',
    ))

    # ── Column thumbnails ──────────────────────────────────────────────────
    col_cells = []
    for ci, (layer, layer_type) in enumerate(columns):
        is_mp = layer_type == 'maxpool'
        short = _short(layer)
        cell = html.Div([
            html.Div(short, style=S(
                textAlign='center', fontSize='10px',
                fontWeight='700' if is_mp else '400',
                color=MP_COLOR if is_mp else TEXT_DIM,
                letterSpacing='0.04em',
                padding='3px 0 3px',
                borderBottom=f'2px solid {MP_COLOR}' if is_mp else 'none',
                marginBottom='2px',
            )),
            html.Div(
                html.Img(
                    src=thumbs[ci],
                    style=S(width='100%', display='block', pointerEvents='none'),
                ),
                id={'type': 'col', 'idx': ci},
                n_clicks=0,
                className='traj-col',
                style=S(
                    cursor='pointer',
                    border=f'1.5px solid {MP_COLOR}' if is_mp
                           else f'1px solid {BORDER}',
                    boxSizing='border-box',
                ),
            ),
        ], className='mp-col' if is_mp else '',
           style=S(flex='1', minWidth='0', padding='0 2px',
                   display='none' if is_mp else ''))
        col_cells.append(cell)

    layer_strip = html.Div([
        html.Div(
            'Click column to expand  ·  click map point to track  ·  violet = max-pool',
            style=S(fontSize='10px', color=TEXT_DIM, padding='0 0 6px 2px'),
        ),
        html.Div(col_cells, style=S(
            display='flex', alignItems='flex-start',
            overflowX='hidden', paddingBottom='8px', gap='0',
        )),
    ], style=S(padding='12px 16px', borderBottom=f'1px solid {BORDER}'))

    # ── Detail panel ───────────────────────────────────────────────────────
    initial_fig = build_detail_figure(
        layers_data, 0, columns,
        list(STATUS_COLORS.keys()), [],
    )
    n0 = len(layers_data[columns[0][0]]['bg'])
    initial_hint = f'Viewing: Mixed 0  ·  {n0:,} variants  ·  0 tracked'

    detail = html.Div([
        html.Div(initial_hint, id='detail-hint',
                 style=S(fontSize='11px', color=TEXT_DIM, padding='0 0 4px 2px')),
        dcc.Graph(
            id='detail-graph',
            figure=initial_fig,
            config=S(displayModeBar=True, scrollZoom=True,
                     displaylogo=False, modeBarButtonsToRemove=['lasso2d']),
            style=S(flex='1', minHeight='220px', width='100%', backgroundColor=DARK_BG),
        ),
    ], style=S(
        padding='12px 16px', backgroundColor=DARK_BG,
        flex='1', minHeight='0',
        display='flex', flexDirection='column', overflow='hidden',
    ))

    # ── Bottom section (dynamic tracked-variants table + annotation image) ─
    bottom_section = build_bottom_section()

    # ── Layout ─────────────────────────────────────────────────────────────
    app.layout = html.Div([
        html.Div([
            html.Div([
                html.Span('Miscall Trajectory Explorer',
                          style=S(fontWeight='700', fontSize='15px', color=TEXT)),
                html.Span(
                    'HG002 chr20  ·  ~45k variants  ·  inception mixed0–10 + max-pool layers',
                    style=S(fontSize='11px', color=TEXT_DIM, marginLeft='16px'),
                ),
            ], style=S(
                background=SURFACE, borderBottom=f'1px solid {BORDER}',
                padding='10px 20px', display='flex', alignItems='center',
                flexShrink='0',
            )),
            html.Div([
                sidebar,
                html.Div([
                    layer_strip,
                    html.Div([
                        dcc.Loading(
                            html.Div(id='variant-trajectory-strip'),
                            type='dot', color=ACCENT,
                        ),
                        html.Button('×',
                            id='close-trajectory',
                            n_clicks=0,
                            style=S(
                                position='absolute', top='8px', right='14px',
                                background='none', border='none', color=TEXT_DIM,
                                cursor='pointer', fontSize='18px', lineHeight='1',
                                padding='0', zIndex='10', opacity='0.6',
                            ),
                        ),
                    ], style=S(position='relative', flexShrink='0')),
                    detail,
                ], style=S(flex='1', display='flex', flexDirection='column',
                           overflow='hidden', minWidth='0')),
            ], style=S(display='flex', flex='1', overflow='hidden')),

            dcc.Store(id='selected-col', data=0),
            dcc.Store(id='tracked-variants', data=[]),
            dcc.Store(id='trajectory-eid', data=None),
            html.Div(id='_dummy', style=S(display='none')),
        ], style=S(
            background=DARK_BG, color=TEXT,
            fontFamily='-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif',
            height='100vh', display='flex', flexDirection='column', overflow='hidden',
            flexShrink='0',
        )),

        html.Div([
            html.Hr(style=S(border='none', borderTop=f'1px solid {BORDER}', margin='0')),
            bottom_section,
        ], style=S(background=DARK_BG)),

    ], style=S(
        background=DARK_BG, color=TEXT,
        fontFamily='-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif',
        overflowY='auto',
    ))

    # ── Callbacks ──────────────────────────────────────────────────────────

    @app.callback(
        Output('detail-graph', 'figure'),
        Output('detail-hint', 'children'),
        Output('selected-col', 'data'),
        Input({'type': 'col', 'idx': ALL}, 'n_clicks'),
        Input('status-checklist', 'value'),
        Input('pt-size-slider', 'value'),
        Input('tracked-variants', 'data'),
        Input('color-by-dropdown', 'value'),
        State('selected-col', 'data'),
        prevent_initial_call=True,
    )
    def on_interaction(_clicks, vis_statuses, pt_size, tracked_eids, color_by, current_col):
        if isinstance(ctx.triggered_id, dict) and ctx.triggered_id.get('type') == 'col':
            col_idx = ctx.triggered_id['idx']
        else:
            col_idx = current_col if current_col is not None else 0

        layer, _ = columns[col_idx]
        ld   = layers_data[layer]
        disp = layer.replace('max_pooling2d_', 'MaxPool ').replace('mixed', 'Mixed ')
        n    = len(ld['bg'])
        nt   = len(tracked_eids or [])
        cb   = color_by or 'status'
        hint = f'Viewing: {disp}  ·  {n:,} variants  ·  {nt} tracked  ·  color: {cb}'
        fig  = build_detail_figure(
            layers_data, col_idx, columns,
            vis_statuses or list(STATUS_COLORS.keys()),
            tracked_eids or [],
            pt_size=pt_size or 4,
            color_by=cb,
        )
        return fig, hint, col_idx

    @app.callback(
        Output('tracked-variants', 'data', allow_duplicate=True),
        Input('detail-graph', 'clickData'),
        State('tracked-variants', 'data'),
        prevent_initial_call=True,
    )
    def on_click_point(click_data, tracked):
        if not click_data:
            return no_update
        point = click_data['points'][0]
        eid = point.get('customdata')
        if eid is None:
            return no_update
        eid = int(eid)
        if eid in (tracked or []):
            return no_update
        return (tracked or []) + [eid]

    @app.callback(
        Output('tracked-variants', 'data', allow_duplicate=True),
        Input({'type': 'rm-variant', 'eid': ALL}, 'n_clicks'),
        State('tracked-variants', 'data'),
        prevent_initial_call=True,
    )
    def remove_variant(clicks, tracked):
        if not any(c for c in (clicks or []) if c):
            return no_update
        triggered = ctx.triggered_id
        if triggered and triggered.get('type') == 'rm-variant':
            eid = triggered['eid']
            return [e for e in (tracked or []) if e != eid]
        return no_update

    @app.callback(
        Output('tracked-variants', 'data', allow_duplicate=True),
        Input('clear-tracked', 'n_clicks'),
        prevent_initial_call=True,
    )
    def clear_tracked(_):
        return []

    @app.callback(
        Output('trajectory-eid', 'data'),
        Input('detail-graph', 'clickData'),
        prevent_initial_call=True,
    )
    def on_click_set_trajectory(click_data):
        if not click_data:
            return no_update
        point = click_data['points'][0]
        eid = point.get('customdata')
        if eid is None:
            return no_update
        return int(eid)

    @app.callback(
        Output('trajectory-eid', 'data', allow_duplicate=True),
        Input('close-trajectory', 'n_clicks'),
        prevent_initial_call=True,
    )
    def close_trajectory(_):
        return None

    @app.callback(
        Output('variant-trajectory-strip', 'children'),
        Input('trajectory-eid', 'data'),
        Input('show-maxpool', 'value'),
        State('tracked-variants', 'data'),
    )
    def update_variant_trajectory_strip(eid, show_mp, tracked):
        if not eid:
            return html.Div(
                'Click any point in the map to view its layer trajectory.',
                style=S(fontSize='10px', color=TEXT_DIM, padding='6px 0'),
            )
        eid = int(eid)
        track_color = _TRACK_COLORS[0]
        if tracked and eid in tracked:
            vi = tracked.index(eid)
            track_color = _TRACK_COLORS[vi % len(_TRACK_COLORS)]
        show_maxpool = 'show' in (show_mp or [])
        return build_variant_trajectory_strip(
            eid, layers_data, columns, track_color, show_maxpool,
        )

    @app.callback(
        Output('tracked-list', 'children'),
        Input('tracked-variants', 'data'),
    )
    def update_tracked_list(tracked):
        if not tracked:
            return html.Div('None', style=S(fontSize='11px', color=TEXT_DIM))
        items = []
        for vi, eid in enumerate(tracked):
            info   = _eid_info.get(eid, {})
            status = info.get('status', '?')
            chrom  = info.get('chrom', '?')
            start  = info.get('start', '?')
            tcolor = _TRACK_COLORS[vi % len(_TRACK_COLORS)]
            scolor = STATUS_COLORS.get(status, TEXT)
            loc    = f'{chrom}:{start:,}' if isinstance(start, (int, float)) else f'eid:{eid}'
            items.append(html.Div([
                html.Span(f'#{vi+1}', style=S(color=tcolor, fontSize='10px',
                                               fontWeight='700', minWidth='20px')),
                html.Div([
                    html.Div(loc,    style=S(color=TEXT, fontSize='10px', lineHeight='1.2')),
                    html.Div(status, style=S(color=scolor, fontSize='9px')),
                ], style=S(flex='1', minWidth='0', overflow='hidden')),
                html.Button('×',
                    id={'type': 'rm-variant', 'eid': eid},
                    n_clicks=0,
                    style=S(background='none', border='none', color=TEXT_DIM,
                            cursor='pointer', fontSize='14px', padding='0 2px',
                            lineHeight='1', flexShrink='0'),
                ),
            ], style=S(
                display='flex', alignItems='center', gap='4px',
                padding='3px 0', borderBottom=f'1px solid {BORDER}',
            )))
        return items

    @app.callback(
        Output('tracked-table-body', 'children'),
        Input('tracked-variants', 'data'),
    )
    def update_tracked_table(tracked):
        def td(content, color=None, bold=False):
            return html.Td(content, style=S(
                padding='6px 12px', fontSize='11px',
                color=color or TEXT, fontWeight='700' if bold else '400',
                borderBottom=f'1px solid {BORDER}',
            ))

        if not tracked:
            return [html.Tr([html.Td(
                'No variants tracked — click any point in the map above.',
                colSpan=11,
                style=S(padding='12px', fontSize='11px', color=TEXT_DIM,
                        textAlign='center'),
            )])]
        rows = []
        for vi, eid in enumerate(tracked):
            info   = _eid_info.get(eid, {})
            status = info.get('status', '?')
            tcolor = _TRACK_COLORS[vi % len(_TRACK_COLORS)]
            scolor = STATUS_COLORS.get(status, TEXT)
            start  = info.get('start', '?')
            start_s = f"{start:,}" if isinstance(start, (int, float)) else str(start)
            gq  = info.get('GQ');   gq_s  = str(int(gq))  if gq  is not None and not (isinstance(gq,  float) and np.isnan(gq))  else '–'
            dp  = info.get('DP');   dp_s  = str(int(dp))  if dp  is not None and not (isinstance(dp,  float) and np.isnan(dp))  else '–'
            vaf = info.get('VAF');  vaf_s = f'{vaf:.3f}'  if vaf is not None and not (isinstance(vaf, float) and np.isnan(vaf)) else '–'
            ref = info.get('ref', '?')
            alt = info.get('alt', '?')
            rows.append(html.Tr([
                td(f'#{vi+1}',           color=tcolor, bold=True),
                td(str(eid)),
                td(status,               color=scolor),
                td(info.get('chrom',     '?')),
                td(start_s,              color=ACCENT),
                td(f'{ref}→{alt}'),
                td(info.get('pred_gt',   '?')),
                td(info.get('truth_gt',  '?')),
                td(gq_s),
                td(dp_s),
                td(vaf_s),
            ]))
        return rows

    @app.callback(
        Output('flip-trajectory', 'children'),
        Input('tracked-variants', 'data'),
    )
    def update_flip_trajectory(tracked):
        if not tracked:
            return html.Div('No variants tracked.',
                            style=S(fontSize='11px', color=TEXT_DIM))

        assignments = compute_flip_assignments(layers_data, columns, tracked)
        col_labels = [_short(l) for l, _ in columns]

        th_s = S(padding='4px 8px', fontSize='9px', textTransform='uppercase',
                 letterSpacing='0.06em', color=TEXT_DIM,
                 borderBottom=f'1px solid {BORDER}', fontWeight='600',
                 whiteSpace='nowrap')
        td_s = S(padding='3px 6px', fontSize='10px',
                 borderBottom=f'1px solid {BORDER}', textAlign='center',
                 whiteSpace='nowrap')

        header = html.Tr([html.Th('Variant', style=th_s)] +
                         [html.Th(lbl, style=th_s) for lbl in col_labels])
        rows = []
        for vi, eid in enumerate(tracked):
            tcolor = _TRACK_COLORS[vi % len(_TRACK_COLORS)]
            assgn  = assignments.get(eid, [None] * len(columns))
            cells  = [html.Td(f'#{vi+1} eid:{eid}',
                              style=S(**td_s, color=tcolor, fontWeight='700',
                                      textAlign='left'))]
            prev = None
            for a in assgn:
                if a is None:
                    cells.append(html.Td('–', style=S(**td_s, color=TEXT_DIM)))
                    prev = None
                    continue
                acolor = _ASSIGN_COLORS.get(a, TEXT)
                border = ''
                if prev is not None and a != prev:
                    border = f'2px solid {FLIP_COLOR}'
                cells.append(html.Td(
                    html.Span(a, style=S(
                        background=acolor + '28',
                        color=acolor, borderRadius='3px',
                        padding='1px 5px', fontSize='9px', fontWeight='600',
                    )),
                    style=S(**td_s, borderLeft=border),
                ))
                prev = a
            rows.append(html.Tr(cells))

        return html.Table(
            [html.Thead(header), html.Tbody(rows)],
            style=S(borderCollapse='collapse', background=SURFACE,
                    border=f'1px solid {BORDER}', borderRadius='4px',
                    minWidth='max-content'),
        )

    # Clientside: highlight selected column cell
    app.clientside_callback(
        """
        function(col_idx) {
            document.querySelectorAll('.traj-col').forEach(function(el) {
                el.classList.remove('traj-col-selected');
            });
            document.querySelectorAll('.traj-col').forEach(function(el) {
                var id = el.getAttribute('id') || '';
                var m = id.match(/"idx":(\\d+)/);
                if (m && parseInt(m[1]) === col_idx) {
                    el.classList.add('traj-col-selected');
                }
            });
            return window.dash_clientside.no_update;
        }
        """,
        Output('_dummy', 'children'),
        Input('selected-col', 'data'),
    )

    # Clientside: toggle max-pool column visibility in the top strip
    app.clientside_callback(
        """
        function(show_mp) {
            var show = show_mp && show_mp.includes('show');
            document.querySelectorAll('.mp-col').forEach(function(el) {
                el.style.display = show ? '' : 'none';
            });
            return window.dash_clientside.no_update;
        }
        """,
        Output('_dummy', 'children', allow_duplicate=True),
        Input('show-maxpool', 'value'),
        prevent_initial_call=True,
    )

    app.index_string = app.index_string.replace(
        '</head>',
        f"""<style>
html, body, #react-entry-point, #_dash-app-content {{
    background: {DARK_BG} !important;
    margin: 0; padding: 0;
}}
#detail-graph, #detail-graph .js-plotly-plot,
#detail-graph .plot-container, #detail-graph .svg-container {{
    background: {DARK_BG} !important;
}}
.traj-col {{ transition: outline 0.08s; }}
.traj-col:hover {{ outline: 1px solid {TEXT_DIM} !important; z-index: 2; }}
.traj-col-selected {{ outline: 2px solid {ACCENT} !important; z-index: 3; }}
</style></head>""",
    )

    return app


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print('Loading data...')
    layers_data, columns = load_data()
    print(f'Columns: {[l for l, _ in columns]}')

    _REPORT_PATH = Path(__file__).parent / 'assets' / 'flip_analysis_report.html'
    print('\nGenerating analysis report…')
    _generate_report(layers_data, columns, _eid_info, _REPORT_PATH)

    app = build_app(layers_data, columns)
    print('\nStarting at http://127.0.0.1:8051')
    app.run(debug=False, port=8051)


if __name__ == '__main__':
    main()
