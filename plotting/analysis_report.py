"""Flip analysis report for DeepVariant miscall trajectory data.

Computes per-variant flip stats across all mixed inception layers and
generates a self-contained interactive HTML report.
"""
from __future__ import annotations

import random
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Theme (matches trajectory_explorer.py) ────────────────────────────────────

DARK_BG  = '#0d1117'
SURFACE  = '#161b22'
BORDER   = '#30363d'
TEXT     = '#e6edf3'
TEXT_DIM = '#8b949e'
ACCENT   = '#58a6ff'

STATUS_COLORS = {
    'TP':      '#22c55e',
    'FP':      '#ef4444',
    'FN':      '#f97316',
    'RefCall': '#3b82f6',
}
STATUS_ORDER = ['TP', 'FP', 'FN', 'RefCall']

_LAYOUT_BASE = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor=DARK_BG,
    font=dict(color=TEXT,
              family='-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif',
              size=12),
    legend=dict(bgcolor=SURFACE, bordercolor=BORDER, borderwidth=1,
                font=dict(color=TEXT, size=11)),
    margin=dict(l=64, r=24, t=64, b=48),
)
_AXIS = dict(gridcolor=BORDER, zerolinecolor=BORDER,
             tickfont=dict(color=TEXT_DIM))


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def _short(layer: str) -> str:
    if layer.startswith('max_pooling2d_'):
        return 'MP' + layer[-1]
    return 'M' + layer.replace('mixed', '')


# ── Core computation ──────────────────────────────────────────────────────────

def compute_all_assignments(
    layers_data: dict,
    mixed_cols: list[tuple[str, str]],
    eids: list[int],
    k: int = 10,
    majority_thresh: float = 0.7,
) -> tuple[pd.DataFrame, dict[int, float]]:
    """k-NN assignment for every eid at every mixed layer.

    Returns (assignments_df, mean_jaccard_dict).
      assignments_df : DataFrame indexed by eid, columns = layer names,
                       values = 'TP'|'FP'|'FN'|'RefCall'|'Mixed'|None
      mean_jaccard   : mean Jaccard similarity of k-NN sets across consecutive
                       mixed-layer pairs (UMAP space); lower = more unstable
    """
    layer_names   = [l for l, _ in mixed_cols]
    rows:          list[dict]      = []
    jaccard_by_eid: dict[int, float] = {}

    for i, eid in enumerate(eids):
        if i % 50 == 0:
            print(f'  assignments {i}/{len(eids)} …', end='\r', flush=True)

        row:          dict               = {'eid': eid}
        prev_nn_set:  frozenset | None   = None
        jaccard_vals: list[float]        = []

        for layer, _ in mixed_cols:
            ld = layers_data.get(layer)
            if ld is None or eid not in ld['eid_idx']:
                row[layer]   = None
                prev_nn_set  = None
                continue

            ri  = ld['eid_idx'][eid]
            bg  = ld['bg']
            vx  = float(bg['x'].iat[ri])
            vy  = float(bg['y'].iat[ri])
            bx  = bg['x'].values
            by  = bg['y'].values
            bs  = bg['status'].values

            sq  = (bx - vx) ** 2 + (by - vy) ** 2
            kk  = min(k + 1, len(sq))
            nn  = np.argpartition(sq, kk - 1)[:kk]
            nn  = nn[np.argsort(sq[nn])]
            nn  = [j for j in nn if j != ri][:k]

            nn_set = frozenset(nn)

            # Assignment label
            counts = Counter(bs[nn].tolist())
            total  = len(nn)
            if not total:
                row[layer] = None
            else:
                top        = max(counts, key=counts.get)
                row[layer] = top if counts[top] / total >= majority_thresh else 'Mixed'

            # Jaccard vs previous layer
            if prev_nn_set is not None:
                u = len(prev_nn_set | nn_set)
                if u:
                    jaccard_vals.append(len(prev_nn_set & nn_set) / u)

            prev_nn_set = nn_set

        rows.append(row)
        jaccard_by_eid[eid] = float(np.mean(jaccard_vals)) if jaccard_vals else np.nan

    print(f'  assignments {len(eids)}/{len(eids)} done       ')
    df = pd.DataFrame(rows).set_index('eid')
    return df[layer_names], jaccard_by_eid


def compute_stats(
    assignments: pd.DataFrame,
    eid_info: dict,
    layer_names: list[str],
    jaccard_by_eid: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Per-variant statistics derived from the assignment trajectory."""
    records: list[dict] = []

    for eid in assignments.index:
        info   = eid_info.get(int(eid), {})
        status = info.get('status', '?')
        assgns = [(i, assignments.at[eid, l]) for i, l in enumerate(layer_names)
                  if assignments.at[eid, l] is not None]

        flip_count     = 0
        commitment_idx = 0
        prev: str | None = None
        for i, a in assgns:
            if prev is not None and a != prev:
                flip_count     += 1
                commitment_idx  = i
            prev = a

        # Correct target neighborhood: TP for FN (should have been called),
        # RefCall for FP (should not have been called).
        correct_target   = {'FP': 'RefCall', 'FN': 'TP'}.get(status)
        ever_correct     = (any(a == correct_target for _, a in assgns)
                            if correct_target else None)
        final_assignment = assgns[-1][1] if assgns else None

        records.append({
            'eid':              eid,
            'status':           status,
            'flip_count':       flip_count,
            'commitment_idx':   commitment_idx,
            'commitment_layer': layer_names[commitment_idx],
            'ever_correct':     ever_correct,
            'correct_target':   correct_target,
            'final_assignment': final_assignment,
            'mean_jaccard':     (jaccard_by_eid or {}).get(int(eid), np.nan),
            'GQ':               info.get('GQ',  np.nan),
            'DP':               info.get('DP',  np.nan),
            'VAF':              info.get('VAF', np.nan),
        })

    return pd.DataFrame(records)


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_flip_count_violin(stats: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for s in STATUS_ORDER:
        vals = stats.loc[stats['status'] == s, 'flip_count'].tolist()
        if not vals:
            continue
        fig.add_trace(go.Violin(
            y=vals, name=s,
            box_visible=True, meanline_visible=True,
            fillcolor=_rgba(STATUS_COLORS[s], 0.27),
            line_color=STATUS_COLORS[s],
            points='all', pointpos=0, jitter=0.35,
            marker=dict(size=3, opacity=0.4, color=STATUS_COLORS[s]),
        ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text='<b>Flip count distribution by call status</b>',
                   font=dict(size=15, color=ACCENT), x=0.01),
        xaxis=dict(**_AXIS),
        yaxis=dict(**_AXIS, title='Mixed-to-mixed flips per variant'),
        showlegend=False, height=440,
    )
    return fig


def fig_commitment_layer(stats: pd.DataFrame, layer_names: list[str]) -> go.Figure:
    shorts = [_short(l) for l in layer_names]
    fig = go.Figure()
    for s in STATUS_ORDER:
        sub    = stats[stats['status'] == s]
        counts = [(sub['commitment_idx'] == i).sum() for i in range(len(layer_names))]
        fig.add_trace(go.Bar(
            x=shorts, y=counts, name=s,
            marker_color=STATUS_COLORS[s], opacity=0.85,
        ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text='<b>Commitment layer — final assignment change by layer</b>',
                   font=dict(size=15, color=ACCENT), x=0.01),
        barmode='group',
        xaxis=dict(**_AXIS, title='Mixed layer'),
        yaxis=dict(**_AXIS, title='Variant count'),
        height=420,
    )
    return fig


def fig_jaccard_vs_flips(stats: pd.DataFrame) -> go.Figure:
    rng = np.random.default_rng(42)
    fig = go.Figure()
    for s in STATUS_ORDER:
        sub = stats[(stats['status'] == s) & stats['mean_jaccard'].notna()]
        if sub.empty:
            continue
        jitter = rng.uniform(-0.25, 0.25, size=len(sub))
        fig.add_trace(go.Scatter(
            x=sub['mean_jaccard'].tolist(),
            y=(sub['flip_count'].values + jitter).tolist(),
            mode='markers', name=s,
            marker=dict(
                size=7 if s in ('FP', 'FN') else 4,
                color=STATUS_COLORS[s],
                opacity=0.85 if s in ('FP', 'FN') else 0.25,
            ),
        ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text='<b>Mean Jaccard vs flip count — neighborhood stability across layers</b>',
                   font=dict(size=15, color=ACCENT), x=0.01),
        xaxis=dict(**_AXIS, title='Mean Jaccard similarity (0 = fully unstable, 1 = fully stable)',
                   range=[0, 1]),
        yaxis=dict(**_AXIS, title='Flip count (jittered)'),
        height=440,
    )
    return fig


def fig_recovery_rate(stats: pd.DataFrame) -> go.Figure:
    # Correct target: RefCall for FP (shouldn't be called), TP for FN (should be called)
    correct_label = {'FP': 'RefCall', 'FN': 'TP'}
    fig = go.Figure()
    for s in ['FP', 'FN']:
        sub = stats[stats['status'] == s]
        if sub.empty:
            continue
        target    = correct_label[s]
        recovered = int(sub['ever_correct'].sum())
        never     = len(sub) - recovered
        fig.add_trace(go.Bar(
            x=[f'Visited {target} neighborhood', f'Never reached {target}'],
            y=[recovered, never],
            name=s,
            marker_color=[STATUS_COLORS[s], _rgba(STATUS_COLORS[s], 0.27)],
            text=[str(recovered), str(never)],
            textposition='outside',
            textfont=dict(color=TEXT, size=13),
        ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text='<b>Recovery rate — did miscalls ever visit the correct neighborhood?</b>',
                   font=dict(size=15, color=ACCENT), x=0.01),
        barmode='group',
        xaxis=dict(**_AXIS),
        yaxis=dict(**_AXIS, title='Variant count'),
        height=400,
    )
    return fig


def fig_stabilization_count(stats: pd.DataFrame, layer_names: list[str]) -> go.Figure:
    shorts = [_short(l) for l in layer_names]
    fig = go.Figure()
    for s in STATUS_ORDER:
        sub   = stats[stats['status'] == s]
        total = len(sub)
        if total == 0:
            continue
        cum = [(sub['commitment_idx'] <= i).sum() for i in range(len(layer_names))]
        fig.add_trace(go.Scatter(
            x=shorts, y=cum,
            mode='lines+markers',
            name=f'{s} (n={total})',
            line=dict(color=STATUS_COLORS[s], width=2),
            marker=dict(size=6, color=STATUS_COLORS[s]),
        ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text='<b>Cumulative stabilization — variants settled by each layer (count)</b>',
                   font=dict(size=15, color=ACCENT), x=0.01),
        xaxis=dict(**_AXIS, title='Mixed layer'),
        yaxis=dict(**_AXIS, title='Variants stabilized (cumulative)'),
        height=440,
    )
    return fig


def fig_stabilization_pct(stats: pd.DataFrame, layer_names: list[str]) -> go.Figure:
    shorts = [_short(l) for l in layer_names]
    fig = go.Figure()
    for s in STATUS_ORDER:
        sub   = stats[stats['status'] == s]
        total = len(sub)
        if total == 0:
            continue
        pct = [100.0 * (sub['commitment_idx'] <= i).sum() / total
               for i in range(len(layer_names))]
        fig.add_trace(go.Scatter(
            x=shorts, y=pct,
            mode='lines+markers',
            name=f'{s} (n={total})',
            line=dict(color=STATUS_COLORS[s], width=2),
            marker=dict(size=6, color=STATUS_COLORS[s]),
        ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text='<b>Cumulative stabilization — normalized by group size (%)</b>',
                   font=dict(size=15, color=ACCENT), x=0.01),
        xaxis=dict(**_AXIS, title='Mixed layer'),
        yaxis=dict(**_AXIS, title='% of group stabilized',
                   range=[0, 105], tickformat='.0f', ticksuffix='%'),
        height=440,
    )
    return fig


# ── Report assembly ───────────────────────────────────────────────────────────

_SECTIONS = [
    ('flip_count',        'Flip Count Distribution',
     'How many mixed-to-mixed assignment changes does each variant undergo across the 11 inception '
     'layers? Zero flips means the network committed to its first neighborhood and never reconsidered. '
     'Higher flip counts indicate a variant that moved through multiple cluster regions before settling. '
     'If miscalls systematically show more flips than TPs, it suggests the network genuinely struggles '
     'to place them rather than confidently misclassifying them from the start.'),

    ('commitment',        'Commitment Layer',
     'The layer at which each variant makes its <em>final</em> assignment change — after this point '
     'its k-NN neighborhood label is stable through mixed10. Variants committing at early layers '
     '(M0–M3) are decided quickly; those committing at late layers (M7–M10) remain representationally '
     'ambiguous deep into the network. Comparing the commitment-layer distributions of TPs vs miscalls '
     'reveals whether miscalls are hard because they commit wrongly early, or because they stay '
     'unresolved longer.'),

    ('jaccard_flips',      'Mean Jaccard vs Flip Count',
     'Mean Jaccard similarity of k-NN neighbor sets across consecutive mixed-layer pairs, '
     'computed in UMAP space. A value near 1 means the variant\'s immediate neighborhood was '
     'nearly identical at every layer — a stable trajectory. A value near 0 means the neighborhood '
     'reshuffled completely between layers. Because flip count is derived from the same trajectory, '
     'the two should correlate negatively: more flips → lower mean Jaccard. The shape of the '
     'relationship reveals whether miscalls differ from correct calls in overall trajectory stability, '
     'or only at specific transition points.'),

    ('recovery',          'Recovery Rate',
     'For each miscall, did the variant ever pass through the <em>correct</em> neighborhood '
     'at any layer before committing to the wrong assignment? '
     'For <b>FN</b> (missed variants), the correct target is <b>TP</b> — the network should have '
     'placed it with true positives. '
     'For <b>FP</b> (spurious calls), the correct target is <b>RefCall</b> — the network should have '
     'placed it with reference sites. '
     '"Visited correct" means the network transiently represented the variant correctly but lost that '
     'representation — a near-miss failure. '
     '"Never reached correct" means the network never found the right cluster at any depth — a '
     'more fundamental representational failure. These two failure modes likely have different '
     'biological or sequencing-artefact causes.'),

    ('stab_count',        'Cumulative Stabilization (count)',
     'Cumulative count of variants whose trajectory has settled (made its last flip) by each layer, '
     'broken down by call status. A steep early rise means most variants of that type decide quickly; '
     'a gradual curve extending to late layers means the group stays ambiguous deep into the network. '
     'The absolute gap between TP and FP/FN curves shows how many more miscalls remain unsettled '
     'at each checkpoint.'),

    ('stab_pct',          'Cumulative Stabilization (%)',
     'Same as above, normalized to percentage of each group. Removes the effect of group size '
     'differences and allows direct comparison of stabilization <em>rates</em> between TP, FP, FN, '
     'and RefCall populations. If FP/FN curves lag substantially behind TP, it means miscalls '
     'require more network depth to resolve — regardless of how many there are.'),
]


def generate_report(
    layers_data: dict,
    columns: list[tuple[str, str]],
    eid_info: dict,
    output_path: Path,
    k: int = 10,
    n_tp_sample: int = 500,
    n_ref_sample: int = 500,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)

    mixed_cols  = [(l, t) for l, t in columns if t == 'inception']
    layer_names = [l for l, _ in mixed_cols]
    ref_layer   = layer_names[0]
    valid_in_data = set(layers_data[ref_layer]['eid_idx'].keys())

    by_status: dict[str, list[int]] = {}
    for eid, info in eid_info.items():
        if eid in valid_in_data:
            by_status.setdefault(info.get('status', '?'), []).append(eid)

    fp_eids  = by_status.get('FP', [])
    fn_eids  = by_status.get('FN', [])
    tp_eids  = rng.sample(by_status.get('TP', []),
                          min(n_tp_sample, len(by_status.get('TP', []))))
    ref_eids = rng.sample(by_status.get('RefCall', []),
                          min(n_ref_sample, len(by_status.get('RefCall', []))))
    sample   = fp_eids + fn_eids + tp_eids + ref_eids

    print(f'  Sample: {len(fp_eids)} FP  {len(fn_eids)} FN  '
          f'{len(tp_eids)} TP  {len(ref_eids)} RefCall  '
          f'({len(sample)} total)')

    assignments, jaccard_by_eid = compute_all_assignments(layers_data, mixed_cols, sample, k=k)
    stats = compute_stats(assignments, eid_info, layer_names, jaccard_by_eid)

    # ── Summary table ─────────────────────────────────────────────────────────
    shorts = [_short(l) for l in layer_names]
    summary_rows = ''
    for s in STATUS_ORDER:
        sub = stats[stats['status'] == s]
        if sub.empty:
            continue
        n            = len(sub)
        mean_flips   = sub['flip_count'].mean()
        pct_zero     = 100.0 * (sub['flip_count'] == 0).mean()
        med_commit   = shorts[int(sub['commitment_idx'].median())]
        pct_multi    = 100.0 * (sub['flip_count'] > 1).mean()
        target       = {'FP': 'RefCall', 'FN': 'TP'}.get(s)
        ever_tp_str  = (f"{100.0 * sub['ever_correct'].mean():.1f}% (→{target})"
                        if target else '—')
        c = STATUS_COLORS[s]
        summary_rows += (
            f'<tr>'
            f'<td style="color:{c};font-weight:700">{s}</td>'
            f'<td>{n:,}</td>'
            f'<td>{mean_flips:.2f}</td>'
            f'<td>{pct_zero:.1f}%</td>'
            f'<td>{pct_multi:.1f}%</td>'
            f'<td>{med_commit}</td>'
            f'<td>{ever_tp_str}</td>'
            f'</tr>\n'
        )

    # ── Figures ───────────────────────────────────────────────────────────────
    figs = {
        'flip_count': fig_flip_count_violin(stats),
        'commitment': fig_commitment_layer(stats, layer_names),
        'jaccard_flips': fig_jaccard_vs_flips(stats),
        'recovery':   fig_recovery_rate(stats),
        'stab_count': fig_stabilization_count(stats, layer_names),
        'stab_pct':   fig_stabilization_pct(stats, layer_names),
    }

    plot_blocks = ''
    for i, (key, title, desc) in enumerate(_SECTIONS):
        fig_html = figs[key].to_html(
            full_html=False,
            include_plotlyjs='cdn' if i == 0 else False,
            config={'displayModeBar': False, 'responsive': True},
        )
        plot_blocks += (
            f'<div class="section">'
            f'<h2>{title}</h2>'
            f'<p class="desc">{desc}</p>'
            f'<div class="plot">{fig_html}</div>'
            f'</div>\n'
        )

    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DeepVariant Flip Analysis</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e6edf3;
     font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
     padding:40px 36px 100px;max-width:1240px;margin:0 auto}}
a{{color:#58a6ff;text-decoration:none}}
a:hover{{text-decoration:underline}}
.back{{display:inline-block;margin-bottom:28px;font-size:13px}}
h1{{font-size:22px;color:#58a6ff;margin-bottom:6px}}
.meta{{font-size:12px;color:#8b949e;margin-bottom:36px;line-height:1.7}}
hr{{border:none;border-top:1px solid #30363d;margin:48px 0}}
h2{{font-size:15px;color:#58a6ff;margin-bottom:8px}}
.desc{{font-size:13px;color:#8b949e;line-height:1.65;
       margin-bottom:16px;max-width:820px}}
.section{{margin-bottom:56px}}
.plot{{width:100%}}
table{{border-collapse:collapse;width:100%;margin-bottom:48px;font-size:13px}}
th{{padding:8px 14px;text-align:left;font-size:11px;text-transform:uppercase;
    letter-spacing:.06em;color:#8b949e;border-bottom:1px solid #30363d;font-weight:600}}
td{{padding:9px 14px;border-bottom:1px solid #30363d}}
tr:last-child td{{border-bottom:none}}
tbody tr:hover td{{background:#161b22}}
</style>
</head>
<body>
<a class="back" href="/">← Back to explorer</a>
<h1>DeepVariant Flip Analysis</h1>
<p class="meta">
  Generated {now}&nbsp;&nbsp;·&nbsp;&nbsp;
  {len(sample):,} variants analysed
  ({len(fp_eids)} FP · {len(fn_eids)} FN · {len(tp_eids)} TP · {len(ref_eids)} RefCall)&nbsp;&nbsp;·&nbsp;&nbsp;
  k = {k} nearest neighbours&nbsp;&nbsp;·&nbsp;&nbsp;
  {len(layer_names)} mixed layers (max-pool excluded from flip detection)
</p>

<h2>Summary statistics</h2>
<table>
  <thead><tr>
    <th>Status</th>
    <th>n</th>
    <th>Mean flips</th>
    <th>Zero-flip %</th>
    <th>Multi-flip %</th>
    <th>Median commit layer</th>
    <th>Ever visited correct</th>
  </tr></thead>
  <tbody>{summary_rows}</tbody>
</table>

<hr>
{plot_blocks}
<a class="back" href="/">← Back to explorer</a>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding='utf-8')
    print(f'  Report → {output_path}')
