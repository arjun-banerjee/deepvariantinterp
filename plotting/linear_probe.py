#!/usr/bin/env python3
"""Linear probe for superpopulation classification from DeepVariant activations.

Trains a probe (linear logistic regression or small MLP) on intermediate
representations (mixed1, mixed5, …) using leave-one-group-out CV at sample
resolution to avoid leakage.

Usage:
  python plotting/linear_probe.py \\
    --cache_dir CACHE_DIR \\
    --population_metadata data/1kg_file_mapping.csv \\
    --layer mixed5 --probe_kind linear

  Multiple layers:

  python plotting/linear_probe.py ... --layers mixed1,mixed3,mixed5,mixed7,mixed10 \\
    --probe_kind linear

  Training loss PNG (mean LOGO MLP folds only) when probing with ``--probe_kind mlp``.

  python plotting/linear_probe.py ... --probe_kind mlp --plot_loss_curve

Requires activation_cache/ with multiple activations_*.npz shards (one shard per
sample, matching ``file_index`` in the CSV) so leave-one-group-out CV has at
least two samples to split.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        '--cache_dir',
        required=True,
        help='Directory containing activations_*.npz files.',
    )
    p.add_argument(
        '--population_metadata',
        required=True,
        help=(
            'Path to CSV with population labels. Must contain file_index '
            'and super_population columns (e.g. data/1kg_file_mapping.csv '
            'or 1kg_file_mapping_v3.csv).'
        ),
    )
    p.add_argument(
        '--layer',
        default=None,
        help=(
            'Layer key in npz (e.g. mixed5). Default: first key in file. '
            'Ignored if --layers is set.'
        ),
    )
    p.add_argument(
        '--layers',
        default=None,
        help=(
            'Comma-separated layer keys to run sequentially (writes one '
            '_<layer>_ suffix per output stem). Requires activations.npz '
            'to contain these keys.'
        ),
    )
    p.add_argument(
        '--output',
        '-o',
        default='linear_probe_results.png',
        help='Output PNG path for confusion matrix visualization.',
    )
    p.add_argument(
        '--output_metrics',
        default=None,
        help='Output path for metrics JSON. Default: derived from --output.',
    )
    p.add_argument(
        '--subtract_sample_mean',
        action='store_true',
        help=(
            'Subtract per-sample mean from each variant activation before '
            'probing. Removes sample-level bias while keeping all data points, '
            'testing whether superpopulation info is in variant-level patterns.'
        ),
    )
    p.add_argument(
        '--use_sample_means',
        action='store_true',
        help=(
            'Probe on per-sample mean vectors only (one data point per sample). '
            'Tests whether superpopulation info is in sample-level averages.'
        ),
    )
    p.add_argument(
        '--n_permutations',
        type=int,
        default=100,
        help='Number of label-shuffle permutations for the null baseline.',
    )
    p.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility.',
    )
    p.add_argument(
        '--pca_components',
        type=int,
        default=50,
        help='PCA components used inside each CV fold.',
    )
    p.add_argument(
        '--logreg_c',
        type=float,
        default=1.0,
        help='Inverse regularization strength C for logistic regression.',
    )
    p.add_argument(
        '--class_weight_balanced',
        action='store_true',
        help='Use class_weight=\"balanced\" in logistic regression.',
    )
    p.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Cap on number of variants to load (useful for quick testing).',
    )
    p.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='Figure DPI.',
    )
    p.add_argument(
        '--probe_kind',
        choices=['linear', 'mlp'],
        default='linear',
        help='Probe: multinomial logistic (linear) or small MLP (nonlinear).',
    )
    p.add_argument(
        '--epochs',
        type=int,
        default=80,
        help='MLP: max_iter per LOGO fold (tol inflated so training runs epochs).',
    )
    p.add_argument(
        '--mlp_hidden',
        type=str,
        default='128',
        help='MLP hidden sizes, comma-separated (e.g. 128 or 256,128).',
    )
    p.add_argument(
        '--plot_loss_curve',
        action='store_true',
        help=(
            'MLP only: save mean LOGO-fold training loss vs iteration (`*_train_loss.png`). '
            '(Ignored when --probe_kind linear.)'
        ),
    )

    return p.parse_args()


def _parse_mlp_hidden(s: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in s.split(',') if p.strip()]
    if not parts:
        return (64,)
    return tuple(int(x) for x in parts)


def resolve_layer_list(args: argparse.Namespace) -> List[Optional[str]]:
    if getattr(args, 'layers', None):
        xs = [x.strip() for x in args.layers.split(',') if x.strip()]
        return xs
    return [args.layer]


def resolve_output_paths(
    output: str,
    layer_name: str,
    *,
    suffix: Optional[str] = None,
) -> Tuple[str, str]:
    stem, ext = os.path.splitext(output)
    if ext == '':
        ext = '.png'
        stem = output
    tag = suffix or layer_name.replace('/', '_')
    out_png = f'{stem}_{tag}{ext}'
    out_metrics = f'{stem}_{tag}_metrics.json'
    return out_png, out_metrics


def plot_training_loss_curve(
    train_curve: np.ndarray,
    iter_axis: np.ndarray,
    output_path: str,
    title: str,
    dpi: int = 150,
) -> None:
    """Save mean ± std cross-entropy / sklearn loss vs iteration."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    mean_tr = train_curve.mean(axis=0)
    std_tr = train_curve.std(axis=0)
    x = iter_axis[: mean_tr.shape[0]]
    ax.plot(x, mean_tr, label='train (mean across folds)', color='#4C72B0')
    ax.fill_between(
        x, mean_tr - std_tr, mean_tr + std_tr, alpha=0.25, color='#4C72B0')
    ax.set_xlabel('Iteration / epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, linestyle='--')
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved probe training loss curve: {output_path}')


def _make_mlp_classifier(
    epochs: int,
    mlp_hidden: Tuple[int, ...],
    random_state: int,
):
    """Adam MLP for LOGO probing / loss curves.

    Do not pass enormous ``tol`` (e.g. 1e100): sklearn computes ``best_loss_ - tol``,
    which overflows / raises cast warnings during training bookkeeping.
    We keep default-like ``tol`` and set ``n_iter_no_change=max_iter`` so training
    usually runs up to ``max_iter`` epochs unless loss diverges.
    """
    from sklearn.neural_network import MLPClassifier

    n_ep = max(int(epochs), 1)
    return MLPClassifier(
        hidden_layer_sizes=mlp_hidden,
        activation='relu',
        solver='adam',
        max_iter=n_ep,
        early_stopping=False,
        tol=1e-4,
        n_iter_no_change=n_ep,
        random_state=int(random_state % (2 ** 31)),
        learning_rate_init=1e-3,
        batch_size='auto',
        verbose=False,
    )


def collect_logo_training_curves_mlp(
    X: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    epochs: int,
    pca_components: int,
    mlp_hidden: Tuple[int, ...],
    class_weight_balanced: bool,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """One MLPClassifier fit per LOGO fold; stack loss_curve_ (padded)."""
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.utils.class_weight import compute_sample_weight

    le = LabelEncoder()
    y = le.fit_transform(labels)

    logo = LeaveOneGroupOut()
    curves: List[np.ndarray] = []
    rs = random_state

    for train_idx, test_idx in logo.split(X, y, groups):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        X_train, X_test = _reduce_dimensions(
            X_train, X_test, max_components=pca_components
        )
        yt = y[train_idx]
        sw = compute_sample_weight('balanced', yt) if class_weight_balanced else None
        clf = _make_mlp_classifier(epochs, mlp_hidden, rs)
        rs += 7919

        clf.fit(X_train, yt, sample_weight=sw)

        lc = np.asarray(clf.loss_curve_, dtype=np.float64)
        curves.append(lc)

    maxlen = max(len(c) for c in curves)
    stacked = []
    for c in curves:
        if len(c) < maxlen:
            pad = np.full(maxlen - len(c), c[-1])
            stacked.append(np.concatenate([c, pad]))
        else:
            stacked.append(c)
    mat = np.vstack(stacked)
    itr = np.arange(1, maxlen + 1, dtype=float)
    return mat, itr


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_activations_with_shape(
    cache_dir: str,
    layer: str | None,
) -> Tuple[np.ndarray, str, np.ndarray, List[str], Tuple[int, ...]]:
    """Load activation tensors, returning both pooled features and raw shape.

    Returns:
        X: (N, C) array after global average pooling over spatial dims.
        layer_name: resolved layer key.
        file_indices: (N,) int array mapping each row to its source file index.
        file_paths: list of source .npz paths.
        raw_shape: the (H, W, C) shape of a single activation tensor.
    """
    pattern = os.path.join(cache_dir, 'activations_*.npz')
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f'No files matching {pattern}')

    rows: list[np.ndarray] = []
    file_idx_per_row: list[int] = []
    resolved_layer: str | None = layer
    raw_shape: Tuple[int, ...] | None = None

    for fi, path in enumerate(paths):
        data = np.load(path)
        keys = list(data.keys())
        if not keys:
            raise ValueError(f'Empty npz: {path}')
        if resolved_layer is None:
            resolved_layer = keys[0]
        if resolved_layer not in data:
            raise KeyError(
                f'Layer {resolved_layer!r} not in {path}; available: {keys}. '
                f'Likely stale shard (different --hook_layers). '
                f'Run with consistent hooks or probe only layers every file has.'
            )

        arr = np.asarray(data[resolved_layer], dtype=np.float32)

        # Expect (batch, H, W, C) or (H, W, C)
        if arr.ndim == 3:
            arr = arr[np.newaxis, ...]
        elif arr.ndim != 4:
            raise ValueError(
                f'{path} key {resolved_layer}: expected 3-4D, got {arr.shape}'
            )

        if raw_shape is None:
            raw_shape = arr.shape[1:]  # (H, W, C)

        # Global average pool: (batch, H, W, C) -> (batch, C)
        pooled = arr.mean(axis=(1, 2))
        rows.append(pooled)
        file_idx_per_row.extend([fi] * pooled.shape[0])

    X = np.vstack(rows)
    file_indices = np.array(file_idx_per_row, dtype=np.int32)
    assert resolved_layer is not None
    assert raw_shape is not None
    return X, resolved_layer, file_indices, paths, raw_shape


def validate_layer_keys_uniform_cache(
    cache_dir: str, layer_names: List[str],
) -> None:
    """Fail fast if mixed hooked runs left inconsistent keys across shards."""
    pattern = os.path.join(cache_dir, 'activations_*.npz')
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f'No files matching {pattern}')

    offenders: List[Tuple[str, List[str], List[str]]] = []
    for path in paths:
        with np.load(path) as z:
            keys_set = set(z.keys())
        missing = sorted([ln for ln in layer_names if ln not in keys_set])
        if missing:
            offenders.append((path, missing, sorted(keys_set)))

    if not offenders:
        return

    print(
        'ERROR: Activation shards disagree on hooked layers '
        '(re-run inference with consistent --hook_layers, or wipe stale shards).',
        file=sys.stderr,
    )
    print(
        '\n'
        '  Each activations_XXXXXXXX.npz must contain every layer you probe. '
        'Older runs often only cached mixed5; newer defaults use '
        'mixed1,mixed3,mixed5,mixed7,mixed10.',
        file=sys.stderr,
    )
    print('\nShards missing requested keys:', file=sys.stderr)
    for path, missing, avail in offenders:
        try:
            rel = os.path.relpath(path)
        except ValueError:
            rel = path
        print(f'  {rel}', file=sys.stderr)
        print(f'    missing: {missing!r}; available: {avail!r}', file=sys.stderr)
    print(
        '\nRemediation (recommended): '
        'remove combined cache + rerun hooked pipeline:',
        file=sys.stderr,
    )
    print(
        '  rm 1kg_hooked_output/activation_cache/activations_*.npz',
        file=sys.stderr,
    )
    print(
        '  rm -rf 1kg_hooked_output/samples/*/activation_cache  # stale per-sample shards',
        file=sys.stderr,
    )
    print(
        '  bash scripts/run_1kg_hooked_defaults.sh\n',
        file=sys.stderr,
    )
    print(
        'Or probe only layers every shard still has (e.g. --layer mixed5).',
        file=sys.stderr,
    )
    raise SystemExit(1)


def load_population_metadata(
    metadata_path: str,
) -> 'pandas.DataFrame':
    """Load population metadata CSV. Expects file_index + super_population."""
    import pandas as pd

    df = pd.read_csv(metadata_path)
    required = {'file_index', 'super_population'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f'Metadata missing columns: {missing}. Found: {list(df.columns)}'
        )
    return df


def assign_labels(
    file_indices: np.ndarray,
    population_df: 'pandas.DataFrame',
) -> Tuple[np.ndarray, np.ndarray]:
    """Map file indices to superpopulation labels and sample group IDs.

    Returns:
        labels: (N,) string array of superpopulation codes.
        groups: (N,) int array of sample IDs (for group-aware CV splits).
    """
    labels = np.full(len(file_indices), 'UNKNOWN', dtype=object)
    groups = np.full(len(file_indices), -1, dtype=np.int32)

    fi_to_pop = dict(
        zip(population_df['file_index'], population_df['super_population'])
    )
    fi_to_sample = dict(
        zip(population_df['file_index'], range(len(population_df)))
    )

    for i, fi in enumerate(file_indices):
        fi_int = int(fi)
        if fi_int in fi_to_pop:
            labels[i] = fi_to_pop[fi_int]
            groups[i] = fi_to_sample[fi_int]

    return labels, groups


def subtract_sample_means(
    X: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Subtract per-sample mean from each variant's activation.

    Removes sample-level bias while keeping all data points. Forces the
    classifier to rely on variant-level patterns that differ across
    superpopulations rather than sample-level averages.
    """
    X_centered = X.copy()
    for g in np.unique(groups):
        if g < 0:
            continue
        mask = groups == g
        X_centered[mask] -= X[mask].mean(axis=0)
    return X_centered


def aggregate_to_sample_means(
    X: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse to one mean vector per sample."""
    unique_groups = np.unique(groups[groups >= 0])
    X_agg, labels_agg, groups_agg = [], [], []
    for g in unique_groups:
        mask = groups == g
        X_agg.append(X[mask].mean(axis=0))
        labels_agg.append(labels[mask][0])
        groups_agg.append(g)
    return (
        np.array(X_agg),
        np.array(labels_agg),
        np.array(groups_agg, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def _reduce_dimensions(
    X_train: np.ndarray,
    X_test: np.ndarray,
    max_components: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """PCA reduction fitted on train, applied to test."""
    from sklearn.decomposition import PCA

    n_components = min(max_components, X_train.shape[0], X_train.shape[1])
    pca = PCA(n_components=n_components)
    X_train_r = pca.fit_transform(X_train)
    X_test_r = pca.transform(X_test)
    return X_train_r, X_test_r


def run_linear_probe(
    X: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    random_state: int,
    pca_components: int = 50,
    logreg_c: float = 1.0,
    class_weight_balanced: bool = False,
) -> Dict:
    """Leave-one-group-out logistic regression probe.

    Each fold holds out all data from one sample. Uses PCA to reduce
    dimensionality before fitting (fast, avoids overfitting on high-dim
    features). Uses a fixed regularization strength C=1.0 to keep runtime
    tractable for permutation testing.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import (
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
    )

    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names = list(le.classes_)

    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train, X_test = _reduce_dimensions(
            X_train, X_test, max_components=pca_components
        )

        clf = LogisticRegression(
            C=logreg_c,
            max_iter=1000,
            solver='lbfgs',
            class_weight='balanced' if class_weight_balanced else None,
            random_state=random_state,
        )
        clf.fit(X_train, y[train_idx])

        preds = clf.predict(X_test)
        y_true_all.extend(y[test_idx].tolist())
        y_pred_all.extend(preds.tolist())

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)
    report = classification_report(
        y_true_all, y_pred_all, target_names=class_names, output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true_all, y_pred_all)

    return {
        'balanced_accuracy': float(bal_acc),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'y_true': y_true_all.tolist(),
        'y_pred': y_pred_all.tolist(),
        'n_samples': len(X),
        'n_groups': len(np.unique(groups)),
        'n_features': X.shape[1],
    }


def run_mlp_probe(
    X: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    random_state: int,
    pca_components: int = 50,
    epochs: int = 80,
    mlp_hidden: Tuple[int, ...] = (128,),
    class_weight_balanced: bool = False,
) -> Dict:
    """Leave-one-group-out probe with sklearn MLPClassifier."""
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import (
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
    )
    from sklearn.utils.class_weight import compute_sample_weight

    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names = list(le.classes_)

    logo = LeaveOneGroupOut()
    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    rs_fold = random_state

    for train_idx, test_idx in logo.split(X, y, groups):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        X_train, X_test = _reduce_dimensions(
            X_train, X_test, max_components=pca_components
        )
        yt = y[train_idx]
        sw = compute_sample_weight('balanced', yt) if class_weight_balanced else None
        clf = _make_mlp_classifier(epochs, mlp_hidden, rs_fold)
        rs_fold += 7937
        clf.fit(X_train, yt, sample_weight=sw)
        preds = clf.predict(X_test)
        y_true_all.extend(y[test_idx].tolist())
        y_pred_all.extend(preds.tolist())

    y_true_all_a = np.array(y_true_all)
    y_pred_all_a = np.array(y_pred_all)
    bal_acc = balanced_accuracy_score(y_true_all_a, y_pred_all_a)
    report = classification_report(
        y_true_all_a, y_pred_all_a,
        target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true_all_a, y_pred_all_a)
    return {
        'balanced_accuracy': float(bal_acc),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'y_true': y_true_all_a.tolist(),
        'y_pred': y_pred_all_a.tolist(),
        'n_samples': len(X),
        'n_groups': len(np.unique(groups)),
        'n_features': X.shape[1],
    }


def run_probe(
    X: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    random_state: int,
    *,
    probe_kind: str,
    pca_components: int,
    logreg_c: float,
    class_weight_balanced: bool,
    epochs: int,
    mlp_hidden: Tuple[int, ...],
) -> Dict:
    if probe_kind == 'linear':
        return run_linear_probe(
            X,
            labels,
            groups,
            random_state,
            pca_components=pca_components,
            logreg_c=logreg_c,
            class_weight_balanced=class_weight_balanced,
        )
    return run_mlp_probe(
        X,
        labels,
        groups,
        random_state,
        pca_components=pca_components,
        epochs=epochs,
        mlp_hidden=mlp_hidden,
        class_weight_balanced=class_weight_balanced,
    )


def compute_majority_baseline(labels: np.ndarray) -> float:
    """Balanced accuracy of always predicting the most common class."""
    from sklearn.metrics import balanced_accuracy_score

    unique, counts = np.unique(labels, return_counts=True)
    majority = unique[np.argmax(counts)]
    preds = np.full_like(labels, majority)
    return float(balanced_accuracy_score(labels, preds))


def _shuffle_labels_by_group(
    labels: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute labels at sample/group level, then broadcast to rows."""
    permuted = labels.copy()
    valid_groups = np.unique(groups[groups >= 0])
    group_labels = np.array([labels[groups == g][0] for g in valid_groups], dtype=object)
    shuffled_group_labels = rng.permutation(group_labels)
    for g, lab in zip(valid_groups, shuffled_group_labels):
        permuted[groups == g] = lab
    return permuted


def run_permutation_generic(
    X: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_permutations: int,
    random_state: int,
    probe_fn: Callable[
        [np.ndarray, np.ndarray, np.ndarray, int], Dict
    ],
) -> Dict:
    """Group-label permutations; probe_fn must return dict with balanced_accuracy."""
    rng = np.random.default_rng(random_state)
    null_scores: list[float] = []

    for i in range(n_permutations):
        shuffled = _shuffle_labels_by_group(labels, groups, rng)
        result = probe_fn(X, shuffled, groups, i)
        null_scores.append(result['balanced_accuracy'])
        if (i + 1) % max(1, n_permutations // 5) == 0:
            print(f'  Permutation {i+1}/{n_permutations}: '
                  f'mean null acc = {np.mean(null_scores):.4f}')

    return {
        'null_scores': null_scores,
        'null_mean': float(np.mean(null_scores)),
        'null_std': float(np.std(null_scores)),
        'null_p95': float(np.percentile(null_scores, 95)),
        'null_p99': float(np.percentile(null_scores, 99)),
    }


def run_permutation_baseline(
    X: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_permutations: int,
    random_state: int,
    pca_components: int = 50,
    logreg_c: float = 1.0,
    class_weight_balanced: bool = False,
    probe_kind: str = 'linear',
    epochs: int = 80,
    mlp_hidden: Tuple[int, ...] = (128,),
) -> Dict:
    """Shuffle labels and re-run the probe to build a null distribution."""

    def _probe(
        x: np.ndarray, lab: np.ndarray, grp: np.ndarray, rs: int
    ) -> Dict:
        return run_probe(
            x,
            lab,
            grp,
            rs,
            probe_kind=probe_kind,
            pca_components=pca_components,
            logreg_c=logreg_c,
            class_weight_balanced=class_weight_balanced,
            epochs=epochs,
            mlp_hidden=mlp_hidden,
        )

    return run_permutation_generic(
        X, labels, groups, n_permutations, random_state, _probe
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(
    probe_result: Dict,
    majority_baseline: float,
    permutation_result: Dict | None,
    output_path: str,
    layer_name: str,
    centered_mode: bool,
    sample_means_mode: bool = False,
    dpi: int = 150,
    probe_kind_label: str = 'linear',
) -> None:
    """Confusion matrix + accuracy comparison bar chart."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cm = np.array(probe_result['confusion_matrix'])
    class_names = probe_result['class_names']
    bal_acc = probe_result['balanced_accuracy']

    has_permutation = permutation_result is not None
    n_plots = 2 if has_permutation else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots + 2, 5))
    if n_plots == 1:
        axes = [axes]

    # --- Confusion matrix ---
    ax = axes[0]
    n_classes = len(class_names)

    # Normalize per row (true label)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues',
                   vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(n_classes):
        for j in range(n_classes):
            count = cm[i, j]
            pct = cm_norm[i, j]
            color = 'white' if pct > 0.5 else 'black'
            ax.text(j, i, f'{count}\n({pct:.0%})', ha='center', va='center',
                    fontsize=9, color=color)

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    if centered_mode:
        mode_str = 'sample-mean-subtracted'
    elif sample_means_mode:
        mode_str = 'sample-means-only'
    else:
        mode_str = 'per-variant'
    ax.set_title(
        f'{probe_kind_label.capitalize()} probe ({mode_str})\n'
        f'{layer_name} | Balanced Acc: {bal_acc:.2%}',
        fontsize=11, fontweight='bold',
    )

    # --- Accuracy comparison ---
    if has_permutation:
        ax2 = axes[1]
        bar_labels = ['Probe', 'Majority', 'Permutation\n(mean)']
        bar_values = [
            bal_acc,
            majority_baseline,
            permutation_result['null_mean'],
        ]
        bar_colors = ['#4C72B0', '#DD8452', '#C44E52']

        bars = ax2.bar(bar_labels, bar_values, color=bar_colors, width=0.5,
                       edgecolor='white', linewidth=1.2)

        # Permutation 95th percentile line
        ax2.axhline(
            y=permutation_result['null_p95'], color='#C44E52',
            linestyle='--', linewidth=1, alpha=0.7,
            label=f'Perm. 95th pctl ({permutation_result["null_p95"]:.2%})',
        )

        for bar, val in zip(bars, bar_values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.2%}', ha='center', va='bottom', fontsize=10,
                fontweight='bold',
            )

        ax2.set_ylim(0, min(1.15, max(bar_values) + 0.15))
        ax2.set_ylabel('Balanced Accuracy')
        ax2.set_title('Probe vs. Baselines', fontsize=11, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        # p-value annotation
        null_scores = np.array(permutation_result['null_scores'])
        p_value = (np.sum(null_scores >= bal_acc) + 1) / (len(null_scores) + 1)
        ax2.text(
            0.02, 0.98,
            f'p = {p_value:.3f} (n={len(null_scores)} perms)',
            transform=ax2.transAxes, fontsize=8, va='top',
            fontstyle='italic',
        )

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved visualization to: {output_path}')


def save_metrics(
    output_path: str,
    probe_result: Dict,
    majority_baseline: float,
    permutation_result: Dict | None,
    layer_name: str,
    centered_mode: bool,
    args: argparse.Namespace,
    sample_means_mode: bool = False,
) -> None:
    """Write all metrics to a JSON file."""
    metrics_path = args.output_metrics
    if metrics_path is None:
        metrics_path = os.path.splitext(output_path)[0] + '_metrics.json'

    output = {
        'probe': {
            'balanced_accuracy': probe_result['balanced_accuracy'],
            'classification_report': probe_result['classification_report'],
            'confusion_matrix': probe_result['confusion_matrix'],
            'class_names': probe_result['class_names'],
            'n_samples': probe_result['n_samples'],
            'n_groups': probe_result['n_groups'],
            'n_features': probe_result['n_features'],
        },
        'baselines': {
            'majority_class_balanced_accuracy': majority_baseline,
        },
        'parameters': {
            'layer': layer_name,
            'subtract_sample_mean': centered_mode,
            'use_sample_means': sample_means_mode,
            'probe_kind': getattr(args, 'probe_kind', 'linear'),
            'epochs': getattr(args, 'epochs', 80),
            'mlp_hidden': getattr(args, 'mlp_hidden', '128'),
            'plot_loss_curve': getattr(args, 'plot_loss_curve', False),
            'random_state': args.random_state,
            'n_permutations': args.n_permutations,
            'pca_components': args.pca_components,
            'logreg_c': args.logreg_c,
            'class_weight_balanced': args.class_weight_balanced,
        },
    }

    if permutation_result is not None:
        null_scores = np.array(permutation_result['null_scores'])
        p_value = float(
            (np.sum(null_scores >= probe_result['balanced_accuracy']) + 1)
            / (len(null_scores) + 1)
        )
        output['baselines']['permutation'] = {
            'null_mean': permutation_result['null_mean'],
            'null_std': permutation_result['null_std'],
            'null_p95': permutation_result['null_p95'],
            'null_p99': permutation_result['null_p99'],
            'p_value': p_value,
        }

    os.makedirs(os.path.dirname(metrics_path) or '.', exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'Saved metrics to: {metrics_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    if args.plot_loss_curve and args.probe_kind != 'mlp':
        print(
            'NOTE: --plot_loss_curve applies only when --probe_kind=mlp '
            '(ignored for linear / logistic regression).',
            file=sys.stderr,
        )
    if args.layers and args.output_metrics:
        print(
            'NOTE: --output_metrics is ignored with --layers '
            '(metrics path is derived per layer).',
            file=sys.stderr,
        )

    layer_keys = resolve_layer_list(args)
    probe_layer_names = [
        x.strip() for x in layer_keys
        if x is not None and str(x).strip()
    ]
    if probe_layer_names:
        validate_layer_keys_uniform_cache(args.cache_dir, probe_layer_names)

    print('=' * 60)
    print('DeepVariant representation probe')
    print('=' * 60)

    pop_df = load_population_metadata(args.population_metadata)
    mlp_tuple = _parse_mlp_hidden(args.mlp_hidden)

    for lk in layer_keys:
        lyr = lk.strip() if lk else None

        print(f'\nLoading activations from: {args.cache_dir}')
        X, resolved_layer, file_indices, file_paths, raw_shape = (
            load_activations_with_shape(args.cache_dir, lyr)
        )
        print(f'  Loaded {X.shape[0]} variants, {X.shape[1]} features '
              f'(global avg pooled from {raw_shape})')
        print(f'  Layer key: {resolved_layer}')
        print(f'  Source files: {len(file_paths)}')

        print(f'\nJoining labels from: {args.population_metadata}')
        labels, groups = assign_labels(file_indices, pop_df)

        known_mask = labels != 'UNKNOWN'
        if known_mask.sum() == 0:
            print(
                'ERROR: No variants matched to population labels.',
                file=sys.stderr,
            )
            raise SystemExit(1)

        X = X[known_mask]
        labels = labels[known_mask]
        groups = groups[known_mask]

        unique_pops, pop_counts = np.unique(labels, return_counts=True)
        print('  Population distribution:')
        for pop, count in zip(unique_pops, pop_counts):
            print(f'    {pop}: {count} variants')

        if args.max_samples is not None and args.max_samples < X.shape[0]:
            rng = np.random.default_rng(args.random_state)
            idx = rng.choice(X.shape[0], size=args.max_samples, replace=False)
            X, labels, groups = X[idx], labels[idx], groups[idx]
            print(f'  Subsampled to {X.shape[0]} variants.')

        if args.subtract_sample_mean and args.use_sample_means:
            print(
                'ERROR: --subtract_sample_mean and --use_sample_means are '
                'mutually exclusive.',
                file=sys.stderr,
            )
            raise SystemExit(1)

        centered_mode = args.subtract_sample_mean
        sample_means_mode = args.use_sample_means

        if centered_mode:
            print('\nSubtracting per-sample mean from variant activations...')
            X = subtract_sample_means(X, groups)
            print(f'  Centered {X.shape[0]} variants across '
                  f'{len(np.unique(groups[groups >= 0]))} samples')
        elif sample_means_mode:
            print('\nAggregating to per-sample mean vectors...')
            X, labels, groups = aggregate_to_sample_means(X, labels, groups)
            print(f'  Collapsed to {X.shape[0]} sample-mean vectors')

        n_groups = len(np.unique(groups))
        print(f'  Distinct samples (LOGO groups): {n_groups} '
              f'(from {len(file_paths)} activation .npz shard(s))')
        if n_groups < 2:
            print(
                'ERROR: Need at least 2 sequenced samples for leave-one-group-out '
                f'cross-validation (found {n_groups} group(s)).',
                file=sys.stderr,
            )
            print(
                '  Each row inherits a shard index from the sorted list '
                '`activations_00000000.npz`, `activations_00000001.npz`, … → '
                'file_index 0, 1, … which must match the mapping CSV.',
                file=sys.stderr,
            )
            print(
                '\n'
                '  Only one shard (e.g. `tmp_hooked_run/activation_cache/` from a '
                'single-sample run) assigns every variant to one sample.\n'
                '  Fix: use a combined cache such as '
                '`1kg_hooked_output/activation_cache/` produced by '
                '`bash scripts/run_1kg_hooked.sh …`, '
                'or any directory containing multiple activation .npz files '
                'in the same index order as your CSV.',
                file=sys.stderr,
            )
            raise SystemExit(1)

        if args.layers is not None:
            out_png, metrics_auto = resolve_output_paths(
                args.output, resolved_layer)
        else:
            out_png = args.output
            metrics_auto = (
                args.output_metrics
                or os.path.splitext(args.output)[0] + '_metrics.json'
            )

        loop_args = argparse.Namespace(**vars(args))
        loop_args.output_metrics = metrics_auto
        # Training loss PNG exists only for MLP; keep metrics JSON honest.
        if loop_args.probe_kind != 'mlp':
            loop_args.plot_loss_curve = False

        print(f'\n{"=" * 60}')
        print(f'Probe: {args.probe_kind} ({n_groups} LOGO folds)')
        print('=' * 60)
        probe_result = run_probe(
            X,
            labels,
            groups,
            args.random_state,
            probe_kind=args.probe_kind,
            pca_components=args.pca_components,
            logreg_c=args.logreg_c,
            class_weight_balanced=args.class_weight_balanced,
            epochs=args.epochs,
            mlp_hidden=mlp_tuple,
        )
        print(f'  Balanced accuracy: {probe_result["balanced_accuracy"]:.4f}')

        print(f'\n{"=" * 60}')
        print('Computing baselines...')
        print('=' * 60)
        majority_bl = compute_majority_baseline(labels)

        permutation_result = None
        p_value = 0.0
        if args.n_permutations > 0:
            print(f'\n  Running {args.n_permutations} group-label permutations...')
            permutation_result = run_permutation_baseline(
                X,
                labels,
                groups,
                args.n_permutations,
                args.random_state,
                pca_components=args.pca_components,
                logreg_c=args.logreg_c,
                class_weight_balanced=args.class_weight_balanced,
                probe_kind=args.probe_kind,
                epochs=args.epochs,
                mlp_hidden=mlp_tuple,
            )
            ns = np.array(permutation_result['null_scores'])
            p_value = float(
                (np.sum(ns >= probe_result['balanced_accuracy']) + 1)
                / (len(ns) + 1)
            )
            print(
                f'  Permutation null: {permutation_result["null_mean"]:.4f} '
                f'+/- {permutation_result["null_std"]:.4f}'
            )
            print(f'  p-value: {p_value:.4f}')

        if args.plot_loss_curve and args.probe_kind == 'mlp':
            loss_stem = os.path.splitext(out_png)[0] + '_train_loss.png'
            mats, itr = collect_logo_training_curves_mlp(
                X,
                labels,
                groups,
                args.epochs,
                args.pca_components,
                mlp_tuple,
                args.class_weight_balanced,
                args.random_state,
            )
            plot_training_loss_curve(
                mats,
                itr,
                loss_stem,
                title=(
                    f'MLP probe training loss '
                    f'({resolved_layer}, mean LOGO folds)'
                ),
                dpi=args.dpi,
            )

        print('\nGenerating confusion matrix...')
        plot_results(
            probe_result,
            majority_bl,
            permutation_result,
            output_path=out_png,
            layer_name=resolved_layer,
            centered_mode=centered_mode,
            sample_means_mode=sample_means_mode,
            dpi=args.dpi,
            probe_kind_label=args.probe_kind,
        )

        save_metrics(
            out_png,
            probe_result,
            majority_bl,
            permutation_result,
            layer_name=resolved_layer,
            centered_mode=centered_mode,
            sample_means_mode=sample_means_mode,
            args=loop_args,
        )

        if centered_mode:
            mode_label = 'sample-mean-subtracted'
        elif sample_means_mode:
            mode_label = 'sample-means-only'
        else:
            mode_label = 'raw per-variant'

        print(f'\n{"=" * 60}')
        print('Probe complete')
        print('=' * 60)
        print(f'  Layer: {resolved_layer}')
        print(f'  Kind: {args.probe_kind}')
        print(f'  Mode: {mode_label}')
        print(f'  Balanced Accuracy: {probe_result["balanced_accuracy"]:.4f}')
        print(f'  Majority Baseline: {majority_bl:.4f}')
        if permutation_result:
            print(f'  Permutation Mean: {permutation_result["null_mean"]:.4f}')
            print(f'  p-value: {p_value:.4f}')
        print()


if __name__ == '__main__':
    main()
