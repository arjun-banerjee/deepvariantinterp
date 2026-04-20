#!/usr/bin/env python3
"""Linear probe for superpopulation classification from DeepVariant activations.

Trains logistic regression on intermediate representations (e.g. mixed5) to
quantify how much superpopulation information is linearly decodable. Uses
leave-one-out cross-validation at the sample level to avoid data leakage.

Usage:
  python plotting/linear_probe.py \
    --cache_dir /path/to/activation_cache \
    --population_metadata data/1kg_file_mapping.csv \
    --layer mixed5 \
    --output linear_probe_results.png \
    --aggregate_per_sample \
    --n_permutations 100
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

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
            'and super_population columns (e.g. data/1kg_file_mapping.csv).'
        ),
    )
    p.add_argument(
        '--layer',
        default=None,
        help='Layer name key in the npz files. Default: first key found.',
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

    return p.parse_args()


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
                f'Layer {resolved_layer!r} not in {path}; available: {keys}'
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


def run_permutation_baseline(
    X: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_permutations: int,
    random_state: int,
    pca_components: int = 50,
    logreg_c: float = 1.0,
    class_weight_balanced: bool = False,
) -> Dict:
    """Shuffle labels and re-run the probe to build a null distribution."""
    rng = np.random.default_rng(random_state)
    null_scores: list[float] = []

    for i in range(n_permutations):
        shuffled = _shuffle_labels_by_group(labels, groups, rng)
        result = run_linear_probe(
            X,
            shuffled,
            groups,
            random_state=i,
            pca_components=pca_components,
            logreg_c=logreg_c,
            class_weight_balanced=class_weight_balanced,
        )
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
        f'Linear Probe ({mode_str})\n'
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

    print('=' * 60)
    print('DeepVariant Linear Probe: Superpopulation Classification')
    print('=' * 60)

    # -- Load activations --
    print(f'\nLoading activations from: {args.cache_dir}')
    X, layer_name, file_indices, file_paths, raw_shape = (
        load_activations_with_shape(args.cache_dir, args.layer)
    )
    print(f'  Loaded {X.shape[0]} variants, {X.shape[1]} features '
          f'(global avg pooled from {raw_shape})')
    print(f'  Layer: {layer_name}')
    print(f'  Source files: {len(file_paths)}')

    # -- Load labels --
    print(f'\nLoading population metadata from: {args.population_metadata}')
    pop_df = load_population_metadata(args.population_metadata)
    labels, groups = assign_labels(file_indices, pop_df)

    known_mask = labels != 'UNKNOWN'
    n_known = known_mask.sum()
    print(f'  Matched {n_known}/{len(labels)} variants to population labels')

    if n_known == 0:
        print('ERROR: No variants matched to population labels.', file=sys.stderr)
        raise SystemExit(1)

    X = X[known_mask]
    labels = labels[known_mask]
    groups = groups[known_mask]

    unique_pops, pop_counts = np.unique(labels, return_counts=True)
    print('  Population distribution:')
    for pop, count in zip(unique_pops, pop_counts):
        print(f'    {pop}: {count} variants')

    # -- Optional subsampling --
    if args.max_samples is not None and args.max_samples < X.shape[0]:
        rng = np.random.default_rng(args.random_state)
        idx = rng.choice(X.shape[0], size=args.max_samples, replace=False)
        X, labels, groups = X[idx], labels[idx], groups[idx]
        print(f'  Subsampled to {X.shape[0]} variants.')

    # -- Optional preprocessing modes --
    if args.subtract_sample_mean and args.use_sample_means:
        print('ERROR: --subtract_sample_mean and --use_sample_means are '
              'mutually exclusive.', file=sys.stderr)
        raise SystemExit(1)

    centered_mode = args.subtract_sample_mean
    sample_means_mode = args.use_sample_means

    if centered_mode:
        print(f'\nSubtracting per-sample mean from variant activations...')
        X = subtract_sample_means(X, groups)
        print(f'  Centered {X.shape[0]} variants across '
              f'{len(np.unique(groups[groups >= 0]))} samples')
    elif sample_means_mode:
        print(f'\nAggregating to per-sample mean vectors...')
        X, labels, groups = aggregate_to_sample_means(X, labels, groups)
        print(f'  Collapsed to {X.shape[0]} sample-mean vectors')

    n_groups = len(np.unique(groups))
    if n_groups < 2:
        print('ERROR: Need at least 2 samples for cross-validation.',
              file=sys.stderr)
        raise SystemExit(1)

    # -- Run probe --
    print(f'\n{"="*60}')
    print(f'Running LOOCV linear probe ({n_groups} folds, 1 per sample)...')
    print(f'{"="*60}')
    probe_result = run_linear_probe(
        X,
        labels,
        groups,
        args.random_state,
        pca_components=args.pca_components,
        logreg_c=args.logreg_c,
        class_weight_balanced=args.class_weight_balanced,
    )
    print(f'  Balanced accuracy: {probe_result["balanced_accuracy"]:.4f}')

    # -- Baselines --
    print(f'\n{"="*60}')
    print('Computing baselines...')
    print(f'{"="*60}')

    majority_bl = compute_majority_baseline(labels)
    print(f'  Majority class baseline: {majority_bl:.4f}')

    permutation_result = None
    if args.n_permutations > 0:
        print(f'\n  Running {args.n_permutations} permutations...')
        permutation_result = run_permutation_baseline(
            X,
            labels,
            groups,
            args.n_permutations,
            args.random_state,
            pca_components=args.pca_components,
            logreg_c=args.logreg_c,
            class_weight_balanced=args.class_weight_balanced,
        )
        null_scores = np.array(permutation_result['null_scores'])
        p_value = (
            (np.sum(null_scores >= probe_result['balanced_accuracy']) + 1)
            / (len(null_scores) + 1)
        )
        print(f'  Permutation null: {permutation_result["null_mean"]:.4f} '
              f'+/- {permutation_result["null_std"]:.4f}')
        print(f'  p-value: {p_value:.4f}')

    # -- Visualize --
    print(f'\n{"="*60}')
    print('Generating plots...')
    print(f'{"="*60}')
    plot_results(
        probe_result,
        majority_bl,
        permutation_result,
        output_path=args.output,
        layer_name=layer_name,
        centered_mode=centered_mode,
        sample_means_mode=sample_means_mode,
        dpi=args.dpi,
    )

    # -- Save metrics --
    save_metrics(
        args.output,
        probe_result,
        majority_bl,
        permutation_result,
        layer_name=layer_name,
        centered_mode=centered_mode,
        sample_means_mode=sample_means_mode,
        args=args,
    )

    # -- Summary --
    if centered_mode:
        mode_label = 'sample-mean-subtracted'
    elif sample_means_mode:
        mode_label = 'sample-means-only'
    else:
        mode_label = 'raw per-variant'
    print(f'\n{"="*60}')
    print('Linear Probe Complete!')
    print(f'{"="*60}')
    print(f'  Layer: {layer_name}')
    print(f'  Mode: {mode_label}')
    print(f'  Balanced Accuracy: {probe_result["balanced_accuracy"]:.4f}')
    print(f'  Majority Baseline: {majority_bl:.4f}')
    if permutation_result:
        print(f'  Permutation Mean:  {permutation_result["null_mean"]:.4f}')
        print(f'  p-value:           {p_value:.4f}')
    print()


if __name__ == '__main__':
    main()
