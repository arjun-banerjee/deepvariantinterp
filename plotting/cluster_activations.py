#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Tuple

import numpy as np


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input/Output
    p.add_argument(
        '--cache_dir',
        required=True,
        help='Directory containing activations_*.npz files.',
    )
    p.add_argument(
        '--layer',
        default=None,
        help='Layer name in the npz (e.g. mixed5). Default: first key found.',
    )
    p.add_argument(
        '--output',
        '-o',
        default='clustering_analysis.png',
        help='Output PNG path for visualization.',
    )
    p.add_argument(
        '--output_metrics',
        default=None,
        help='Optional output path for metrics JSON file.',
    )

    # UMAP parameters
    p.add_argument(
        '--umap_intermediate_dim',
        type=int,
        default=20,
        help='Intermediate UMAP dimensionality for clustering (10-50 recommended).',
    )
    p.add_argument(
        '--umap_n_neighbors',
        type=int,
        default=15,
        help='UMAP n_neighbors parameter.',
    )
    p.add_argument(
        '--umap_min_dist',
        type=float,
        default=0.1,
        help='UMAP min_dist parameter.',
    )

    # HDBSCAN parameters
    p.add_argument(
        '--min_cluster_size',
        type=int,
        default=10,
        help='HDBSCAN min_cluster_size parameter.',
    )
    p.add_argument(
        '--min_samples',
        type=int,
        default=None,
        help='HDBSCAN min_samples parameter (defaults to min_cluster_size).',
    )
    p.add_argument(
        '--cluster_selection_epsilon',
        type=float,
        default=0.0,
        help='HDBSCAN cluster_selection_epsilon parameter.',
    )
    p.add_argument(
        '--cluster_selection_method',
        default='eom',
        choices=['eom', 'leaf'],
        help='HDBSCAN cluster selection method.',
    )

    # General parameters
    p.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility.',
    )
    p.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Optional cap on number of samples to analyze.',
    )
    p.add_argument(
        '--title',
        default=None,
        help='Plot title (default: auto-generated).',
    )
    p.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='Output figure DPI.',
    )
    p.add_argument(
        '--noise_color',
        default='#d3d3d3',
        help='Color for noise points (default: light grey).',
    )
    p.add_argument(
        '--noise_alpha',
        type=float,
        default=0.3,
        help='Alpha (opacity) for noise points.',
    )
    p.add_argument(
        '--population_metadata',
        default=None,
        help='Path to CSV/TSV file with sample population info (columns: sample_id, population, super_population)',
    )
    p.add_argument(
        '--color_by_population',
        action='store_true',
        help='Color points by population instead of cluster (requires --population_metadata)',
    )

    return p.parse_args()


def load_population_metadata(metadata_path: str) -> dict:
    """Load population metadata from CSV/TSV file.

    Expected columns: sample_id, population, super_population

    Returns:
        Dictionary mapping file indices to population info
    """
    import pandas as pd

    # Try to detect delimiter
    with open(metadata_path, 'r') as f:
        first_line = f.readline()
        delimiter = '\t' if '\t' in first_line else ','

    df = pd.read_csv(metadata_path, sep=delimiter)

    # Validate required columns
    required_cols = ['sample_id', 'super_population']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f'Population metadata missing required columns: {missing_cols}. '
            f'Found columns: {list(df.columns)}'
        )

    print(f'\nLoaded population metadata from: {metadata_path}')
    print(f'  Total samples: {len(df)}')
    print(f'  Super-populations: {df["super_population"].unique()}')

    # Count samples per super-population
    pop_counts = df['super_population'].value_counts()
    for pop, count in pop_counts.items():
        print(f'    {pop}: {count} samples')

    return df


def load_activation_matrix(
    cache_dir: str,
    layer: str | None
) -> Tuple[np.ndarray, str, np.ndarray, list]:
    """Load and flatten activation tensors from .npz files."""
    pattern = os.path.join(cache_dir, 'activations_*.npz')
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f'No files matching {pattern}')

    rows = []
    file_idx_per_row = []
    resolved_layer: str | None = layer

    for fi, path in enumerate(paths):
        data = np.load(path)
        keys = list(data.keys())
        if not keys:
            raise ValueError(f'Empty npz: {path}')
        if resolved_layer is None:
            resolved_layer = keys[0]
        key = resolved_layer
        if key not in data:
            raise KeyError(f'Layer {key!r} not in {path}; keys: {keys}')
        arr = np.asarray(data[key])

        # Expected: (batch, H, W, C) or (H, W, C) for a single example
        if arr.ndim == 3:
            arr = arr[np.newaxis, ...]
        elif arr.ndim != 4:
            raise ValueError(
                f'{path} key {key}: expected 3D or 4D array, got shape {arr.shape}'
            )

        b = arr.shape[0]
        flat = arr.reshape(b, -1).astype(np.float32, copy=False)
        rows.append(flat)
        file_idx_per_row.extend([fi] * b)

    X = np.vstack(rows)
    file_idx_per_row = np.array(file_idx_per_row, dtype=np.int32)
    assert resolved_layer is not None
    return X, resolved_layer, file_idx_per_row, paths


def perform_umap_reduction(
    X: np.ndarray,
    intermediate_dim: int,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform UMAP dimensionality reduction to create both an intermediate-dimensional 
    embedding for clustering and a 2D embedding for visualization."""
    try:
        import umap
    except ImportError:
        print('ERROR: umap-learn not installed. Run: pip install umap-learn',
              file=sys.stderr)
        raise SystemExit(1)

    n = X.shape[0]
    n_neighbors_capped = min(n_neighbors, max(2, n - 1))

    if verbose:
        print(f'\n{"="*60}')
        print('STEP 1: UMAP Dimensionality Reduction')
        print(f'{"="*60}')
        print(f'Input shape: {X.shape}')
        print(f'Reducing to {intermediate_dim}D for clustering...')

    # Intermediate UMAP for clustering
    reducer_intermediate = umap.UMAP(
        n_components=intermediate_dim,
        n_neighbors=n_neighbors_capped,
        min_dist=min_dist,
        random_state=random_state,
        verbose=verbose,
        metric='euclidean'
    )
    X_intermediate = reducer_intermediate.fit_transform(X)

    if verbose:
        print(f'Intermediate embedding shape: {X_intermediate.shape}')
        print(f'\nReducing to 2D for visualization...')

    # 2D UMAP for visualization
    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors_capped,
        min_dist=min_dist,
        random_state=random_state,
        verbose=verbose,
        metric='euclidean'
    )
    X_2d = reducer_2d.fit_transform(X)

    if verbose:
        print(f'2D embedding shape: {X_2d.shape}')

    return X_intermediate, X_2d


def perform_hdbscan_clustering(
    X: np.ndarray,
    min_cluster_size: int,
    min_samples: int | None,
    cluster_selection_epsilon: float,
    cluster_selection_method: str,
    verbose: bool = True
) -> Tuple[np.ndarray, object]:
    """Perform HDBSCAN clustering on reduced feature space."""
    try:
        import hdbscan
    except ImportError:
        print('ERROR: hdbscan not installed. Run: pip install hdbscan',
              file=sys.stderr)
        raise SystemExit(1)

    if min_samples is None:
        min_samples = min_cluster_size

    if verbose:
        print(f'\n{"="*60}')
        print('STEP 2: HDBSCAN Clustering')
        print(f'{"="*60}')
        print(f'Input shape: {X.shape}')
        print(f'Parameters:')
        print(f'  min_cluster_size: {min_cluster_size}')
        print(f'  min_samples: {min_samples}')
        print(f'  cluster_selection_epsilon: {cluster_selection_epsilon}')
        print(f'  cluster_selection_method: {cluster_selection_method}')

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        metric='euclidean',
        algorithm='best',
        core_dist_n_jobs=-1  # Use all cores
    )

    labels = clusterer.fit_predict(X)

    if verbose:
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        print(f'\nClustering complete:')
        print(f'  Number of clusters: {n_clusters}')
        print(f'  Number of noise points: {n_noise}')
        print(f'  Cluster sizes: {np.bincount(labels[labels >= 0])}')

    return labels, clusterer


def analyze_noise(
    labels: np.ndarray,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Analyze and separate noise points from core clusters."""
    noise_mask = labels == -1
    core_mask = ~noise_mask

    n_total = len(labels)
    n_noise = np.sum(noise_mask)
    noise_pct = 100.0 * n_noise / n_total if n_total > 0 else 0.0

    if verbose:
        print(f'\n{"="*60}')
        print('STEP 3: Noise Analysis')
        print(f'{"="*60}')
        print(f'Total points: {n_total}')
        print(f'Noise points: {n_noise} ({noise_pct:.2f}%)')
        print(f'Core cluster points: {np.sum(core_mask)} ({100-noise_pct:.2f}%)')

    return noise_mask, core_mask, noise_pct


def calculate_clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    core_mask: np.ndarray,
    verbose: bool = True
) -> dict:
    """Calculate clustering quality metrics on non-noise points."""
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    # Filter to only core cluster points
    X_core = X[core_mask]
    labels_core = labels[core_mask]

    n_clusters = len(set(labels_core))

    metrics = {}

    if verbose:
        print(f'\n{"="*60}')
        print('STEP 4: Clustering Metrics')
        print(f'{"="*60}')
        print(f'Computing metrics on {len(X_core)} core points...')
        print(f'Number of clusters: {n_clusters}')

    # Need at least 2 clusters for metrics
    if n_clusters < 2:
        if verbose:
            print('WARNING: Less than 2 clusters found. Metrics not applicable.')
        metrics['silhouette_score'] = None
        metrics['davies_bouldin_index'] = None
        metrics['dbcv_score'] = None
        return metrics

    # Silhouette Score
    try:
        sil_score = silhouette_score(X_core, labels_core, metric='euclidean')
        metrics['silhouette_score'] = float(sil_score)
        if verbose:
            print(f'Silhouette Score: {sil_score:.4f}')
    except Exception as e:
        if verbose:
            print(f'WARNING: Could not compute Silhouette Score: {e}')
        metrics['silhouette_score'] = None

    # Davies-Bouldin Index
    try:
        db_score = davies_bouldin_score(X_core, labels_core)
        metrics['davies_bouldin_index'] = float(db_score)
        if verbose:
            print(f'Davies-Bouldin Index: {db_score:.4f}')
    except Exception as e:
        if verbose:
            print(f'WARNING: Could not compute Davies-Bouldin Index: {e}')
        metrics['davies_bouldin_index'] = None

    # DBCV (if hdbscan provides it)
    try:
        from hdbscan import validity_index
        dbcv_score = validity_index(X_core, labels_core, metric='euclidean')
        metrics['dbcv_score'] = float(dbcv_score)
        if verbose:
            print(f'DBCV Score: {dbcv_score:.4f}')
    except ImportError:
        if verbose:
            print('DBCV: Not available (requires hdbscan with validity_index)')
        metrics['dbcv_score'] = None
    except Exception as e:
        if verbose:
            print(f'WARNING: Could not compute DBCV: {e}')
        metrics['dbcv_score'] = None

    return metrics


def create_cluster_visualization(
    X_2d: np.ndarray,
    labels: np.ndarray,
    noise_mask: np.ndarray,
    output_path: str,
    title: str | None = None,
    noise_color: str = '#d3d3d3',
    noise_alpha: float = 0.3,
    dpi: int = 150,
    verbose: bool = True,
    population_labels: np.ndarray | None = None,
    color_by_population: bool = False
) -> None:
    """Create scatter plot visualization with noise in background."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        print('ERROR: matplotlib not installed. Run: pip install matplotlib',
              file=sys.stderr)
        raise SystemExit(1)

    if verbose:
        print(f'\n{"="*60}')
        print('STEP 5: Visualization')
        print(f'{"="*60}')

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(noise_mask)

    fig, ax = plt.subplots(figsize=(12, 9))

    # Define super-population colors
    pop_colors = {
        'AFR': '#FF6B6B',  # African - Red
        'AMR': '#4ECDC4',  # American - Teal
        'EAS': '#FFE66D',  # East Asian - Yellow
        'EUR': '#95E1D3',  # European - Light green
        'SAS': '#AA96DA',  # South Asian - Purple
    }

    # Plot noise points FIRST (so they're in background)
    if n_noise > 0:
        noise_points = X_2d[noise_mask]
        ax.scatter(
            noise_points[:, 0],
            noise_points[:, 1],
            c=noise_color,
            alpha=noise_alpha,
            s=20,
            label=f'Noise (n={n_noise})',
            edgecolors='none',
            rasterized=True
        )

    # Plot clustered points on top
    if color_by_population and population_labels is not None:
        if verbose:
            print('Coloring points by super-population')

        core_points = X_2d[~noise_mask]
        core_pops = population_labels[~noise_mask]
        core_clusters = labels[~noise_mask]

        unique_pops = sorted(set(core_pops))
        for pop in unique_pops:
            pop_mask = core_pops == pop
            points = core_points[pop_mask]
            color = pop_colors.get(pop, '#999999')

            ax.scatter(
                points[:, 0],
                points[:, 1],
                c=color,
                alpha=0.7,
                s=30,
                label=f'{pop} (n={len(points)})',
                edgecolors='white',
                linewidths=0.5,
                rasterized=True
            )

        if verbose:
            print('\nCluster composition by super-population:')
            for cluster_id in sorted(set(core_clusters)):
                cluster_mask = core_clusters == cluster_id
                cluster_pops = core_pops[cluster_mask]
                pop_counts = {}
                for pop in cluster_pops:
                    pop_counts[pop] = pop_counts.get(pop, 0) + 1
                print(f'  Cluster {cluster_id}:')
                for pop, count in sorted(pop_counts.items()):
                    pct = 100.0 * count / len(cluster_pops)
                    print(f'    {pop}: {count} ({pct:.1f}%)')

    elif n_clusters > 0:
        if n_clusters <= 10:
            cmap = plt.cm.tab10
        elif n_clusters <= 20:
            cmap = plt.cm.tab20
        else:
            cmap = plt.cm.nipy_spectral

        cluster_labels = labels[~noise_mask]
        cluster_points = X_2d[~noise_mask]

        unique_labels = sorted(set(cluster_labels))
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            points = cluster_points[mask]
            color = cmap(i / max(len(unique_labels) - 1, 1))
            ax.scatter(
                points[:, 0],
                points[:, 1],
                c=[color],
                alpha=0.7,
                s=30,
                label=f'Cluster {label} (n={len(points)})',
                edgecolors='white',
                linewidths=0.5,
                rasterized=True
            )

    # Formatting
    if title is None:
        if color_by_population and population_labels is not None:
            title = f'Population Distribution: {len(set(population_labels[~noise_mask]))} populations, {n_clusters} clusters'
        else:
            title = f'HDBSCAN Clustering: {n_clusters} clusters, {n_noise} noise points'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP-1', fontsize=12)
    ax.set_ylabel('UMAP-2', fontsize=12)

    ax.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        frameon=True,
        fancybox=True,
        shadow=True
    )

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f'Saved visualization to: {output_path}')


def save_results(
    output_path: str,
    embedding_2d: np.ndarray,
    embedding_intermediate: np.ndarray,
    labels: np.ndarray,
    metrics: dict,
    layer_name: str,
    file_paths: list,
    file_indices: np.ndarray,
    args: argparse.Namespace,
    verbose: bool = True,
    population_labels: np.ndarray | None = None
) -> None:
    """Save analysis results to .npz and optional JSON files."""
    import json

    # Save embeddings and labels
    npz_path = os.path.splitext(output_path)[0] + '_results.npz'
    save_data = {
        'embedding_2d': embedding_2d,
        'embedding_intermediate': embedding_intermediate,
        'cluster_labels': labels,
        'layer': np.array(layer_name),
        'source_files': np.array(file_paths),
        'file_index_per_row': file_indices,
    }
    if population_labels is not None:
        save_data['population_labels'] = population_labels

    np.savez_compressed(npz_path, **save_data)
    if verbose:
        print(f'\nSaved embeddings and labels to: {npz_path}')

    # Save metrics to JSON
    metrics_path = args.output_metrics
    if metrics_path is None:
        metrics_path = os.path.splitext(output_path)[0] + '_metrics.json'

    metrics_output = {
        'clustering_metrics': metrics,
        'summary': {
            'n_samples': int(len(labels)),
            'n_clusters': int(len(set(labels)) - (1 if -1 in labels else 0)),
            'n_noise': int(np.sum(labels == -1)),
            'noise_percentage': float(100.0 * np.sum(labels == -1) / len(labels)),
        },
        'parameters': {
            'layer': layer_name,
            'umap_intermediate_dim': args.umap_intermediate_dim,
            'umap_n_neighbors': args.umap_n_neighbors,
            'umap_min_dist': args.umap_min_dist,
            'min_cluster_size': args.min_cluster_size,
            'min_samples': args.min_samples or args.min_cluster_size,
            'cluster_selection_method': args.cluster_selection_method,
            'random_state': args.random_state,
        }
    }

    os.makedirs(os.path.dirname(metrics_path) or '.', exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics_output, f, indent=2)

    if verbose:
        print(f'Saved metrics to: {metrics_path}')


def main() -> None:
    """Main clustering analysis pipeline."""
    args = _parse_args()

    print('='*60)
    print('DeepVariant Activation Clustering Analysis')
    print('Experiment 3: Embedding Clustering Pipeline')
    print('='*60)

    # Load population metadata if provided
    population_df = None
    population_labels = None
    if args.population_metadata:
        population_df = load_population_metadata(args.population_metadata)

    # Load data
    print(f'\nLoading activations from: {args.cache_dir}')
    X, layer_name, file_indices, file_paths = load_activation_matrix(
        args.cache_dir, args.layer
    )
    print(f'Loaded {X.shape[0]} samples, feature dim {X.shape[1]}, layer={layer_name!r}')

    # Map file paths to sample IDs and then to populations
    if population_df is not None:
        import pandas as pd
        population_labels = np.array(['UNKNOWN'] * len(file_indices))

        # Extract sample IDs from file paths
        for idx, (file_idx, path) in enumerate(zip(file_indices, [file_paths[fi] for fi in file_indices])):
            # Extract sample ID from filename
            basename = os.path.basename(path)
            # Assume format like activations_SAMPLEID.npz or similar
            # You may need to adjust this parsing logic based on your file naming
            sample_id = basename.replace('activations_', '').replace('.npz', '').split('_')[0]

            # Look up population for this sample
            sample_pop = population_df[population_df['sample_id'] == sample_id]
            if not sample_pop.empty:
                population_labels[idx] = sample_pop.iloc[0]['super_population']

        print(f'\nMatched {np.sum(population_labels != "UNKNOWN")} samples to population metadata')
        unique_pops, pop_counts = np.unique(population_labels, return_counts=True)
        print('Population distribution in activations:')
        for pop, count in zip(unique_pops, pop_counts):
            print(f'  {pop}: {count} samples')

    # Optional subsampling
    if args.max_samples is not None and args.max_samples < X.shape[0]:
        rng = np.random.default_rng(args.random_state)
        idx = rng.choice(X.shape[0], size=args.max_samples, replace=False)
        X = X[idx]
        file_indices = file_indices[idx]
        if population_labels is not None:
            population_labels = population_labels[idx]
        print(f'Subsampled to {X.shape[0]} samples.')

    if X.shape[0] < args.min_cluster_size:
        print(
            f'ERROR: Need at least {args.min_cluster_size} samples '
            f'(have {X.shape[0]}). Reduce --min_cluster_size.',
            file=sys.stderr
        )
        raise SystemExit(1)

    # Step 1: UMAP reduction
    X_intermediate, X_2d = perform_umap_reduction(
        X,
        intermediate_dim=args.umap_intermediate_dim,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        random_state=args.random_state,
        verbose=True
    )

    # Step 2: HDBSCAN clustering
    labels, clusterer = perform_hdbscan_clustering(
        X_intermediate,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
        cluster_selection_method=args.cluster_selection_method,
        verbose=True
    )

    # Step 3: Noise analysis
    noise_mask, core_mask, noise_pct = analyze_noise(labels, verbose=True)

    # Step 4: Metrics
    metrics = calculate_clustering_metrics(
        X_intermediate, labels, core_mask, verbose=True
    )

    # Step 5: Visualization
    create_cluster_visualization(
        X_2d,
        labels,
        noise_mask,
        output_path=args.output,
        title=args.title,
        noise_color=args.noise_color,
        noise_alpha=args.noise_alpha,
        dpi=args.dpi,
        verbose=True,
        population_labels=population_labels,
        color_by_population=args.color_by_population
    )

    # Save results
    save_results(
        args.output,
        embedding_2d=X_2d,
        embedding_intermediate=X_intermediate,
        labels=labels,
        metrics=metrics,
        layer_name=layer_name,
        file_paths=file_paths,
        file_indices=file_indices,
        args=args,
        verbose=True,
        population_labels=population_labels
    )

    # Final summary
    print(f'\n{"="*60}')
    print('Analysis Complete!')
    print(f'{"="*60}')
    print(f'\nResults:')
    print(f'  Visualization: {args.output}')
    print(f'  Embeddings: {os.path.splitext(args.output)[0]}_results.npz')
    print(f'  Metrics: {args.output_metrics or os.path.splitext(args.output)[0] + "_metrics.json"}')
    print(f'\nClustering Summary:')
    print(f'  Clusters: {len(set(labels)) - (1 if -1 in labels else 0)}')
    print(f'  Noise: {np.sum(labels == -1)} ({noise_pct:.2f}%)')
    if metrics['silhouette_score'] is not None:
        print(f'  Silhouette Score: {metrics["silhouette_score"]:.4f}')
    if metrics['davies_bouldin_index'] is not None:
        print(f'  Davies-Bouldin Index: {metrics["davies_bouldin_index"]:.4f}')
    if metrics['dbcv_score'] is not None:
        print(f'  DBCV Score: {metrics["dbcv_score"]:.4f}')
    print()


if __name__ == '__main__':
    main()
