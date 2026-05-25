#!/usr/bin/env python3
"""UMAP embedding of DeepVariant activation cache (.npz) files.

Loads all ``activations_*.npz`` files produced by ``activation_hooks.ActivationCache``,
flattens the specified layer tensor (preserving spatial + channel information), optionally
reduces with PCA, clusters with KMeans, then runs UMAP (2D).

Can be run per-layer (--layer mixed5) or across multiple layers in sequence (--layers
mixed0,mixed5,mixed10), saving a separate embedding NPZ and plot per layer.

Dependencies::
    pip install umap-learn numpy matplotlib scikit-learn

Example::
    # Single layer
    python plotting/umap_activations.py \\
      --cache_dir output/activations \\
      --layer mixed5 \\
      --output output/activations/umap_mixed5.png

    # All 11 layers sequentially
    python plotting/umap_activations.py \\
      --cache_dir output/activations \\
      --layers mixed0,mixed1,mixed2,mixed3,mixed4,mixed5,mixed6,mixed7,mixed8,mixed9,mixed10 \\
      --output_dir output/activations
"""

from __future__ import annotations

import argparse
import glob
import os
import sys


def _parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument(
      '--cache_dir',
      required=True,
      help='Directory containing activations_*.npz files.',
  )

  layer_group = p.add_mutually_exclusive_group()
  layer_group.add_argument(
      '--layer',
      default=None,
      help='Single layer name to process (e.g. mixed5).',
  )
  layer_group.add_argument(
      '--layers',
      default=None,
      help=(
          'Comma-separated list of layer names to process sequentially '
          '(e.g. mixed0,mixed5,mixed10). Each produces its own output files.'
      ),
  )

  p.add_argument(
      '--output',
      '-o',
      default=None,
      help='Output PNG path (single-layer mode). Defaults to cache_dir/umap_{layer}.png.',
  )
  p.add_argument(
      '--output_dir',
      default=None,
      help='Output directory for multi-layer mode. Defaults to cache_dir.',
  )
  p.add_argument(
      '--n_neighbors',
      type=int,
      default=15,
      help='UMAP n_neighbors (capped automatically for small n).',
  )
  p.add_argument(
      '--min_dist',
      type=float,
      default=0.1,
      help='UMAP min_dist.',
  )
  p.add_argument(
      '--random_state',
      type=int,
      default=42,
      help='Random seed.',
  )
  p.add_argument(
      '--max_samples',
      type=int,
      default=None,
      help='Optional cap on number of rows (random subsample).',
  )
  p.add_argument(
      '--pca_components',
      type=int,
      default=200,
      help=(
          'Reduce to this many PCA dims before UMAP. '
          'Set to 0 to run UMAP directly on flattened features (slow for large N).'
      ),
  )
  p.add_argument(
      '--n_clusters',
      type=int,
      default=15,
      help='KMeans clusters in PCA space (0 to skip clustering).',
  )
  p.add_argument(
      '--title',
      default=None,
      help='Plot title override.',
  )
  return p.parse_args()


def _load_matrix(
    cache_dir: str,
    layer: str | None,
    max_samples: int | None = None,
    random_state: int = 42,
) -> tuple:
  """Load and flatten activations for one layer from all NPZ files.

  Flattens (B, H, W, C) → (B, H*W*C) preserving spatial and channel information.
  If max_samples is set, randomly selects that many files to load (files are
  shuffled before selection so the subsample is spatially random) — this avoids
  loading the full dataset into memory before subsampling.
  Returns (X, layer_used, file_indices, paths).
  """
  import numpy as np

  pattern = os.path.join(cache_dir, 'activations_*.npz')
  all_paths = sorted(glob.glob(pattern))
  if not all_paths:
    raise FileNotFoundError(f'No files matching {pattern}')

  # If subsampling, randomly select a subset of files so we never load more
  # data than needed (avoids OOM on large datasets with high-dim layers).
  if max_samples is not None:
    rng = np.random.default_rng(random_state)
    # Peek at first file to get batch size, then compute how many files needed
    probe = np.load(all_paths[0])
    probe_keys = list(probe.keys())
    probe_layer = layer if layer is not None else probe_keys[0]
    batch_size = np.asarray(probe[probe_layer]).shape[0]
    n_files_needed = min(len(all_paths), int(np.ceil(max_samples / batch_size)))
    chosen_indices = sorted(rng.choice(len(all_paths), size=n_files_needed, replace=False))
    paths = [all_paths[i] for i in chosen_indices]
    print(f'  Subsampling: loading {n_files_needed}/{len(all_paths)} files '
          f'(~{n_files_needed * batch_size} rows) for max_samples={max_samples}')
  else:
    paths = all_paths

  rows = []
  file_idx_per_row = []
  global_indices = []   # global_idx matching variants_metadata.csv
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
      raise KeyError(f'Layer {key!r} not in {path}; available: {keys}')
    arr = np.asarray(data[key])

    # Handle (H, W, C) single-example case
    if arr.ndim == 3:
      arr = arr[np.newaxis, ...]
    elif arr.ndim != 4:
      raise ValueError(
          f'{path} key {key}: expected 3D or 4D array, got shape {arr.shape}'
      )

    b = arr.shape[0]
    # Flatten: (B, H, W, C) → (B, H*W*C) — no information loss
    flat = arr.reshape(b, -1).astype(np.float32, copy=False)
    rows.append(flat)
    file_idx_per_row.extend([fi] * b)

    # Derive global_idx from filename: activations_00000XXX.npz → npz_file_idx=XXX
    # global_idx = npz_file_idx * batch_size + example_idx_within_file
    npz_file_idx = int(os.path.basename(path).replace('activations_', '').replace('.npz', ''))
    global_indices.extend(range(npz_file_idx * b, npz_file_idx * b + b))

  X = np.vstack(rows)
  file_idx_per_row = np.array(file_idx_per_row, dtype=np.int32)
  global_indices = np.array(global_indices, dtype=np.int64)
  assert resolved_layer is not None
  return X, resolved_layer, file_idx_per_row, paths, global_indices


def _run_single_layer(
    cache_dir: str,
    layer: str | None,
    output_png: str,
    args: argparse.Namespace,
) -> None:
  """Full pipeline for one layer: load → flatten → scale → PCA → KMeans → UMAP → save."""
  import numpy as np
  import umap as umap_lib
  from sklearn.cluster import MiniBatchKMeans
  from sklearn.decomposition import PCA
  from sklearn.preprocessing import StandardScaler

  X, layer_used, file_idx, paths, global_indices = _load_matrix(
      cache_dir, layer,
      max_samples=args.max_samples,
      random_state=args.random_state,
  )
  n = X.shape[0]
  print(f'\n[{layer_used}] {n} samples, flat dim={X.shape[1]}')

  if n < 2:
    print(f'[{layer_used}] Need ≥2 samples. Skipping.', file=sys.stderr)
    return

  # Standardize
  print(f'[{layer_used}] StandardScaler ...')
  X = StandardScaler().fit_transform(X)

  # PCA
  if args.pca_components > 0 and X.shape[1] > args.pca_components:
    n_components = min(args.pca_components, n - 1, X.shape[1])
    print(f'[{layer_used}] PCA: {X.shape[1]} → {n_components} dims ...')
    pca = PCA(n_components=n_components, random_state=args.random_state)
    X = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_.sum() * 100
    print(f'[{layer_used}]   Variance explained: {var_explained:.1f}%')

  # KMeans in PCA space (before UMAP — UMAP distorts inter-cluster distances)
  cluster_labels = None
  if args.n_clusters > 0:
    n_clusters = min(args.n_clusters, n)
    print(f'[{layer_used}] KMeans k={n_clusters} ...')
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=args.random_state,
        n_init=5,
        batch_size=min(10_000, n),
    )
    cluster_labels = kmeans.fit_predict(X)
    print(f'[{layer_used}]   Inertia: {kmeans.inertia_:.1f}')
    from collections import Counter
    dist = Counter(cluster_labels.tolist())
    for k in sorted(dist):
      print(f'[{layer_used}]   cluster {k:2d}: {dist[k]:6d} samples')

  # UMAP (visualization only)
  n_neighbors = min(args.n_neighbors, max(2, n - 1))
  print(f'[{layer_used}] UMAP n_neighbors={n_neighbors} ...')
  reducer = umap_lib.UMAP(
      n_components=2,
      n_neighbors=n_neighbors,
      min_dist=args.min_dist,
      random_state=args.random_state,
      verbose=True,
  )
  embedding = reducer.fit_transform(X)

  # Plot
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots(figsize=(8, 6))
  if cluster_labels is not None:
    sc = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=cluster_labels, cmap='tab20', alpha=0.7, s=10, rasterized=True,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('KMeans cluster (PCA space)')
  elif file_idx.max() > 0:
    sc = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=file_idx, cmap='tab20', alpha=0.7, s=10, rasterized=True,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('NPZ file index')
  else:
    ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, s=10, rasterized=True)

  title = args.title or f'UMAP: {layer_used} (n={n})'
  ax.set_title(title)
  ax.set_xlabel('UMAP-1')
  ax.set_ylabel('UMAP-2')
  fig.tight_layout()

  os.makedirs(os.path.dirname(output_png) or '.', exist_ok=True)
  fig.savefig(output_png, dpi=150)
  plt.close(fig)
  print(f'[{layer_used}] Wrote {output_png}')

  # Save embedding NPZ
  emb_path = os.path.splitext(output_png)[0] + '_embedding.npz'
  save_kwargs = dict(
      embedding=embedding,
      layer=np.array(layer_used),
      n_neighbors=np.array(n_neighbors),
      source_files=np.array(paths),
      file_index_per_row=file_idx,
      global_indices=global_indices,  # maps each row → global_idx in variants_metadata.csv
  )
  if cluster_labels is not None:
    save_kwargs['cluster_labels'] = cluster_labels
  np.savez(emb_path, **save_kwargs)
  print(f'[{layer_used}] Wrote {emb_path}')


def main() -> None:
  args = _parse_args()

  try:
    import numpy as np
    import umap  # noqa: F401
    from sklearn.cluster import MiniBatchKMeans  # noqa: F401
    from sklearn.decomposition import PCA  # noqa: F401
    from sklearn.preprocessing import StandardScaler  # noqa: F401
  except ImportError as e:
    print(
        'Missing dependency:\n  pip install umap-learn numpy matplotlib scikit-learn',
        file=sys.stderr,
    )
    raise SystemExit(1) from e

  # Resolve layer list
  if args.layers:
    layer_list = [l.strip() for l in args.layers.split(',') if l.strip()]
  elif args.layer:
    layer_list = [args.layer]
  else:
    layer_list = [None]  # auto-detect from first NPZ key

  output_dir = args.output_dir or args.cache_dir

  for layer in layer_list:
    if args.output and len(layer_list) == 1:
      output_png = args.output
    else:
      layer_tag = layer if layer else 'auto'
      output_png = os.path.join(output_dir, f'umap_{layer_tag}.png')

    _run_single_layer(args.cache_dir, layer, output_png, args)


if __name__ == '__main__':
  main()
