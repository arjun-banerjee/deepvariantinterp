#!/usr/bin/env python3
"""UMAP embedding of DeepVariant activation cache (.npz) files.

Loads all ``activations_*.npz`` files produced by ``activation_hooks.ActivationCache``,
flattens the specified layer tensor per example, and runs UMAP (2D by default).

Dependencies::
    pip install umap-learn numpy matplotlib scikit-learn

Example::
    python plotting/umap_activations.py \\
      --cache_dir quickstart-output/activation_cache \\
      --layer mixed5 \\
      --output quickstart-output/activation_cache/umap_mixed5.png

If you have only one or very few examples, UMAP may warn or fall back; use
``--max_samples`` to subsample large caches.
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
  p.add_argument(
      '--layer',
      default=None,
      help=(
          'Layer name in the npz (e.g. mixed5). '
          'Default: first key found in the first file.'
      ),
  )
  p.add_argument(
      '--output',
      '-o',
      default='umap_activations.png',
      help='Output PNG path.',
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
      help='Random seed for UMAP.',
  )
  p.add_argument(
      '--max_samples',
      type=int,
      default=None,
      help='Optional cap on number of rows (after shuffle).',
  )
  p.add_argument(
      '--title',
      default=None,
      help='Plot title (default: layer name + n samples).',
  )
  return p.parse_args()


def _load_matrix(cache_dir: str, layer: str | None) -> tuple:
  """Returns (X, layer_names_used, file_indices_per_row).

  X rows are flattened activations; one row per example in batch order
  across sorted npz files.
  """
  import numpy as np

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

    # Expected: (batch, H, W, C) or (H, W, C) for a single example.
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


def main() -> None:
  args = _parse_args()

  try:
    import numpy as np
    import umap
  except ImportError as e:
    print(
        'Missing dependency. Install with:\n'
        '  pip install umap-learn numpy matplotlib scikit-learn',
        file=sys.stderr,
    )
    raise SystemExit(1) from e

  X, layer_used, file_idx, paths = _load_matrix(args.cache_dir, args.layer)
  n = X.shape[0]
  print(f'Loaded {n} samples, feature dim {X.shape[1]}, layer={layer_used!r}')

  if args.max_samples is not None and args.max_samples < n:
    rng = np.random.default_rng(args.random_state)
    idx = rng.choice(n, size=args.max_samples, replace=False)
    X = X[idx]
    file_idx = file_idx[idx]
    n = X.shape[0]
    print(f'Subsampled to {n} rows.')

  if n < 2:
    print('Need at least 2 samples for UMAP.', file=sys.stderr)
    raise SystemExit(1)

  n_neighbors = min(args.n_neighbors, max(2, n - 1))
  reducer = umap.UMAP(
      n_components=2,
      n_neighbors=n_neighbors,
      min_dist=args.min_dist,
      random_state=args.random_state,
      verbose=True,
  )
  embedding = reducer.fit_transform(X)

  try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
  except ImportError as e:
    print('matplotlib required for plotting:', e, file=sys.stderr)
    raise SystemExit(1) from e

  fig, ax = plt.subplots(figsize=(8, 6))
  if file_idx.max() > 0:
    sc = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=file_idx,
        cmap='tab20',
        alpha=0.85,
        s=20,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('cache file index (sorted activations_*.npz)')
  else:
    ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.85, s=40)

  title = args.title or f'UMAP: {layer_used} (n={n})'
  ax.set_title(title)
  ax.set_xlabel('UMAP-1')
  ax.set_ylabel('UMAP-2')
  fig.tight_layout()

  out = args.output
  os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
  fig.savefig(out, dpi=150)
  print(f'Wrote {out}')

  # also save embedding for downstream use
  emb_path = os.path.splitext(out)[0] + '_embedding.npz'
  np.savez(
      emb_path,
      embedding=embedding,
      layer=np.array(layer_used),
      n_neighbors=np.array(n_neighbors),
      source_files=np.array(paths),
      file_index_per_row=file_idx,
  )
  print(f'Wrote {emb_path}')


if __name__ == '__main__':
  main()
