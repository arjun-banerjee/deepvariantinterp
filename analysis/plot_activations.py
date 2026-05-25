#!/usr/bin/env python3
"""PCA and UMAP plots of mixed5 activations — unlabeled clusters."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

# ── Load activations ──────────────────────────────────────────────────────────
data = np.load('output/activations/activations_00000000.npz')
acts = data['mixed5']          # (24, 4, 12, 768)
print(f'Loaded: {acts.shape}')

# Global average pool over spatial dims → (24, 768)
X = acts.mean(axis=(1, 2))
print(f'After global avg pool: {X.shape}')

# Standardize
X = StandardScaler().fit_transform(X)

# ── PCA ───────────────────────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
var_explained = pca.explained_variance_ratio_ * 100

# ── UMAP ──────────────────────────────────────────────────────────────────────
reducer = umap.UMAP(n_components=2, random_state=42,
                    n_neighbors=min(10, len(X)-1), min_dist=0.3)
X_umap = reducer.fit_transform(X)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('DeepVariant InceptionV3 — mixed5 activations', fontsize=13)

scatter_kw = dict(s=60, alpha=0.85, edgecolors='white', linewidths=0.4)

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], **scatter_kw)
axes[0].set_title('PCA', fontsize=11)
axes[0].set_xlabel(f'PC1 ({var_explained[0]:.1f}%)')
axes[0].set_ylabel(f'PC2 ({var_explained[1]:.1f}%)')

axes[1].scatter(X_umap[:, 0], X_umap[:, 1], **scatter_kw)
axes[1].set_title('UMAP', fontsize=11)
axes[1].set_xlabel('UMAP-1')
axes[1].set_ylabel('UMAP-2')

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
out = 'output/activations/pca_umap_mixed5.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
plt.show()
