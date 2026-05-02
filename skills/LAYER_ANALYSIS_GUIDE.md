# Running the Complexity Probe & Channel PCA on a New Layer

This guide walks through adapting the existing population complexity analysis to
a different DeepVariant inception layer. It assumes you already have:

- The `deepVariant` conda env set up
- Embeddings for your target layer in `data/embeddings/<sample>_<layer>/`
- The 13-sample set from `data/embeddings/` (same samples used for mixed5)

---

## Step 0 — Pick Your Layer

The model has 11 inception blocks. Here is what each one gives you:

| Layer | Spatial dims | Channels | Character |
|-------|-------------|----------|-----------|
| `mixed0`–`mixed2` | ~28×28 | 256–288 | Early texture / low-level features |
| `mixed3`–`mixed4` | ~14×14 | 768 | Mid-level, after spatial pooling |
| `mixed5`–`mixed7` | ~4×12 | 768 | **Default analysis layer** |
| `mixed8` | ~4×12 | 1280 | Transition block, richer |
| `mixed9`–`mixed10` | ~4×12 | 2048 | Deepest features before classification |

**Later layers (mixed8–mixed10) are closer to the decision boundary**, so their
representations directly encode what the model "thinks" about a variant. Earlier
layers capture more general sequence features that are shared across many contexts.

If you want to test whether the population bias is already present in low-level
features, compare an early layer (mixed3) against a late one (mixed10).

**First, confirm what you actually have on disk:**

```bash
ls data/embeddings/ | grep -o '_mixed[^/]*' | sort -u
```

Then inspect the shape of one file:

```python
import numpy as np

layer = "mixed10"   # replace with your layer
sample = "NA12878"
d = np.load(f"data/embeddings/{sample}_{layer}/activation_cache/activations_00000000.npz",
            allow_pickle=True)
print(d.files)           # e.g. ['mixed10']
print(d[layer].shape)    # (n_sites, H, W, channels)
```

Note the number of channels — you will need it in a moment (768, 1280, or 2048).

---

## Step 1 — Run the Complexity Probe

The probe is a logistic regression trained on NA12878 to separate GIAB
high-confidence (TP) sites from uncertain (UNK) sites. The weight vector defines
a "complexity axis": high score = clean callable site, low = ambiguous context.

**Copy the existing script and make three edits:**

```bash
cp scripts/population_complexity.py scripts/population_complexity_<layer>.py
```

Open the copy and change these three lines near the top:

```python
# CHANGE 1 — layer name (controls which data/embeddings/<sample>_<layer>/ is read)
LAYER = "mixed10"           # was implicitly "mixed5" in the original

# CHANGE 2 — output directory (keeps results separate per layer)
OUT_DIR = Path("results/population_complexity_mixed10")

# CHANGE 3 — embedding loading function (update the npz path and key)
def load_embeddings(sample_id: str) -> np.ndarray:
    npz_path = EMBED_BASE / f"{sample_id}_{LAYER}" / "activation_cache" / "activations_00000000.npz"
    d = np.load(npz_path, allow_pickle=True)
    key = next((k for k in d.files if LAYER in k.lower()), d.files[0])
    act = d[key]                    # (n_sites, H, W, channels)
    return act.mean(axis=(1, 2))    # → (n_sites, channels)
```

Everything else — the probe fitting, the scaler, the stats, the plots — works
unchanged because it operates on the pooled `(n_sites, channels)` matrix.

**Run it:**

```bash
conda run -n deepVariant python3 scripts/population_complexity_<layer>.py
```

Results land in `results/population_complexity_mixed10/` (or whatever you named
`OUT_DIR`). Compare `FINDINGS.md` and `stats.json` across layers to see whether
the EAS effect strengthens, weakens, or disappears as you move deeper.

> **Note on the probe AUROC sanity check:** the script refits the probe on
> NA12878 and asserts AUROC > 0.98. This should still pass for any layer that
> carries variant-relevant information — if it fails, the layer may not separate
> TP from UNK in NA12878, which is itself an interesting finding.

---

## Step 2 — Top-20 Channel PCA

After running the probe you know which 20 channels carry the most weight on the
complexity axis (highest |weight|). Projecting all samples into the space of
just those 20 channels reveals whether the population separation you see in the
full-dimensional score also appears in the embedding geometry, or whether the
score is averaging out a lot of noise.

Save the following as `scripts/top_channel_pca.py` and run it **after** the probe
analysis (it reads the probe weights from the output directory):

```python
"""
PCA on the top-20 probe-weight channels, coloured by superpopulation.
Run after population_complexity_<layer>.py so probe_weights.npy exists.

Usage:
  conda run -n deepVariant python3 scripts/top_channel_pca.py \
      --layer mixed10 \
      --results_dir results/population_complexity_mixed10
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

COLORS = {
    "AFR": "#E24B4A",
    "AMR": "#EF9F27",
    "EAS": "#1D9E75",
    "EUR": "#378ADD",
    "SAS": "#7F77DD",
}

SAMPLES = {
    "NA12878": ("EUR", "CEU"),
    "HG00731": ("EUR", "GBR"),
    "NA20502": ("EUR", "TSI"),
    "HG01985": ("AFR", "LWK"),
    "HG02922": ("AFR", "LWK"),
    "HG01048": ("AMR", "CLM"),
    "HG01197": ("AMR", "CLM"),
    "HG01565": ("AMR", "PEL"),
    "HG00759": ("EAS", "CHS"),
    "NA18939": ("EAS", "JPT"),
    "HG00864": ("EAS", "CDX"),
    "HG03009": ("SAS", "GIH"),
    "NA20847": ("SAS", "GIH"),
}


def load_embeddings(sample_id, layer, embed_base):
    npz_path = embed_base / f"{sample_id}_{layer}" / "activation_cache" / "activations_00000000.npz"
    d = np.load(npz_path, allow_pickle=True)
    key = next((k for k in d.files if layer in k.lower()), d.files[0])
    act = d[key]                    # (n_sites, H, W, channels)
    return act.mean(axis=(1, 2))    # → (n_sites, channels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", required=True, help="Layer name, e.g. mixed10")
    parser.add_argument("--results_dir", required=True,
                        help="Directory containing probe_weights.npy and scaler_*.npy")
    parser.add_argument("--embed_base", default="data/embeddings",
                        help="Root of embedding directories")
    parser.add_argument("--top_n", type=int, default=20,
                        help="Number of top channels to use (default 20)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    embed_base  = Path(args.embed_base)
    out_dir     = results_dir  # plots go alongside other results

    # ── Load probe artefacts ──────────────────────────────────────────────────
    weights      = np.load(results_dir / "probe_weights.npy")       # (channels,)
    scaler_mean  = np.load(results_dir / "scaler_mean.npy")         # (channels,)
    scaler_scale = np.load(results_dir / "scaler_scale.npy")        # (channels,)

    # Top-N channels by absolute probe weight
    top_idx = np.argsort(np.abs(weights))[-args.top_n:][::-1]
    print(f"Top {args.top_n} channel indices (by |weight|):")
    for rank, ch in enumerate(top_idx):
        print(f"  rank {rank+1:2d}: ch{ch:4d}  weight={weights[ch]:+.4f}")

    # ── Load all sample embeddings, apply scaler, extract top channels ────────
    X_top_list  = []
    labels_list = []
    popname_list = []

    for sid, (superpop, pop) in SAMPLES.items():
        npz_path = embed_base / f"{sid}_{args.layer}" / "activation_cache" / "activations_00000000.npz"
        if not npz_path.exists():
            print(f"  SKIP {sid} — {npz_path} not found")
            continue
        X = load_embeddings(sid, args.layer, embed_base)       # (n_sites, channels)
        X_sc  = (X - scaler_mean) / scaler_scale               # standardise with NA12878 scaler
        X_sub = X_sc[:, top_idx]                               # (n_sites, top_n)
        X_top_list.append(X_sub)
        labels_list.extend([superpop] * len(X))
        popname_list.extend([pop] * len(X))
        print(f"  Loaded {sid:12s}  n={len(X)}  superpop={superpop}")

    X_all    = np.vstack(X_top_list)        # (total_sites, top_n)
    labels   = np.array(labels_list)
    popnames = np.array(popname_list)

    # ── PCA on top-N channel subspace ─────────────────────────────────────────
    pca = PCA(n_components=min(3, args.top_n), random_state=42)
    pcs = pca.fit_transform(X_all)
    print(f"\nExplained variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%  "
          f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%")

    # ── Plot 1: PC1 vs PC2, coloured by superpopulation ──────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for sp in sorted(set(labels)):
        mask = labels == sp
        ax.scatter(pcs[mask, 0], pcs[mask, 1],
                   s=8, alpha=0.35, color=COLORS[sp], label=sp, rasterized=True)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"PCA — top {args.top_n} probe channels ({args.layer})\ncoloured by superpopulation")
    ax.legend(markerscale=2, fontsize=9)
    plt.tight_layout()
    out1 = out_dir / f"top{args.top_n}_channel_pca_superpop.png"
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"Saved {out1}")

    # ── Plot 2: PC1 vs PC2, coloured by population (finer grain) ─────────────
    unique_pops  = sorted(set(popnames))
    cmap_fine    = plt.cm.get_cmap("tab20", len(unique_pops))
    pop_color    = {p: cmap_fine(i) for i, p in enumerate(unique_pops)}

    fig, ax = plt.subplots(figsize=(9, 6))
    for pop in unique_pops:
        mask = popnames == pop
        ax.scatter(pcs[mask, 0], pcs[mask, 1],
                   s=8, alpha=0.35, color=pop_color[pop], label=pop, rasterized=True)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"PCA — top {args.top_n} probe channels ({args.layer})\ncoloured by population")
    ax.legend(markerscale=2, fontsize=7, ncol=2)
    plt.tight_layout()
    out2 = out_dir / f"top{args.top_n}_channel_pca_population.png"
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"Saved {out2}")

    # ── Plot 3: complexity score vs PC1 ───────────────────────────────────────
    # Recompute complexity scores (X_sc @ weights) for all samples in the same order
    scores_list = []
    for sid in SAMPLES:
        npz_path = embed_base / f"{sid}_{args.layer}" / "activation_cache" / "activations_00000000.npz"
        if not npz_path.exists():
            continue
        X = load_embeddings(sid, args.layer, embed_base)
        X_sc = (X - scaler_mean) / scaler_scale
        scores_list.append(X_sc @ weights)
    scores_all = np.concatenate(scores_list)

    fig, ax = plt.subplots(figsize=(8, 6))
    for sp in sorted(set(labels)):
        mask = labels == sp
        ax.scatter(pcs[mask, 0], scores_all[mask],
                   s=8, alpha=0.35, color=COLORS[sp], label=sp, rasterized=True)
    ax.axhline(0, color="black", lw=0.5, linestyle="--")
    ax.set_xlabel(f"PC1 of top-{args.top_n} channels ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel("Complexity score (full probe)")
    ax.set_title(f"Is the complexity axis aligned with the top-{args.top_n} channel PC1?\n({args.layer})")
    ax.legend(markerscale=2, fontsize=9)
    plt.tight_layout()
    out3 = out_dir / f"top{args.top_n}_channel_pc1_vs_score.png"
    plt.savefig(out3, dpi=150)
    plt.close()
    print(f"Saved {out3}")

    # ── Save top channel list ─────────────────────────────────────────────────
    top_channel_info = [
        {"rank": i+1, "channel": int(ch), "weight": float(weights[ch])}
        for i, ch in enumerate(top_idx)
    ]
    with open(out_dir / f"top{args.top_n}_channels.json", "w") as f:
        json.dump(top_channel_info, f, indent=2)
    print(f"Saved top{args.top_n}_channels.json")


if __name__ == "__main__":
    main()
```

**Run it:**

```bash
conda run -n deepVariant python3 scripts/top_channel_pca.py \
    --layer mixed10 \
    --results_dir results/population_complexity_mixed10
```

**What the three plots show:**

| Plot | Question it answers |
|------|---------------------|
| `top20_channel_pca_superpop.png` | Do the top probe channels geometrically separate superpopulations in 2D? |
| `top20_channel_pca_population.png` | Same but at population (CEU, LWK, CHS…) level — are within-superpop differences visible? |
| `top20_channel_pc1_vs_score.png` | Is the complexity axis aligned with the main axis of variance in the top-channel subspace? If PC1 ≈ score, the probe is capturing a geometrically dominant signal; if PC1 is orthogonal to score, the probe is picking up something subtle. |

---

## Step 3 — Compare Across Layers

Once you have results for two or more layers, compare these numbers from each
layer's `stats.json`:

- **Kruskal-Wallis p-value** — is there any significant superpop effect at all?
- **EAS vs EUR Cohen's d** — does the effect size grow toward the decision
  boundary (deeper layers) or fade?
- **AUROC on NA12878** — does the probe still find a TP/UNK axis? A drop here
  means the layer doesn't encode GIAB callability as cleanly.

A pattern of *growing* effect size in later layers would suggest the bias is
amplified as the model integrates information. A pattern of *shrinking* or
*absent* effect in earlier layers would suggest it enters at a specific depth.

---

## Troubleshooting

**`KeyError` when loading NPZ** — the key inside the file matches the layer name
used when hooking. Check with:

```python
import numpy as np
d = np.load("data/embeddings/NA12878_mixed10/activation_cache/activations_00000000.npz",
            allow_pickle=True)
print(d.files)   # tells you the actual key name
```

If the key is `"mixed10_0"` instead of `"mixed10"`, update the `load_embeddings`
function to use `d.files[0]` (it already falls back to this).

**Shape mismatch error in probe fitting** — if the layer has a different number
of channels (e.g. 2048 for mixed10 vs 768 for mixed5), the probe weights will
be a different length. This is expected and handled automatically; just make sure
you are loading `probe_weights.npy` from the *same layer's* results directory,
not the mixed5 one.

**`assert auroc > 0.98` fails** — the probe can't find a clean TP/UNK axis in
NA12878 for this layer. Try relaxing the assert to `> 0.90` and check the actual
AUROC. If it is around 0.75–0.85, the layer still has signal but the linear
probe is less powerful; results are interpretable but weaker.