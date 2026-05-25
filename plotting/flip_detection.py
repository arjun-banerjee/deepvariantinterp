"""Statistical flip detection using k-NN neighborhood analysis in activation space.

Detects meaningful flips by comparing k-NN composition across layers.
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


@dataclass
class FlipResult:
    """Result of flip detection analysis."""
    eid: int
    layer_from: str
    layer_to: str
    jaccard: float
    tp_change: float
    fp_change: float
    fn_change: float
    chi2_pval: float
    is_robust: bool
    is_real_flip: bool
    confidence: float  # 0.0-1.0


def compute_knn_set(
    variant_idx: int,
    activations: np.ndarray,
    k: int = 20,
) -> set[int]:
    """Compute k-NN set for a variant in activation space.

    Args:
        variant_idx: Index of variant
        activations: shape (n_variants, d) activation array
        k: number of neighbors

    Returns:
        set of k nearest neighbor indices (excluding self)
    """
    variant_embedding = activations[variant_idx]
    distances = np.linalg.norm(activations - variant_embedding, axis=1)
    knn_indices = np.argsort(distances)[1:k+1]  # exclude self
    return set(knn_indices)


def compute_jaccard_similarity(
    variant_idx: int,
    activations_l: np.ndarray,
    activations_l1: np.ndarray,
    k: int = 20,
) -> float:
    """Jaccard similarity between k-NN sets across two layers."""
    knn_l = compute_knn_set(variant_idx, activations_l, k=k)
    knn_l1 = compute_knn_set(variant_idx, activations_l1, k=k)

    intersection = len(knn_l & knn_l1)
    union = len(knn_l | knn_l1)

    return intersection / union if union > 0 else 0.0


def get_neighborhood_composition(
    variant_idx: int,
    activations: np.ndarray,
    status_map: dict[int, str],
    k: int = 20,
) -> dict:
    """Get composition (TP/FP/FN counts) of k-NN neighborhood.

    Args:
        variant_idx: Index of variant
        activations: shape (n_variants, d)
        status_map: dict mapping variant_idx → status ('TP', 'FP', 'FN')
        k: number of neighbors

    Returns:
        dict with counts and proportions
    """
    knn_set = compute_knn_set(variant_idx, activations, k=k)
    knn_statuses = [status_map.get(i, 'unknown') for i in knn_set]

    counts = Counter(knn_statuses)

    return {
        'tp_count': counts.get('TP', 0),
        'fp_count': counts.get('FP', 0),
        'fn_count': counts.get('FN', 0),
        'tp_prop': counts.get('TP', 0) / k,
        'fp_prop': counts.get('FP', 0) / k,
        'fn_prop': counts.get('FN', 0) / k,
    }


def chi_squared_composition_test(
    comp_l: dict,
    comp_l1: dict,
    k: int = 20,
) -> tuple[float, float]:
    """Chi-squared test for composition distribution change.

    Returns: (chi2_stat, p_value)
    """
    observed = np.array([
        [comp_l['tp_count'], comp_l['fp_count'], comp_l['fn_count']],
        [comp_l1['tp_count'], comp_l1['fp_count'], comp_l1['fn_count']],
    ])

    chi2, p_value, dof, expected = chi2_contingency(observed)
    return chi2, p_value


def detect_flip_in_transition(
    variant_idx: int,
    eid: int,
    activations_l: np.ndarray,
    activations_l1: np.ndarray,
    status_map: dict[int, str],
    layer_from: str,
    layer_to: str,
    k: int = 20,
    jaccard_threshold: float = 0.6,
    composition_threshold: float = 0.1,
    chi2_threshold: float = 0.05,
) -> FlipResult:
    """Detect if a flip occurs between two layers.

    A flip is "real" if:
    - Jaccard < threshold (composition changed)
    - χ² test p < threshold (change is significant)
    - Effect is not k-dependent (robustness)
    """
    # Normalize activations
    act_l = activations_l.copy().astype(np.float32)
    act_l1 = activations_l1.copy().astype(np.float32)

    act_l = (act_l - act_l.mean(axis=0)) / (act_l.std(axis=0) + 1e-8)
    act_l1 = (act_l1 - act_l1.mean(axis=0)) / (act_l1.std(axis=0) + 1e-8)

    # Compute metrics at k=20
    jaccard = compute_jaccard_similarity(variant_idx, act_l, act_l1, k=k)
    comp_l = get_neighborhood_composition(variant_idx, act_l, status_map, k=k)
    comp_l1 = get_neighborhood_composition(variant_idx, act_l1, status_map, k=k)

    tp_change = abs(comp_l['tp_prop'] - comp_l1['tp_prop'])
    fp_change = abs(comp_l['fp_prop'] - comp_l1['fp_prop'])
    fn_change = abs(comp_l['fn_prop'] - comp_l1['fn_prop'])
    max_comp_change = max(tp_change, fp_change, fn_change)

    chi2, pval = chi_squared_composition_test(comp_l, comp_l1, k=k)

    # Check robustness across k values
    is_robust = True
    jaccard_values = [jaccard]
    for test_k in [10, 15, 25, 30]:
        j_test = compute_jaccard_similarity(variant_idx, act_l, act_l1, k=test_k)
        jaccard_values.append(j_test)

    jaccard_std = np.std(jaccard_values)
    if jaccard_std > 0.1:  # Not robust if std is high
        is_robust = False

    # Decision: is this a real flip?
    is_real = (
        jaccard < jaccard_threshold and
        pval < chi2_threshold and
        max_comp_change > composition_threshold and
        is_robust
    )

    # Confidence score: 0-1
    votes = 0
    if jaccard < jaccard_threshold:
        votes += 1
    if pval < chi2_threshold:
        votes += 1
    if max_comp_change > composition_threshold:
        votes += 1
    confidence = votes / 3.0

    return FlipResult(
        eid=eid,
        layer_from=layer_from,
        layer_to=layer_to,
        jaccard=jaccard,
        tp_change=tp_change,
        fp_change=fp_change,
        fn_change=fn_change,
        chi2_pval=pval,
        is_robust=is_robust,
        is_real_flip=is_real,
        confidence=confidence,
    )


def detect_all_flips(
    variant_idx: int,
    eid: int,
    activations_by_layer: dict[str, np.ndarray],
    layer_order: list[str],
    status_map: dict[int, str],
    k: int = 20,
) -> list[FlipResult]:
    """Detect flips across all layer transitions for a variant.

    Args:
        variant_idx: Index of variant in activation arrays
        eid: Variant ID (for output)
        activations_by_layer: dict mapping layer name → activation array
        layer_order: ordered list of layer names
        status_map: dict mapping variant index → status
        k: k-NN parameter

    Returns:
        List of FlipResult objects (only includes detected flips)
    """
    flips = []

    for i in range(len(layer_order) - 1):
        layer_from = layer_order[i]
        layer_to = layer_order[i + 1]

        if layer_from not in activations_by_layer or layer_to not in activations_by_layer:
            continue

        result = detect_flip_in_transition(
            variant_idx=variant_idx,
            eid=eid,
            activations_l=activations_by_layer[layer_from],
            activations_l1=activations_by_layer[layer_to],
            status_map=status_map,
            layer_from=layer_from,
            layer_to=layer_to,
            k=k,
        )

        # Only include if it's a real flip
        if result.is_real_flip:
            flips.append(result)

    return flips


def get_layer_assignments(
    variant_idx: int,
    eid: int,
    activations_by_layer: dict[str, np.ndarray],
    layer_order: list[str],
    status_map: dict[int, str],
    k: int = 20,
) -> dict[str, str]:
    """Get k-NN status assignment at each layer.

    Returns: dict mapping layer name → dominant status ('TP', 'FP', 'FN', 'Mixed')
    """
    assignments = {}

    for layer in layer_order:
        if layer not in activations_by_layer:
            assignments[layer] = None
            continue

        comp = get_neighborhood_composition(
            variant_idx,
            activations_by_layer[layer],
            status_map,
            k=k,
        )

        # Determine dominant status (70% threshold for "Mixed")
        if comp['tp_prop'] >= 0.7:
            assignments[layer] = 'TP'
        elif comp['fp_prop'] >= 0.7:
            assignments[layer] = 'FP'
        elif comp['fn_prop'] >= 0.7:
            assignments[layer] = 'FN'
        else:
            assignments[layer] = 'Mixed'

    return assignments
