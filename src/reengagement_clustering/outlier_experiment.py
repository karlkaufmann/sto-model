"""Two-run comparison: CDP mixed k-means with vs. without a silhouette-based
outlier filter.

Run A (baseline): the current production candidate - MiniBatchKMeans(k=10)
fit on all learned-cluster candidates, no outlier handling.

Run B (filtered): same pipeline and same random_state, but rows whose
silhouette score (measured on a random sample, since silhouette is O(n^2))
falls in the bottom `outlier_pct` are excluded from the *fit* before
re-running k-means. This never touches routing_segment / fit_candidate -
it only changes what MiniBatchKMeans sees during fit.

Usage:
    python -m src.reengagement_clustering.outlier_experiment
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score

from .approaches import fit_cdp_kmeans
from .data import add_rule_segments, load_clustering_export, prepare_features

OUT_DIR = Path("data/processed/clustering_experiments")
SILHOUETTE_SAMPLE_SIZE = 20_000
OUTLIER_PCT = 0.02  # bottom 2% by silhouette score, measured on the sample
RANDOM_STATE = 42


def _cluster_sizes(labels: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def _balance_ratio(sizes: dict[str, int]) -> float:
    values = list(sizes.values())
    return round(max(values) / min(values), 2) if min(values) > 0 else float("inf")


def _silhouette_on_sample(
    estimator, fit_df: pd.DataFrame, numeric: list[str], categorical: list[str], labels: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Transform the fitted preprocessing step and score a random sample."""
    n = len(fit_df)
    sample_size = min(SILHOUETTE_SAMPLE_SIZE, n)
    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(n, size=sample_size, replace=False)

    prepared = prepare_features(fit_df)
    transformed = estimator.named_steps["preprocess"].transform(prepared[numeric + categorical])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    sample_X = transformed[sample_idx]
    sample_labels = labels[sample_idx]

    score = float(silhouette_score(sample_X, sample_labels))
    per_point = silhouette_samples(sample_X, sample_labels)
    return score, per_point, sample_idx


def run() -> dict[str, Any]:
    raw = load_clustering_export()
    routed = add_rule_segments(raw)

    # --- Run A: baseline, no outlier filter ---------------------------------
    run_a = fit_cdp_kmeans(routed, n_clusters=10)
    fit_df_a = routed[routed["fit_candidate"]].copy()
    labels_a = run_a.assignments.values
    silhouette_a, per_point_a, sample_idx_a = _silhouette_on_sample(
        run_a.artifact.estimator, fit_df_a, run_a.artifact.numeric_features, run_a.artifact.categorical_features, labels_a
    )
    sizes_a = _cluster_sizes(labels_a)

    # --- Run B: drop bottom outlier_pct silhouette rows (from the sample),
    #     then refit on the reduced fit frame with the same random_state ----
    cutoff = np.quantile(per_point_a, OUTLIER_PCT)
    outlier_positions_in_sample = sample_idx_a[per_point_a <= cutoff]
    outlier_index = fit_df_a.index[outlier_positions_in_sample]
    fit_df_b_input = routed.drop(index=outlier_index)

    run_b = fit_cdp_kmeans(fit_df_b_input, n_clusters=10)
    fit_df_b = fit_df_b_input[fit_df_b_input["fit_candidate"]].copy()
    labels_b = run_b.assignments.values
    silhouette_b, _, _ = _silhouette_on_sample(
        run_b.artifact.estimator, fit_df_b, run_b.artifact.numeric_features, run_b.artifact.categorical_features, labels_b
    )
    sizes_b = _cluster_sizes(labels_b)

    comparison = {
        "silhouette_sample_size": SILHOUETTE_SAMPLE_SIZE,
        "outlier_pct_removed": OUTLIER_PCT,
        "run_a_baseline": {
            "fit_rows": int(len(fit_df_a)),
            "silhouette_score_sample": round(silhouette_a, 4),
            "cluster_sizes": sizes_a,
            "max_min_ratio": _balance_ratio(sizes_a),
        },
        "run_b_outlier_filtered": {
            "fit_rows": int(len(fit_df_b)),
            "rows_dropped_as_outliers": int(len(outlier_index)),
            "silhouette_score_sample": round(silhouette_b, 4),
            "cluster_sizes": sizes_b,
            "max_min_ratio": _balance_ratio(sizes_b),
        },
        "delta": {
            "silhouette_improvement": round(silhouette_b - silhouette_a, 4),
            "max_min_ratio_change": round(_balance_ratio(sizes_b) - _balance_ratio(sizes_a), 2),
        },
    }
    return comparison


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison = run()
    out_path = OUT_DIR / "outlier_filter_comparison.json"
    out_path.write_text(json.dumps(comparison, indent=2))
    print(json.dumps(comparison, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
