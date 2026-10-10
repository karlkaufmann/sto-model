from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .approaches import ApproachRun, fit_cdp_kmeans, fit_latent_gmm, fit_rfm_birch, predict_segments
from .data import add_rule_segments, guest_ref, load_clustering_export

DEFAULT_OUT_DIR = Path("data/processed/clustering_experiments")


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _sample_fit_frame(df: pd.DataFrame, max_fit_rows: int | None) -> pd.DataFrame:
    candidates = df[df["fit_candidate"]].copy()
    if max_fit_rows is None or len(candidates) <= max_fit_rows:
        return df
    sampled_candidates = candidates.sample(max_fit_rows, random_state=42)
    rules = df[~df["fit_candidate"]]
    return pd.concat([sampled_candidates, rules], ignore_index=True)


def _rule_examples(df: pd.DataFrame, segment: str, n: int = 8) -> list[dict[str, Any]]:
    sample = df[df["routing_segment"].eq(segment)].head(n)
    examples: list[dict[str, Any]] = []
    for _, row in sample.iterrows():
        examples.append(
            {
                "guest_ref": guest_ref(row),
                "hotel_id": int(row["hotel_id"]) if pd.notna(row.get("hotel_id")) else None,
                "engagement_state": str(row.get("engagement_state", "")),
                "regime_bucket": str(row.get("regime_bucket", "")),
                "reachable": int(row.get("reachable", 0) or 0),
                "sents_total": int(row.get("sents_total", 0) or 0),
                "stays_total": int(row.get("stays_total", 0) or 0),
                "rank_spend_total": round(float(row.get("rank_spend_total", 0) or 0), 3),
            }
        )
    return examples


def _save_approach(out_dir: Path, run: ApproachRun, source_df: pd.DataFrame) -> dict[str, Any]:
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{run.artifact.name}.pkl"
    with model_path.open("wb") as f:
        pickle.dump(run.artifact, f)

    predictions = predict_segments(run.artifact, source_df)
    pred_counts = predictions.value_counts().sort_index()
    summary = run.summary.copy()
    summary["model_path"] = str(model_path)
    summary["all_row_segment_counts"] = {str(k): int(v) for k, v in pred_counts.to_dict().items()}
    return summary


def run(out_dir: Path, max_fit_rows: int | None, cdp_clusters: int = 10) -> dict[str, Any]:
    t0 = time.time()
    raw = load_clustering_export()
    routed = add_rule_segments(raw)
    fit_frame = _sample_fit_frame(routed, max_fit_rows)
    fit_candidates = fit_frame[fit_frame["fit_candidate"]]

    approach_runs = [
        fit_cdp_kmeans(fit_frame, n_clusters=cdp_clusters),
        fit_rfm_birch(fit_frame),
        fit_latent_gmm(fit_frame),
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = [_save_approach(out_dir, approach, routed) for approach in approach_runs]

    report = {
        "source": "data/raw/clustering.zip",
        "rows_total": int(len(routed)),
        "rows_fit_candidates_total": int(routed["fit_candidate"].sum()),
        "rows_fit_candidates_used": int(len(fit_candidates)),
        "max_fit_rows": max_fit_rows,
        "cdp_clusters": int(cdp_clusters),
        "rule_segment_counts": {
            str(k): int(v) for k, v in routed["routing_segment"].value_counts().to_dict().items()
        },
        "rule_detail_counts": {
            str(k): int(v) for k, v in routed["routing_detail"].value_counts().to_dict().items()
        },
        "engagement_state_counts": {
            str(k): int(v) for k, v in routed["engagement_state"].value_counts(dropna=False).to_dict().items()
        },
        "rule_examples": {
            "COLD_CONTACTS": _rule_examples(routed, "COLD_CONTACTS"),
            "NOT_TARGETABLE": _rule_examples(routed, "NOT_TARGETABLE"),
        },
        "approaches": summaries,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    report_path = out_dir / "approach_comparison.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=_json_default)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--max-fit-rows",
        type=int,
        default=150_000,
        help="Limit learned-cluster fit rows for fast local experiments; pass 0 to use all candidates.",
    )
    parser.add_argument(
        "--cdp-clusters",
        type=int,
        default=10,
        help="Number of clusters for the primary CDP k-means approach.",
    )
    args = parser.parse_args()
    max_fit_rows = None if args.max_fit_rows == 0 else args.max_fit_rows
    report = run(Path(args.out_dir), max_fit_rows=max_fit_rows, cdp_clusters=args.cdp_clusters)
    print(json.dumps({
        "rows_total": report["rows_total"],
        "rows_fit_candidates_total": report["rows_fit_candidates_total"],
        "rows_fit_candidates_used": report["rows_fit_candidates_used"],
        "cdp_clusters": report["cdp_clusters"],
        "rule_segment_counts": report["rule_segment_counts"],
        "approaches": [approach["name"] for approach in report["approaches"]],
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
