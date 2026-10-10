"""Validate production segments (cdp_kmeans.pkl) against a client's hand-made segmentation.

Joins the client's customer list to the clustering export by SHA-256 of the
lower-cased e-mail (hashed in memory; PII columns are never kept) and prints
aggregate tables only - no e-mails, hashes or row-level data.

The core question for a re-engagement model: do our "dormant / no response"
segments over-represent the client's lapsed customers (Z02-Z04) and our
"active" segments over-represent recent buyers (L01)? And do segments separate
customer value (H01-H04)?

Usage: python -m src.reengagement_clustering.validate_client_segments --tenant <tenant_id>
"""
from __future__ import annotations

import argparse
import hashlib
import pickle

import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from serving.clustering_server import CLUSTER_NAMES

from .approaches import predict_segments
from .data import READ_COLS, add_rule_segments, load_clustering_export

DEFAULT_XLSX = "data/secure/client_segments.xlsx"
DEFAULT_MODEL = "models/reengagement_clustering/cdp_kmeans.pkl"

# Mutually exclusive families per the client's METODIKA sheet.
LIFECYCLE = {"L01": "active 25/26", "Z02": "at risk (24/25)", "Z03": "sleeping (22-24)", "Z04": "historic (<=21/22)"}
VALUE = {"H01": "VIP 600+", "H02": "high 250-599", "H03": "mid 100-249", "H04": "low <100"}
FLAGS = {"Z01": "new_25_26", "L03": "loyal_2y_strict", "S06": "season_pass", "R01": "family"}

CLIENT_NUMERIC = {
    "Počet sezón s nákupom": "seasons",
    "Počet objednávok": "orders",
    "Útrata spolu (€)": "spend_eur",
}


def sha256_email(email: str) -> str:
    return hashlib.sha256(email.strip().lower().encode("utf-8")).hexdigest()


def pick_family(codes: set[str], family: dict[str, str]) -> str:
    hits = [family[c] for c in family if c in codes]
    return hits[0] if len(hits) == 1 else ("none" if not hits else "multiple")


def load_client(xlsx: str) -> pd.DataFrame:
    raw = pd.read_excel(xlsx, sheet_name="DB ZÁKAZNÍCI", usecols=["E-mail", "Segmenty (kódy)", *CLIENT_NUMERIC])
    codes = raw["Segmenty (kódy)"].fillna("").astype(str).str.findall(r"[A-Z]\d{2}").map(set)
    out = pd.DataFrame({"email_sha256": raw["E-mail"].astype(str).map(sha256_email)})
    out["lifecycle"] = codes.map(lambda c: pick_family(c, LIFECYCLE))
    out["value"] = codes.map(lambda c: pick_family(c, VALUE))
    for code, name in FLAGS.items():
        out[name] = codes.map(lambda c, code=code: code in c)
    for src, name in CLIENT_NUMERIC.items():
        out[name] = pd.to_numeric(raw[src], errors="coerce")
    return out


def load_our_segments(model_path: str, tenant: str) -> tuple[pd.DataFrame, int]:
    # Predict on the full export so per-frame categorical top-k matches training.
    df = add_rule_segments(load_clustering_export(columns=READ_COLS + ["email_sha256"]))
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    df["segment"] = predict_segments(artifact, df)
    df = df[df["tenant_id"].eq(tenant) & df["email_sha256"].notna()].copy()
    df["email_sha256"] = df["email_sha256"].str.lower()

    # One row per tenant x hotel x contact; collapse to one segment per e-mail (mode).
    conflicting = int((df.groupby("email_sha256")["segment"].nunique() > 1).sum())
    per_email = df.groupby("email_sha256")["segment"].agg(lambda s: s.mode().iloc[0]).rename("segment")
    return per_email.reset_index(), conflicting


def label(segment: str) -> str:
    if ":" not in segment:
        return segment
    cid = int(segment.split(":")[1])
    return f"{cid} {CLUSTER_NAMES.get(cid, '?')}"


def pct_table(df: pd.DataFrame, rows: str, cols: str) -> pd.DataFrame:
    counts = pd.crosstab(df[rows], df[cols])
    pct = (100 * counts.div(counts.sum(axis=1), axis=0)).round(1)
    pct.insert(0, "n", counts.sum(axis=1))
    return pct


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    with pd.option_context("display.width", 220, "display.max_columns", None, "display.max_rows", None):
        print(table.to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=DEFAULT_XLSX)
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    client = load_client(args.xlsx)
    ours, conflicting = load_our_segments(args.model, args.tenant)
    client["matched"] = client["email_sha256"].isin(ours["email_sha256"])
    joined = client.merge(ours, on="email_sha256", how="inner")
    joined["segment"] = joined["segment"].map(label)

    print(f"client customers: {len(client)}  tenant contacts (unique e-mail): {len(ours)}  matched: {len(joined)}")
    print(f"contacts with conflicting segments across hotel rows (resolved by mode): {conflicting}")

    # Is the matched subset representative of the client's base?
    for fam in ("lifecycle", "value"):
        show(f"representativeness: {fam} % (matched vs not)", pct_table(client, "matched", fam))

    show("our segment size", joined["segment"].value_counts().to_frame("n"))
    show("our segment x client lifecycle (row %)", pct_table(joined, "segment", "lifecycle"))
    show("our segment x client value (row %)", pct_table(joined, "segment", "value"))

    profile = joined.groupby("segment").agg(
        n=("segment", "size"),
        median_spend_eur=("spend_eur", "median"),
        mean_seasons=("seasons", "mean"),
        mean_orders=("orders", "mean"),
        **{f"pct_{name}": (name, lambda s: round(100 * s.mean(), 1)) for name in FLAGS.values()},
    ).round(2)
    show("client attributes per our segment", profile)

    base = pd.Series({"median_spend_eur": joined["spend_eur"].median(), "mean_seasons": joined["seasons"].mean(),
                      **{f"pct_{n}": round(100 * joined[n].mean(), 1) for n in FLAGS.values()}}).round(2)
    show("baseline (all matched)", base.to_frame("value"))

    print("\n=== agreement (0 = none, 1 = identical partition) ===")
    for fam in ("lifecycle", "value"):
        sub = joined[~joined[fam].isin(["none", "multiple"])]
        print(f"{fam:<10} ARI={adjusted_rand_score(sub[fam], sub['segment']):.3f}  "
              f"NMI={normalized_mutual_info_score(sub[fam], sub['segment']):.3f}  (n={len(sub)})")


if __name__ == "__main__":
    main()
