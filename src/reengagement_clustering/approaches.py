from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import Birch, MiniBatchKMeans
from scipy.optimize import linear_sum_assignment
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, guest_ref, prepare_features


@dataclass
class FittedApproach:
    name: str
    display_name: str
    description: str
    estimator: Pipeline
    numeric_features: list[str]
    categorical_features: list[str]


@dataclass
class ApproachRun:
    artifact: FittedApproach
    assignments: pd.Series
    summary: dict[str, Any]


ENGAGEMENT_NUMERIC = [
    "sents_30d",
    "sents_90d",
    "opens_30d",
    "opens_90d",
    "opens_ts_90d",
    "clicks_30d",
    "clicks_90d",
    "dsn_90d",
    "dsn_hard_total",
    "days_since_last_send",
    "days_since_last_open",
    "days_since_last_click",
    "open_rate_30d",
    "open_rate_90d",
    "open_rate_lifetime",
    "click_rate_90d",
    "dsn_rate_90d",
    "delivery_rate_90d",
    "rate_delta_30_vs_90",
    "rank_sents_total",
    "rank_opens_total",
    "rank_clicks_total",
    "rank_open_rate_90d",
]

BOOKING_NUMERIC = [
    "stays_total",
    "stays_confirmed",
    "stays_past",
    "stays_future",
    "stays_365d",
    "bookings_created_90d",
    "bookings_created_365d",
    "days_since_last_stay",
    "days_to_next_stay",
    "nights_total",
    "currency_count",
    "lead_time_avg_days",
    "nonrefundable_share",
    "insured_share",
    "days_since_last_booking",
    "hotels_count_contact",
    "booking_custom_params_count",
    "rank_spend_total",
    "rank_stays_total",
    "rank_nights_total",
    "rank_recency_stay",
    "recency_stay_score",
]

PROFILE_NUMERIC = [
    "age",
    "age_known",
    "registration_age_days",
    "consent_marketing_active",
    "consent_profiling_active",
    "consent_doubleoptin_active",
    "newsletter_optout_active",
    "signals_count",
]

LOG_NUMERIC = [
    "log1p_stays_total",
    "log1p_stays_365d",
    "log1p_nights_total",
    "log1p_sents_total",
    "log1p_sents_90d",
    "log1p_opens_total",
    "log1p_opens_90d",
    "log1p_clicks_total",
    "log1p_clicks_90d",
    "log1p_campaigns_distinct",
    "log1p_clicked_campaigns_distinct",
]

MIXED_CATEGORICAL = [
    "language",
    "country",
    "gender",
    "email_domain",
    "booking_channel",
    "booking_source_code",
    "booking_room_type",
    "currency_dominant",
    "mailing_types",
    "engagement_state",
]

PROFILE_COLUMNS = [
    "open_rate_90d",
    "click_rate_90d",
    "sents_90d",
    "opens_90d",
    "clicks_90d",
    "days_since_last_open",
    "stays_total",
    "rank_spend_total",
    "rank_recency_stay",
]

LABEL_DESCRIPTIONS = {
    "Oslovení bez reakce": "Kontakty, které dostávají e-maily, ale v posledních oknech na ně prakticky nereagují.",
    "Dřívější hosté s velmi nízkou/žádnou odezvou": "Dřívější hosté s booking historií, u kterých je e-mailová odezva velmi nízká nebo žádná.",
    "Aktivní klikači": "Aktivní kontakty s čerstvými openy a nadprůměrnou klikací aktivitou.",
    "Aktivní čtenáři": "Aktivní kontakty, které nedávno otevíraly e-maily, ale méně klikají.",
    "Dřívější hosté s klesající odezvou": "Hosté s pobytovou historií a dřívější reakcí, kterým engagement postupně slábne.",
    "Kontakty bez pobytové historie": "Kontakty s nulovou nebo minimální pobytovou historií a nízkou nebo starou e-mailovou aktivitou.",
    "Tichí příjemci": "Nízkoreakční příjemci bez spolehlivého open/click signálu.",
    "Hodnotní hosté": "Hosté s vyšší pobytovou nebo útratovou hodnotou než zbytek tenantového publika.",
    "Klikající zájemci": "Kontakty, které kromě openů reálně klikají na odkazy a projevují zájem.",
    "Středně aktivní CRM kontakty": "Střední segment bez výrazného booking, click nebo recency signálu.",
}


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _mixed_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _one_hot_encoder()),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def _numeric_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ],
        remainder="drop",
    )


def _available(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def _label_cluster(profile: pd.Series) -> str:
    engagement = str(profile.get("engagement_state_mode", ""))
    open_rate = float(profile.get("open_rate_90d_mean", 0) or 0)
    click_rate = float(profile.get("click_rate_90d_mean", 0) or 0)
    spend_rank = float(profile.get("rank_spend_total_mean", 0) or 0)
    recency = float(profile.get("days_since_last_open_median", 9999) or 9999)
    stays = float(profile.get("stays_total_mean", 0) or 0)
    sents_90d = float(profile.get("sents_90d_mean", 0) or 0)

    if engagement == "no_open" and sents_90d >= 2:
        return "Oslovení bez reakce"
    if recency <= 30 and open_rate >= 0.5 and click_rate >= 0.05:
        return "Aktivní klikači"
    if recency <= 45 and open_rate >= 0.5:
        return "Aktivní čtenáři"
    if engagement == "fading" and stays >= 1.2 and recency >= 500:
        return "Dřívější hosté s velmi nízkou/žádnou odezvou"
    if engagement == "fading" and stays >= 1.2 and open_rate >= 0.25:
        return "Dřívější hosté s klesající odezvou"
    if engagement == "fading" and stays < 0.25:
        return "Kontakty bez pobytové historie"
    if engagement == "no_open" or open_rate < 0.05:
        return "Tichí příjemci"
    if stays >= 1.5 or spend_rank >= 0.85:
        return "Hodnotní hosté"
    if click_rate > 0.05:
        return "Klikající zájemci"
    return "Středně aktivní CRM kontakty"


def _label_description(label: str) -> str:
    return LABEL_DESCRIPTIONS[label]


def _cluster_examples(
    df: pd.DataFrame,
    cluster: pd.Series,
    cluster_id: int,
    n: int = 3,
    farthest: bool = False,
) -> list[dict[str, Any]]:
    group = df.loc[cluster.eq(cluster_id)].copy()
    if group.empty:
        return []
    available = [col for col in PROFILE_COLUMNS if col in group.columns]
    medians = group[available].median(numeric_only=True)
    distances = pd.Series(0.0, index=group.index)
    for col in available:
        scale = float(group[col].quantile(0.75) - group[col].quantile(0.25)) or 1.0
        distances += (group[col] - medians[col]).abs().fillna(0) / scale
    ordered = distances.sort_values(ascending=not farthest)
    sample = group.loc[ordered.head(n).index]
    examples: list[dict[str, Any]] = []
    for idx, row in sample.iterrows():
        examples.append(
            {
                "guest_ref": guest_ref(row),
                "distance_to_cluster_profile": round(float(distances.loc[idx]), 3),
                "hotel_id": int(row["hotel_id"]) if pd.notna(row.get("hotel_id")) else None,
                "engagement_state": str(row.get("engagement_state", "")),
                "sents_90d": int(row.get("sents_90d", 0) or 0),
                "opens_90d": int(row.get("opens_90d", 0) or 0),
                "clicks_90d": int(row.get("clicks_90d", 0) or 0),
                "open_rate_90d": round(float(row.get("open_rate_90d", 0) or 0), 3),
                "click_rate_90d": round(float(row.get("click_rate_90d", 0) or 0), 3),
                "days_since_last_open": None
                if pd.isna(row.get("days_since_last_open"))
                else int(row.get("days_since_last_open")),
                "days_since_last_click": None
                if pd.isna(row.get("days_since_last_click"))
                else int(row.get("days_since_last_click")),
                "stays_total": int(row.get("stays_total", 0) or 0),
                "rank_spend_total": round(float(row.get("rank_spend_total", 0) or 0), 3),
                "rank_recency_stay": round(float(row.get("rank_recency_stay", 0) or 0), 3),
            }
        )
    return examples


def _summarize(
    name: str,
    display_name: str,
    description: str,
    df: pd.DataFrame,
    labels: np.ndarray,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Any]:
    cluster = pd.Series(labels, index=df.index, name="cluster_id")
    profiled = df.copy()
    profiled["cluster_id"] = cluster
    profile = (
        profiled.groupby("cluster_id")
        .agg(
            rows=("contact_id", "size"),
            engagement_state_mode=("engagement_state", lambda s: str(s.mode().iloc[0]) if not s.mode().empty else ""),
            open_rate_90d_mean=("open_rate_90d", "mean"),
            click_rate_90d_mean=("click_rate_90d", "mean"),
            sents_90d_mean=("sents_90d", "mean"),
            days_since_last_open_median=("days_since_last_open", "median"),
            stays_total_mean=("stays_total", "mean"),
            rank_spend_total_mean=("rank_spend_total", "mean"),
            rank_recency_stay_mean=("rank_recency_stay", "mean"),
        )
        .round(4)
    )

    clusters: list[dict[str, Any]] = []
    for cluster_id, row in profile.iterrows():
        row_dict = row.to_dict()
        label = _label_cluster(pd.Series(row_dict))
        clusters.append(
            {
                "cluster_id": int(cluster_id),
                "label": label,
                "description": _label_description(label),
                "profile": {
                    key: (int(value) if key == "rows" else float(value) if isinstance(value, (np.floating, float)) else value)
                    for key, value in row_dict.items()
                },
                "examples": _cluster_examples(profiled, cluster, int(cluster_id), n=5),
                "top_examples": _cluster_examples(profiled, cluster, int(cluster_id), n=3),
                "edge_examples": _cluster_examples(profiled, cluster, int(cluster_id), n=3, farthest=True),
            }
        )

    return {
        "name": name,
        "display_name": display_name,
        "description": description,
        "fit_rows": int(len(df)),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "cluster_sizes": {str(k): int(v) for k, v in cluster.value_counts().sort_index().to_dict().items()},
        "clusters": clusters,
    }


def predict_segments(artifact: FittedApproach, df: pd.DataFrame) -> pd.Series:
    prepared = prepare_features(df)
    output = prepared["routing_segment"].astype(str).copy()
    candidate_mask = prepared["fit_candidate"].astype(bool)
    if candidate_mask.any():
        features = prepared.loc[candidate_mask, artifact.numeric_features + artifact.categorical_features]
        learned = artifact.estimator.predict(features)
        output.loc[candidate_mask] = [f"{artifact.name}:{int(value)}" for value in learned]
    return output


def fit_cdp_kmeans(df: pd.DataFrame, n_clusters: int = 5) -> ApproachRun:
    prepared = prepare_features(df)
    numeric = _available(prepared, ENGAGEMENT_NUMERIC + BOOKING_NUMERIC + PROFILE_NUMERIC + LOG_NUMERIC)
    categorical = _available(prepared, MIXED_CATEGORICAL)
    estimator = Pipeline(
        steps=[
            ("preprocess", _mixed_preprocessor(numeric, categorical)),
            (
                "cluster",
                MiniBatchKMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    batch_size=8192,
                    n_init=10,
                    max_iter=100,
                ),
            ),
        ]
    )
    fit_df = prepared[prepared["fit_candidate"]].copy()
    labels = estimator.fit_predict(fit_df[numeric + categorical])
    artifact = FittedApproach(
        name="cdp_kmeans",
        display_name="CDP mixed k-means",
        description="Centroid clustering over engagement, booking, profile, and normalized categorical features.",
        estimator=estimator,
        numeric_features=numeric,
        categorical_features=categorical,
    )
    return ApproachRun(
        artifact=artifact,
        assignments=pd.Series(labels, index=fit_df.index),
        summary=_summarize(artifact.name, artifact.display_name, artifact.description, fit_df, labels, numeric, categorical),
    )


def align_kmeans_to_reference(run: ApproachRun, reference: FittedApproach, df: pd.DataFrame) -> ApproachRun:
    """Reorder a retrained k-means so its cluster ids keep the reference model's slots.

    k-means numbers clusters arbitrarily, while serving maps ids to static names.
    Each new cluster takes the reference id that most of its contacts fall into
    (Hungarian assignment on the contact overlap), so a retrain does not silently
    shuffle names. Overlap per id is reported so names can be re-checked.
    Assumes the reference was trained with the same feature preparation; after a
    change to prepare_features its predictions are not a fair basis for alignment.
    """
    prepared = prepare_features(df)
    fit_df = prepared.loc[run.assignments.index]
    new_labels = run.assignments.to_numpy()
    ref_labels = reference.estimator.predict(fit_df[reference.numeric_features + reference.categorical_features])
    k = run.artifact.estimator.named_steps["cluster"].n_clusters
    # A reference cluster can receive no contacts at all; keep it as an empty column.
    overlap = pd.crosstab(new_labels, ref_labels).reindex(index=range(k), columns=range(k), fill_value=0)
    rows, cols = linear_sum_assignment(-overlap.to_numpy())
    slot_of = {int(overlap.index[r]): int(overlap.columns[c]) for r, c in zip(rows, cols)}

    kmeans = run.artifact.estimator.named_steps["cluster"]
    order = np.array([new for slot in range(kmeans.n_clusters) for new, s in slot_of.items() if s == slot])
    kmeans.cluster_centers_ = kmeans.cluster_centers_[order]
    kmeans.labels_ = np.argsort(order)[kmeans.labels_].astype(kmeans.labels_.dtype)

    aligned = np.vectorize(slot_of.get)(new_labels)
    artifact = run.artifact
    summary = _summarize(
        artifact.name, artifact.display_name, artifact.description, fit_df, aligned,
        artifact.numeric_features, artifact.categorical_features,
    )
    summary["alignment_overlap"] = {
        str(slot): round(float(overlap.loc[new, slot] / overlap.loc[new].sum()), 3) for new, slot in slot_of.items()
    }
    return ApproachRun(artifact=artifact, assignments=pd.Series(aligned, index=run.assignments.index), summary=summary)


def fit_rfm_birch(df: pd.DataFrame, n_clusters: int = 5) -> ApproachRun:
    prepared = prepare_features(df)
    numeric = _available(
        prepared,
        [
            "rank_spend_total",
            "rank_stays_total",
            "rank_nights_total",
            "rank_recency_stay",
            "rank_sents_total",
            "rank_opens_total",
            "rank_clicks_total",
            "rank_open_rate_90d",
            "recency_stay_score",
            "days_since_last_send",
            "days_since_last_open",
            "days_since_last_click",
            "open_rate_90d",
            "click_rate_90d",
            "rate_delta_30_vs_90",
            "log1p_stays_total",
            "log1p_sents_90d",
            "log1p_opens_90d",
            "log1p_clicks_90d",
        ],
    )
    estimator = Pipeline(
        steps=[
            ("preprocess", _numeric_preprocessor(numeric)),
            ("cluster", Birch(n_clusters=n_clusters, threshold=0.7)),
        ]
    )
    fit_df = prepared[prepared["fit_candidate"]].copy()
    labels = estimator.fit_predict(fit_df[numeric])
    artifact = FittedApproach(
        name="rfm_birch",
        display_name="RFM BIRCH",
        description="Incremental numeric clustering focused on recency, frequency, monetary rank, and engagement rank.",
        estimator=estimator,
        numeric_features=numeric,
        categorical_features=[],
    )
    return ApproachRun(
        artifact=artifact,
        assignments=pd.Series(labels, index=fit_df.index),
        summary=_summarize(artifact.name, artifact.display_name, artifact.description, fit_df, labels, numeric, []),
    )


def fit_latent_gmm(df: pd.DataFrame, n_clusters: int = 5) -> ApproachRun:
    prepared = prepare_features(df)
    numeric = _available(prepared, ENGAGEMENT_NUMERIC + BOOKING_NUMERIC + PROFILE_NUMERIC + LOG_NUMERIC)
    categorical = _available(prepared, [col for col in CATEGORICAL_FEATURES if col != "engagement_state"])
    estimator = Pipeline(
        steps=[
            ("preprocess", _mixed_preprocessor(numeric, categorical)),
            ("svd", TruncatedSVD(n_components=16, random_state=42)),
            ("scale", StandardScaler()),
            ("cluster", GaussianMixture(n_components=n_clusters, covariance_type="diag", random_state=42, max_iter=100)),
        ]
    )
    fit_df = prepared[prepared["fit_candidate"]].copy()
    labels = estimator.fit_predict(fit_df[numeric + categorical])
    artifact = FittedApproach(
        name="latent_gmm",
        display_name="Latent SVD Gaussian mixture",
        description="Probabilistic clustering on a dense latent representation of mixed CRM features.",
        estimator=estimator,
        numeric_features=numeric,
        categorical_features=categorical,
    )
    return ApproachRun(
        artifact=artifact,
        assignments=pd.Series(labels, index=fit_df.index),
        summary=_summarize(artifact.name, artifact.display_name, artifact.description, fit_df, labels, numeric, categorical),
    )
