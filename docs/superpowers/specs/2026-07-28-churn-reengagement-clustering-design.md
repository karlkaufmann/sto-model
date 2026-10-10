# Churn Re-Engagement Clustering Design

## Goal

Build an unsupervised contact clustering workflow for hotel CRM re-engagement campaigns. The primary deliverable is 4-6 interpretable personas, with explicit separation of rule-based engagement regimes from k-means clusters fitted only on clusterable contacts.

This is not a supervised churn score. The first production dataset must be built from Zoomle production CRM `nwl3_*` tables. The existing STO export is suitable for prototype validation only.

## Decisions

- Primary output is clustering, not a scalar churn score.
- Universe is the full contact pool, including inactive and cold contacts.
- Grain is one row per client within tenant or hotel.
- Cross-tenant email identities must not be merged.
- STO identity is `tenant_id x recipient_email_sha256`.
- Zoomle identity is `id_klienta` within the tenant or hotel extract.
- Use engagement windows of 30 days and 90 days.
- Rule regimes stay outside k-means and are reported as separate buckets.
- Start k-means with `k=5` on clusterable contacts.

## Success Criteria

- Produces 4-6 interpretable personas usable for re-engagement campaign planning.
- Cleanly separates fading contacts from healthy engaged contacts.
- Keeps cold, inactive, opted-out, and unreachable regimes explainable without forcing them into k-means.
- Reports week-over-week stability so segment definitions are not dominated by extraction noise.
- Writes only code, configs, docs, and lightweight JSON metrics to git-trackable paths; raw extracts, archives, parquet files, model binaries, and secrets remain gitignored.

## Data Sources

### STO Prototype Export

Use `data/raw/data_6_6_2026.zip` only for prototyping. Known properties:

- About 4.18M rows before exact event deduplication.
- About 2.38M events after exact deduplication.
- About 211k `tenant_id x recipient_email_sha256` contacts.
- `mail_event_id` is not globally unique and must not be used as the only deduplication key.
- Existing CUDA V4 open-label training rows are not the contact pool.
- Booking coverage is about 20%; CRM counters coverage is about 58%.
- Engagement windows must be recomputed from events.
- A useful prototype clusterable rule is at least 2 sent emails and at least 1 open event, producing about 96k contacts in the export.

### Zoomle Production CRM

Use `nwl3_*` as source of truth for the production dataset.

Core tables:

- `nwl3_klienti`: client profile and socio-demographics.
- `nwl3_gdpr` and GDPR history: consent state and consent age.
- `nwl3_rezervace`: booking history, including custom fields when available.
- `nwl3_mailer_processed`: send, delivery, bounce, open-like email processing events.
- `nwl3_zoomtrack`: email click events.
- `nwl3_klient2skupina`: group membership.

Secondary tables are optional for v1 because coverage is sparse: custom client columns, tags, loyalty, web behavior, check-in, questionnaires, and e-shop.

## Dataset Contract

The clustering dataset is one row per contact identity. Required identity columns:

- `tenant_id`
- `contact_id`
- `identity_source`, with values such as `sto_sha256` or `zl_id_klienta`

Required target-free feature families:

- Engagement features for 30-day and 90-day windows.
- Socio-demographic features and matching `*_known` indicators.
- Booking aggregate features and matching `*_known` indicators.
- Consent and reachability features.
- Tenant-normalized ranks for engagement, booking value, and recency.
- `regime_bucket`.
- `is_clusterable`.

The dataset must not include campaign/template wide one-hot features in v1. Sending-time features are out of scope because this workflow clusters contacts, not send-time behavior.

## Regime Buckets

Assign rule regimes before clustering. Initial buckets:

- `unreachable`: invalid email, hard bounce, or no active email consent.
- `cold_never_engaged`: no opens and no clicks in the observed history.
- `inactive_low_signal`: too few sent emails to infer engagement reliably.
- `healthy_engaged`: recent open or click activity and stable 30-day versus 90-day rate.
- `fading`: 90-day engagement exists but 30-day engagement has materially declined.
- `clusterable`: enough send and open or click history for k-means.

`clusterable` is an eligibility flag for k-means, not a final persona label by itself. A contact can carry an interpretable regime bucket and, if eligible, a cluster id.

## Feature Set V1

Engagement:

- `sents_30d`, `sents_90d`
- `opens_30d`, `opens_90d`
- `clicks_30d`, `clicks_90d`
- `dsn_30d`, `dsn_90d`
- `open_rate_30d`, `open_rate_90d`
- `click_rate_30d`, `click_rate_90d`
- `days_since_last_email`
- `days_since_last_open`
- `days_since_last_click`
- `open_rate_delta_30_vs_90`
- `click_rate_delta_30_vs_90`

Socio-demo:

- language, country, age, gender where available.
- consent state.
- registration age.
- `*_known` flags for sparse fields.

Booking:

- stay count.
- booking recency.
- total nights.
- total spend.
- average lead time.
- channel or source from custom booking fields when available.
- booking known flags.

Meta:

- `regime_bucket`.
- tenant-normalized percentile ranks.
- extraction timestamp.

## Clustering Approach

Fit k-means only on `is_clusterable = true` contacts. Use a scikit-learn pipeline with:

- numeric imputation.
- categorical imputation.
- one-hot encoding for low-cardinality categorical features.
- robust or standard scaling for numeric features.
- `KMeans(n_clusters=5, random_state=42, n_init="auto")`.

Do not include `tenant_id`, raw contact identifiers, raw email hashes, or `regime_bucket` in the k-means feature matrix. Tenant information should enter only through normalized ranks or other non-identifying aggregates.

## Outputs

Production and prototype runs should write:

- contact dataset parquet under `data/processed/`, gitignored.
- fitted clustering artifact under `models/reengagement_clustering/`, gitignored.
- feature config under `configs/reengagement_clustering/features.json`.
- metrics under `results/metrics/reengagement_clustering.json`.
- segment summary under `results/segments/reengagement_clustering_segments.json`.

The segment summary must include cluster sizes, top feature contrasts, regime counts, booking/engagement profile summaries, and human-readable label candidates.

## Stability Checks

Run stability on two adjacent extraction snapshots when available. Report:

- cluster size deltas.
- adjusted rand index for contacts present in both snapshots.
- persona label continuity based on centroid and feature-profile similarity.
- regime bucket movement matrix.

For the first STO prototype, use deterministic temporal split snapshots derived from event dates if full weekly snapshots are not available.

## Out Of Scope

- Supervised churn scoring.
- Uplift modeling.
- Campaign/template wide features.
- Sending-time optimization.
- Merging cross-tenant email identities.
- Sparse secondary production sources in v1 unless already available in the extract without new integration work.

## Risks

- STO prototype may overrepresent engaged contacts compared with production CRM.
- Sparse socio-demo fields can dominate naive missingness patterns; every sparse field needs a known flag.
- Rule thresholds for fading versus healthy need calibration with campaign owners before production use.
- Week-over-week stability requires snapshots or reproducible temporal cuts; a single flat extract can only approximate this.
