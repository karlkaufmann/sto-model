"""Find which clustering-export tenant matches a client's customer list (by e-mail hash).

E-mails are hashed in memory only; output is aggregate match counts per tenant,
never e-mails or hashes. Tries several normalisations because we don't know how
the exporter normalised e-mails before hashing.

Usage: python src/data/match_client_tenant.py [client.xlsx]
"""
import hashlib
import sys

import pandas as pd

DEFAULT_XLSX = "data/secure/client_segments.xlsx"
RAW_ZIP = "data/raw/ClusteringDataset-07-10-2026.zip"

VARIANTS = {
    "as_is": lambda e: e,
    "strip_lower": lambda e: e.strip().lower(),
    "strip_upper": lambda e: e.strip().upper(),
}


def sha256(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_export_hashes():
    import io
    import zipfile

    import pyarrow.parquet as pq

    frames = []
    with zipfile.ZipFile(RAW_ZIP) as zf:
        for name in zf.namelist():
            if not name.endswith(".parquet"):
                continue
            t = pq.read_table(io.BytesIO(zf.read(name)), columns=["tenant_id", "email_sha256"])
            frames.append(t.to_pandas())
    df = pd.concat(frames, ignore_index=True).dropna(subset=["email_sha256"])
    df["email_sha256"] = df["email_sha256"].str.lower()
    return df.drop_duplicates()


def main(xlsx):
    emails = pd.read_excel(xlsx, sheet_name="DB ZÁKAZNÍCI", usecols=["E-mail"])["E-mail"].dropna().astype(str)
    exp = load_export_hashes()
    print(f"client customers: {len(emails)}")
    print(f"export rows with hash: {len(exp)}  tenants: {exp['tenant_id'].nunique()}\n")

    for vname, fn in VARIANTS.items():
        hashes = set(emails.map(lambda e: sha256(fn(e))))
        hit = exp[exp["email_sha256"].isin(hashes)]
        per_tenant = hit.groupby("tenant_id")["email_sha256"].nunique().sort_values(ascending=False)
        print(f"[{vname}] matched customers (any tenant): {hit['email_sha256'].nunique()}")
        for tenant, n in per_tenant.head(5).items():
            size = exp.loc[exp["tenant_id"] == tenant, "email_sha256"].nunique()
            print(f"   {tenant:<28} matched={n:>6}  ({100*n/len(emails):5.1f}% of client, "
                  f"{100*n/size:5.1f}% of tenant's {size} contacts)")
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_XLSX)
