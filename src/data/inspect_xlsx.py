"""Print the structure of every sheet in an .xlsx file WITHOUT printing cell values.

Output is safe to share: sheet names, dimensions, header row, column names,
types and fill/cardinality stats. No min/max, samples or head() - those leak data.

Usage: python src/data/inspect_xlsx.py [path.xlsx]
"""
import sys
from collections import Counter
from datetime import date, datetime, time

import openpyxl
import pandas as pd

DEFAULT_PATH = "data/secure/client_segments.xlsx"
HEADER_SCAN_ROWS = 20


def kind(v):
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, (datetime, date, time, pd.Timestamp)):
        return "date"
    return "text"


def guess_header_row(ws):
    """0-based index of the densest row among the first rows (header may not be row 1)."""
    counts = [
        sum(c is not None for c in row)
        for row in ws.iter_rows(max_row=HEADER_SCAN_ROWS, values_only=True)
    ]
    return max(range(len(counts)), key=counts.__getitem__) if counts else 0


def main(path):
    wb = openpyxl.load_workbook(path, data_only=True)
    print(f"File: {path}")
    print(f"Sheets: {len(wb.sheetnames)}\n")

    for ws in wb.worksheets:
        print("=" * 80)
        print(f"Sheet: {ws.title!r}  state={ws.sheet_state}")
        print(f"  used range: {ws.dimensions}  (rows={ws.max_row}, cols={ws.max_column})")
        print(f"  merged ranges: {len(ws.merged_cells.ranges)}")
        if ws.max_row <= 1 and ws.max_column <= 1:
            print("  (empty)\n")
            continue

        header = guess_header_row(ws)
        print(f"  guessed header row (1-based): {header + 1}")

        df = pd.read_excel(path, sheet_name=ws.title, header=header)
        print(f"  data rows: {len(df)}  columns: {df.shape[1]}")
        print(f"  fully empty rows: {int(df.isna().all(axis=1).sum())}\n")

        stats = []
        for col in df.columns:
            s = df[col]
            nn = s.dropna()
            kinds = Counter(kind(v) for v in nn)
            text = nn[nn.map(kind) == "text"].astype(str)
            stats.append({
                "column": str(col),
                "dtype": str(s.dtype),
                "kinds": ",".join(f"{k}:{n}" for k, n in kinds.most_common()),
                "non_null": len(nn),
                "null_%": round(100 * s.isna().mean(), 1),
                "nunique": nn.nunique(),
                "unique": bool(len(nn) and nn.is_unique),
                "max_len": int(text.str.len().max()) if len(text) else None,
            })
        with pd.option_context("display.max_rows", None, "display.width", 200,
                               "display.max_colwidth", 50):
            print(pd.DataFrame(stats).to_string(index=False))
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH)
