# pdf_to_excel.py
# Convert a tabular PDF to a single formatted Excel file using tabula + pandas.
# Requires: Java installed, pip install tabula-py pandas xlsxwriter

import os
import re
import pandas as pd

INPUT_PDF  = r"C:\Users\andre\OneDrive\Desktop\2025-10-24_A303_LDND_status.pdf"
OUTPUT_XLSX = r"C:\Users\andre\OneDrive\Desktop\A303_LDND_status.xlsx"

BANNER_RE = re.compile(
    r"(A/C Actual Compliance|Print Date|Page:|Schedule|Type/Series:|A/C Hours:|A/C Cycles:|As Of:|Remaining/E/C)",
    re.IGNORECASE
)

def drop_banners(df):
    if df is None or df.empty:
        return df
    # Drop fully empty cols, then banner rows
    df = df.dropna(axis=1, how="all")
    if df.empty:
        return df
    mask = []
    for _, row in df.astype(str).iterrows():
        row_text = " ".join(row.tolist())
        mask.append(not bool(BANNER_RE.search(row_text)))
    df = df[mask]
    return df

def promote_header(df):
    if df is None or df.empty:
        return df
    # Try to use first row as header if it looks like headers
    cues = ["description", "next", "due", "hrs", "cys", "dys", "dte", "status", "category", "rev", "e/c", "document", "wo", "a/c"]
    first = " ".join(df.iloc[0].astype(str).tolist()).lower()
    if sum(1 for c in cues if c in first) >= 2:
        df = df.copy()
        df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.iloc[0]]
        df = df.iloc[1:].reset_index(drop=True)
    else:
        df = df.copy()
        df.columns = [f"Col_{i+1}" for i in range(df.shape[1])]
    return df

def light_types(df):
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        low = col.lower()
        if any(k in low for k in ["hrs", "hours", "cycles", "cys", "remaining"]):
            # Keep time-like strings such as 21726:58, coerce other numeric-ish cells
            def conv(x):
                s = str(x).strip()
                if ":" in s:
                    return s
                s2 = re.sub(r"[^\d\.\-]", "", s)
                try:
                    return float(s2) if s2 not in ("", ".", "-") else pd.NA
                except:
                    return x
            out[col] = out[col].map(conv)
    return out

def extract_all_tables(pdf_path):
    import tabula  # requires Java
    all_tables = []

    # Try lattice first, then stream
    for lattice in (True, False):
        try:
            tables = tabula.read_pdf(
                pdf_path,
                pages="all",
                multiple_tables=True,
                lattice=lattice,
                stream=not lattice
            )
            if tables:
                all_tables.extend(tables)
        except Exception:
            continue

    return all_tables

def main():
    if not os.path.exists(INPUT_PDF):
        raise FileNotFoundError(f"Not found: {INPUT_PDF}")

    try:
        tables = extract_all_tables(INPUT_PDF)
    except Exception as e:
        raise SystemExit(
            "Tabula could not run. Make sure Java is installed and on PATH. "
            "Install with: winget install --id EclipseAdoptium.Temurin.17.JDK\n"
            f"Original error: {e}"
        )

    cleaned = []
    for t in tables:
        t = drop_banners(t)
        t = promote_header(t)
        if t is not None and not t.empty:
            cleaned.append(t)

    if not cleaned:
        raise SystemExit("No tables were found. Try Camelot or share a sample page to tweak parsing.")

    # Build unified columns in order of first appearance
    cols_order = []
    seen = set()
    for df in cleaned:
        for c in df.columns:
            if c not in seen:
                cols_order.append(c)
                seen.add(c)

    aligned = [df.reindex(columns=cols_order) for df in cleaned]
    merged = pd.concat(aligned, ignore_index=True)
    merged = light_types(merged)

    # Write formatted Excel
    with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
        sheet = "A303_LDND_status"
        # Write raw content first
        merged.to_excel(writer, index=False, sheet_name=sheet)

        # Basic formatting
        wb  = writer.book
        ws  = writer.sheets[sheet]
        header_fmt = wb.add_format({"bold": True, "border": 1, "valign": "vcenter"})
        cell_fmt   = wb.add_format({"border": 1})

        # Rewrite header row with format
        for col_idx, name in enumerate(merged.columns):
            ws.write(0, col_idx, name, header_fmt)

        # Apply borders to body
        nrows, ncols = merged.shape
        for r in range(1, nrows + 1):
            for c in range(ncols):
                ws.write(r, c, merged.iloc[r - 1, c], cell_fmt)

        # Freeze header, add filter, autosize columns
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, nrows, ncols - 1)
        # Autosize
        cap = 50
        for i, col in enumerate(merged.columns):
            max_len = max([len(str(col))] + [len(str(v)) for v in merged[col].astype(str).tolist()])
            ws.set_column(i, i, min(max_len + 2, cap))

    print(f"Saved: {OUTPUT_XLSX}")

if __name__ == "__main__":
    main()
