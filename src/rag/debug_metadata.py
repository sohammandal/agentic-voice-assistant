"""
Debug helper to find CSV rows whose `metadata` field fails to parse by
`VectorStoreService._parse_metadata_from_csv()`.

Usage:
    python -m src.rag.debug_metadata --csv data/homes_preprocessed_data.csv --limit 25

It prints the first N problematic rows with:
 - row index
 - product_id
 - raw metadata string (truncated)
 - parsed result (should be empty dict when parsing failed)

Paste a few of these raw metadata strings here and I'll update the parser to handle them.
"""

import argparse
import textwrap

import pandas as pd

from src.rag.vector_store_service import VectorStoreService


def main():
    parser = argparse.ArgumentParser(description="Debug metadata parsing failures")
    parser.add_argument(
        "--csv",
        type=str,
        default="data/homes_preprocessed_data.csv",
        help="CSV path relative to project root",
    )
    parser.add_argument(
        "--limit", type=int, default=25, help="Maximum number of examples to show"
    )
    args = parser.parse_args()

    svc = VectorStoreService()

    print(f"Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"Rows: {len(df)}")

    problems = []
    for idx, row in df.iterrows():
        raw = row.get("metadata")
        if pd.isna(raw) or raw in ("", "{}"):
            continue

        parsed = svc._parse_metadata_from_csv(raw)
        # If parsing returned empty dict but raw metadata was present → problem
        if isinstance(parsed, dict) and not parsed:
            problems.append((idx, row.get("product_id"), raw))
            if len(problems) >= args.limit:
                break

    if not problems:
        print("No problematic metadata rows found (parser succeeded or metadata empty)")
        return

    print(f"Found {len(problems)} problematic rows (showing up to {args.limit}):\n")

    for i, (idx, pid, raw) in enumerate(problems, 1):
        raw_str = str(raw)
        # Truncate and show escapes
        truncated = (raw_str[:600] + "...") if len(raw_str) > 600 else raw_str
        print(f"{i}. CSV row index: {idx}  product_id: {pid}")
        print("Raw metadata (truncated):")
        print(textwrap.indent(truncated, "    "))
        print("---\n")

    print(
        "\nIf you paste 2-3 of these raw metadata strings here I can update the parser to handle them."
    )


if __name__ == "__main__":
    main()
