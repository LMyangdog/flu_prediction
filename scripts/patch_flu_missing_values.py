"""
Patch residual missing values in the CNIC northern-province weekly flu CSV.

The official CNIC HTML/PDF parser leaves a small number of historical fields
empty because several weekly pages either omit the ILI% paragraph or expose a
broken PDF link. This script fills those residual gaps with explicit lineage:

- `ili_rate` is filled only when a later/neighboring CNIC weekly report gives a
  same-week or previous-week value.
- `positive_rate` is retained as an audit field only; remaining gaps are filled
  by linear time interpolation and marked as such. It is not used for training.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
FLU_PATH = ROOT / "data" / "raw" / "flu" / "cnic_north_weekly_flu.csv"
PARSE_REPORT_PATH = ROOT / "results" / "reports" / "cnic_weekly_parse_report.json"
BACKUP_PATH = ROOT / "data" / "raw" / "flu" / "cnic_north_weekly_flu.before_imputation.csv"


ILI_PATCHES = {
    (2017, 41): {
        "ili_rate": 2.4,
        "source": "CNIC 2020 week 41 report: 2017 same-period northern ILI% = 2.4%",
        "url": "https://ivdc.chinacdc.cn/cnic/zyzx/lgzb/202010/t20201018_222148.htm",
    },
    (2017, 42): {
        "ili_rate": 2.6,
        "source": "CNIC 2017 week 43 report: previous-week northern ILI% = 2.6%",
        "url": "https://ivdc.chinacdc.cn/cnic/zyzx/lgzb/201711/t20171107_154704.htm",
    },
    (2021, 50): {
        "ili_rate": 3.5,
        "source": "CNIC 2022/2023 week 50 reports: 2021 same-period northern ILI% = 3.5%",
        "url": "https://ivdc.chinacdc.cn/cnic/zyzx/lgzb/202212/t20221222_263091.htm",
    },
}


def append_note(old_note: object, new_note: str) -> str:
    old = "" if pd.isna(old_note) else str(old_note).strip()
    if not old:
        return new_note
    if new_note in old:
        return old
    return f"{old}; {new_note}"


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["imputation_method", "imputation_source", "imputation_note"]:
        if col not in df.columns:
            df[col] = ""
    return df


def write_parse_report(df: pd.DataFrame) -> None:
    PARSE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "rows": int(len(df)),
        "date_range": {
            "start": None if df.empty else str(df["date"].min()),
            "end": None if df.empty else str(df["date"].max()),
        },
        "parse_status_counts": df["parse_status"].value_counts(dropna=False).to_dict() if not df.empty else {},
        "missing_counts": df.isna().sum().astype(int).to_dict() if not df.empty else {},
        "output_path": str(FLU_PATH.relative_to(ROOT)),
        "imputation_counts": {
            "ili_rate": int(df["imputation_method"].fillna("").str.contains("ili_rate").sum()),
            "positive_rate": int(df["imputation_method"].fillna("").str.contains("positive_rate").sum()),
        },
    }
    PARSE_REPORT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    if not FLU_PATH.exists():
        raise FileNotFoundError(f"Missing flu CSV: {FLU_PATH}")

    if not BACKUP_PATH.exists():
        shutil.copy2(FLU_PATH, BACKUP_PATH)

    df = pd.read_csv(FLU_PATH)
    df = ensure_columns(df)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    ili_patched = 0
    for (year, week), patch in ILI_PATCHES.items():
        mask = df["year"].eq(year) & df["week"].eq(week)
        if not mask.any():
            raise ValueError(f"Cannot find row year={year}, week={week}")
        if df.loc[mask, "ili_rate"].isna().any():
            df.loc[mask, "ili_rate"] = patch["ili_rate"]
            ili_patched += int(mask.sum())
        df.loc[mask, "parse_status"] = "imputed"
        df.loc[mask, "parse_note"] = df.loc[mask, "parse_note"].apply(
            lambda note: append_note(note, f"filled ili_rate from documented CNIC cross-report value ({patch['ili_rate']}%)")
        )
        df.loc[mask, "imputation_method"] = df.loc[mask, "imputation_method"].apply(
            lambda value: append_note(value, "ili_rate_cross_report")
        )
        df.loc[mask, "imputation_source"] = df.loc[mask, "imputation_source"].apply(
            lambda value: append_note(value, patch["url"])
        )
        df.loc[mask, "imputation_note"] = df.loc[mask, "imputation_note"].apply(
            lambda value: append_note(value, patch["source"])
        )

    positive_missing = df["positive_rate"].isna()
    positive_patched = int(positive_missing.sum())
    if positive_patched:
        interpolated = (
            df.sort_values(["year", "week"])["positive_rate"]
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
            .round(4)
        )
        df.loc[positive_missing, "positive_rate"] = interpolated.loc[positive_missing]
        df.loc[positive_missing, "parse_status"] = "imputed"
        df.loc[positive_missing, "parse_note"] = df.loc[positive_missing, "parse_note"].apply(
            lambda note: append_note(note, "filled positive_rate by linear time interpolation for audit completeness; not used in training")
        )
        df.loc[positive_missing, "imputation_method"] = df.loc[positive_missing, "imputation_method"].apply(
            lambda value: append_note(value, "positive_rate_linear_interpolation")
        )
        df.loc[positive_missing, "imputation_source"] = df.loc[positive_missing, "imputation_source"].apply(
            lambda value: append_note(value, "neighboring CNIC positive_rate values")
        )
        df.loc[positive_missing, "imputation_note"] = df.loc[positive_missing, "imputation_note"].apply(
            lambda value: append_note(value, "positive_rate is an audit-only field and is excluded from model features")
        )

    df.to_csv(FLU_PATH, index=False, encoding="utf-8-sig")
    write_parse_report(df)

    print(f"patched_ili_rate_rows={ili_patched}")
    print(f"patched_positive_rate_rows={positive_patched}")
    print(f"remaining_ili_rate_missing={int(df['ili_rate'].isna().sum())}")
    print(f"remaining_positive_rate_missing={int(df['positive_rate'].isna().sum())}")
    print(f"backup={BACKUP_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
