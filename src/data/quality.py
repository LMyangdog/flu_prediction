"""
数据质量审计工具。

为毕设提供三类留痕：
    1. 原始数据来源清单校验
    2. 原始/合并数据的质量摘要
    3. 与目标变量的关系检查，辅助识别“伪相关”“常数列”“异常高相关”等问题
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def assert_source_manifest(manifest_path: str, strict: bool = True) -> Dict:
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if strict:
        example_path = manifest_path.replace(".json", ".example.json")
        raise FileNotFoundError(
            f"缺少真实数据来源清单: {manifest_path}。"
            f"请参照 {example_path} 创建后再运行。"
        )
    return {}


def assert_required_columns(df: pd.DataFrame, required_cols: Iterable[str], dataset_name: str):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} 数据缺少必要字段: {missing}")


class DataQualityAuditor:
    def __init__(self, reports_dir: str = "results/reports"):
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)

    def audit_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        required_cols: List[str],
        expected_granularity: str,
    ) -> Dict:
        assert_required_columns(df, required_cols, dataset_name)

        report: Dict[str, object] = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "expected_granularity": expected_granularity,
            "missing_counts": {col: int(df[col].isna().sum()) for col in df.columns},
        }

        if "date" in df.columns:
            dates = pd.to_datetime(df["date"])
            report["date_range"] = {"start": str(dates.min().date()), "end": str(dates.max().date())}
            report["duplicate_dates"] = int(dates.duplicated().sum())
            report["is_chronological"] = bool(dates.is_monotonic_increasing)
        else:
            report["date_range"] = None
            report["duplicate_dates"] = None
            report["is_chronological"] = False

        numeric_df = df.select_dtypes(include=[np.number])
        constant_cols = [col for col in numeric_df.columns if numeric_df[col].nunique(dropna=True) <= 1]
        low_cardinality_cols = [col for col in numeric_df.columns if numeric_df[col].nunique(dropna=True) <= 3]

        report["constant_numeric_columns"] = constant_cols
        report["low_cardinality_numeric_columns"] = low_cardinality_cols
        report["summary_stats"] = {
            col: {
                "mean": None if pd.isna(numeric_df[col].mean()) else float(numeric_df[col].mean()),
                "std": None if pd.isna(numeric_df[col].std()) else float(numeric_df[col].std()),
                "min": None if pd.isna(numeric_df[col].min()) else float(numeric_df[col].min()),
                "max": None if pd.isna(numeric_df[col].max()) else float(numeric_df[col].max()),
                "nunique": int(numeric_df[col].nunique(dropna=True)),
            }
            for col in numeric_df.columns
        }

        return report

    def audit_relationships(self, df: pd.DataFrame, target_col: str = "ili_cases") -> Dict:
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if target_col not in numeric_df.columns:
            return {"target_col": target_col, "warnings": ["target_col 不存在于数据中，无法执行关系检查。"]}

        corr_series = numeric_df.corr(numeric_only=True)[target_col].dropna().sort_values(ascending=False)
        suspicious = {
            col: float(value)
            for col, value in corr_series.items()
            if col != target_col and abs(value) >= 0.95
        }

        warnings = []
        if suspicious:
            warnings.append("存在与目标变量绝对相关系数 >= 0.95 的特征，请核查是否存在派生泄漏、伪造数据或重复编码。")

        return {
            "target_col": target_col,
            "correlations": {col: float(value) for col, value in corr_series.items()},
            "suspicious_high_correlations": suspicious,
            "warnings": warnings,
        }

    def save_report(self, report: Dict, filename: str = "data_quality_report.json") -> str:
        path = os.path.join(self.reports_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return path
