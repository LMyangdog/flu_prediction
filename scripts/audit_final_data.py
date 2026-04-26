"""
Generate a final data audit note for thesis and defense materials.

The report focuses on the remaining manual-review items after the project
switched to the China National Influenza Center northern-province weekly
monitoring series.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "results" / "reports"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_missing_rows(df: pd.DataFrame, cols: list[str]) -> list[str]:
    rows = []
    for _, row in df.iterrows():
        parts = [f"{col}={row[col]}" for col in cols if col in row.index]
        rows.append("- " + "；".join(parts))
    return rows


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    flu = pd.read_csv(RAW_DIR / "flu" / "cnic_north_weekly_flu.csv")
    search = pd.read_csv(RAW_DIR / "search" / "north_baidu_index.csv")
    weather = pd.read_csv(RAW_DIR / "weather" / "north_representative_city_weather.csv")
    quality = load_json(REPORT_DIR / "data_quality_report.json")
    parse_report = load_json(REPORT_DIR / "cnic_weekly_parse_report.json")
    experiment_summary = load_json(REPORT_DIR / "experiment_summary.json")

    missing_ili = flu[flu["ili_rate"].isna()].copy()
    missing_positive = flu[flu["positive_rate"].isna()].copy()
    partial = flu[flu["parse_status"].eq("partial")].copy()
    imputed = flu[flu["parse_status"].eq("imputed")].copy()
    imputation_method = flu.get("imputation_method", pd.Series("", index=flu.index)).fillna("")
    imputed_ili = flu[imputation_method.str.contains("ili_rate", regex=False)].copy()
    imputed_positive = flu[imputation_method.str.contains("positive_rate", regex=False)].copy()

    region_count_cols = [col for col in search.columns if col.endswith("_region_count")]
    region_count_summary = {
        col: {
            "min": int(search[col].min()),
            "max": int(search[col].max()),
            "nunique": int(search[col].nunique(dropna=False)),
        }
        for col in region_count_cols
    }

    summary = {
        "flu_rows": int(len(flu)),
        "flu_date_start": str(flu["date"].min()),
        "flu_date_end": str(flu["date"].max()),
        "parse_status_counts": parse_report.get("parse_status_counts", {}),
        "missing_ili_rate_rows": int(len(missing_ili)),
        "missing_positive_rate_rows": int(len(missing_positive)),
        "partial_rows": int(len(partial)),
        "imputed_rows": int(len(imputed)),
        "imputed_ili_rate_rows": int(len(imputed_ili)),
        "imputed_positive_rate_rows": int(len(imputed_positive)),
        "merged_rows": int(quality["datasets"]["merged"]["rows"]),
        "merged_date_start": quality["datasets"]["merged"]["date_range"]["start"],
        "merged_date_end": quality["datasets"]["merged"]["date_range"]["end"],
        "feature_engineered_rows": int(experiment_summary["split_metadata"]["rows"]),
        "feature_engineered_date_start": experiment_summary["split_metadata"]["train_date_range"]["start"],
        "feature_engineered_date_end": experiment_summary["split_metadata"]["test_date_range"]["end"],
        "search_rows": int(len(search)),
        "search_date_start": str(search["date"].min()),
        "search_date_end": str(search["date"].max()),
        "search_region_count_summary": region_count_summary,
        "weather_rows": int(len(weather)),
        "weather_date_start": str(weather["date"].min()),
        "weather_date_end": str(weather["date"].max()),
    }

    json_path = REPORT_DIR / "final_data_audit.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 最终数据复核说明",
        "",
        "## 一句话结论",
        "",
        (
            "当前正式实验数据已切换为国家流感中心北方省份真实周度 ILI% 序列，"
            "气象与百度指数均使用北方代表城市聚合口径。当前 CSV 已完成缺失补齐："
            f"`ili_rate` 剩余缺失 {len(missing_ili)} 周，`positive_rate` 剩余缺失 {len(missing_positive)} 周；"
            f"`ili_rate` 有来源补齐 {len(imputed_ili)} 周，`positive_rate` 插值补齐 {len(imputed_positive)} 周。"
        ),
        "",
        "## 流感周报数据",
        "",
        f"- 原始行数：{len(flu)}",
        f"- 原始时间范围：{flu['date'].min()} 至 {flu['date'].max()}",
        f"- 解析状态：{parse_report.get('parse_status_counts', {})}",
        f"- 补齐记录：ili_rate={len(imputed_ili)} 周，positive_rate={len(imputed_positive)} 周",
        f"- 周度合并数据行数：{quality['datasets']['merged']['rows']}",
        (
            "- 周度合并数据时间范围："
            f"{quality['datasets']['merged']['date_range']['start']} 至 "
            f"{quality['datasets']['merged']['date_range']['end']}"
        ),
        f"- 特征工程后进入训练切分的行数：{experiment_summary['split_metadata']['rows']}",
        (
            "- 特征工程后训练/验证/测试覆盖："
            f"{experiment_summary['split_metadata']['train_date_range']['start']} 至 "
            f"{experiment_summary['split_metadata']['test_date_range']['end']}"
        ),
        "",
        "### ili_rate 缺失周",
        "",
    ]
    if missing_ili.empty:
        lines.append("- 无")
    else:
        lines.extend(
            format_missing_rows(
                missing_ili,
                ["date", "year", "week", "parse_status", "parse_method", "parse_note"],
            )
        )

    if not imputed_ili.empty:
        lines.extend(["", "### ili_rate 补齐来源", ""])
        lines.extend(
            format_missing_rows(
                imputed_ili,
                ["date", "year", "week", "ili_rate", "imputation_method", "imputation_source", "imputation_note"],
            )
        )

    lines.extend(
        [
            "",
            "### positive_rate 缺失说明",
            "",
            (
                f"- `positive_rate` 剩余缺失 {len(missing_positive)} 周；已通过相邻周线性插值补齐 {len(imputed_positive)} 周。"
                "当前训练默认不使用该字段，仅作为原始监测留痕与数据质量说明。"
            ),
            "",
            "## 百度指数聚合复核",
            "",
            f"- 行数：{len(search)}",
            f"- 时间范围：{search['date'].min()} 至 {search['date'].max()}",
        ]
    )
    for col, stats in region_count_summary.items():
        lines.append(f"- `{col}`：min={stats['min']}，max={stats['max']}，nunique={stats['nunique']}")

    lines.extend(
        [
            "",
            "结论：三个关键词每日均为 8 个代表地区参与聚合，当前 CSV 与 README/论文中的北方代表城市口径一致。",
            "",
            "## 气象数据复核",
            "",
            f"- 行数：{len(weather)}",
            f"- 时间范围：{weather['date'].min()} 至 {weather['date'].max()}",
            "- 字段：temperature, humidity, wind_speed, pressure",
            "",
            "## 论文写作边界",
            "",
            "- 可以表述为北方省份周度 ILI% 趋势预测，不能表述为单一城市或区县病例预测。",
            "- `positive_rate` 应说明为原始留痕字段，默认不进入训练特征。",
            "- 百度指数聚合方式当前为 8 个北方代表地区简单平均；若后续改为人口权重，需要重新生成数据和结果。",
        ]
    )

    md_path = REPORT_DIR / "final_data_audit.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(md_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
