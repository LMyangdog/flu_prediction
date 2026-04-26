"""
iTransformer 多源特征消融实验。

用法：
    python scripts/run_ablation.py
    python scripts/run_ablation.py --debug
    python scripts/run_ablation.py --skip-collect
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

from scripts.train import collect_data, feature_engineering, load_config, set_random_seed
from src.data.dataset import create_dataloaders
from src.data.preprocessor import DataPreprocessor
from src.models.itransformer import build_itransformer
from src.training.trainer import Trainer


matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def available(cols, df):
    return [col for col in cols if col in df.columns]


def build_ablation_sets(df: pd.DataFrame, config: dict) -> dict:
    target_col = config.get("features", {}).get("target_col", "ili_rate")
    flu_cols = config.get("features", {}).get("flu_cols", [target_col, "positive_rate"])

    flu_features = [
        *flu_cols,
        "week_sin",
        "week_cos",
        "month_sin",
        "month_cos",
        "is_flu_season",
        f"{target_col}_lag1",
        f"{target_col}_lag2",
        f"{target_col}_lag4",
        f"{target_col}_roll_mean_4",
        f"{target_col}_roll_std_4",
        f"{target_col}_roll_mean_8",
        f"{target_col}_roll_std_8",
    ]
    weather_features = [
        "temperature",
        "humidity",
        "wind_speed",
        "pressure",
        "temp_humidity_interaction",
        "comfort_index",
        "wind_chill",
    ]
    search_features = [
        "flu_search_index",
        "cold_search_index",
        "fever_search_index",
        "flu_search_index_pct_change",
        "flu_search_index_acceleration",
        "cold_search_index_pct_change",
        "cold_search_index_acceleration",
        "fever_search_index_pct_change",
        "fever_search_index_acceleration",
    ]

    sets = {
        "flu_only": {
            "label": "仅流感历史",
            "description": "目标序列、阳性率、季节项、滞后项和滚动统计。",
            "features": flu_features,
        },
        "flu_weather": {
            "label": "流感+气象",
            "description": "在流感历史特征基础上加入温度、湿度、风速、气压和气象交互项。",
            "features": flu_features + weather_features,
        },
        "flu_search": {
            "label": "流感+搜索",
            "description": "在流感历史特征基础上加入百度指数及其变化率、加速度。",
            "features": flu_features + search_features,
        },
        "all_sources": {
            "label": "三源融合",
            "description": "同时使用流感、气象和搜索指数特征。",
            "features": flu_features + weather_features + search_features,
        },
    }

    for spec in sets.values():
        spec["features"] = available(spec["features"], df)
    return sets


def train_one_ablation(base_config: dict, df_featured: pd.DataFrame, name: str, spec: dict) -> dict:
    cfg = copy.deepcopy(base_config)
    cfg.setdefault("features", {})["include_feature_cols"] = spec["features"]
    cfg.setdefault("data", {})["splits_dir"] = os.path.join("data", "splits", "ablation", name)
    cfg.setdefault("training", {})["checkpoint_dir"] = os.path.join("checkpoints", "ablation")
    cfg.setdefault("training", {})["log_dir"] = os.path.join("results", "logs", "ablation")

    set_random_seed(cfg)
    preprocessor = DataPreprocessor(cfg)
    splits = preprocessor.process(df_featured)
    X_train, y_train, X_val, y_val, X_test, y_test = splits

    train_loader, val_loader, test_loader = create_dataloaders(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        batch_size=cfg.get("training", {}).get("batch_size", 32),
        num_workers=cfg.get("training", {}).get("num_workers", 0),
    )

    target_col = cfg.get("features", {}).get("target_col", "ili_cases")
    model = build_itransformer(cfg, X_train.shape[1])
    trainer = Trainer(model, cfg, model_name=f"iTransformer_{name}")
    history = trainer.train(train_loader, val_loader)
    metrics, _, _, horizon_metrics = trainer.evaluate(test_loader, preprocessor.scalers.get(target_col))

    return {
        "name": name,
        "label": spec["label"],
        "description": spec["description"],
        "feature_count": len(preprocessor.feature_cols),
        "feature_cols": preprocessor.feature_cols,
        "best_epoch": int(np.argmin(history["val_loss"]) + 1),
        "best_val_loss": float(np.min(history["val_loss"])),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "horizon_metrics": {key: float(value) for key, value in horizon_metrics.items()},
    }


def save_ablation_outputs(results: list, reports_dir: str, figures_dir: str) -> None:
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    summary = {
        "model": "iTransformer",
        "purpose": "比较流感历史、气象、搜索指数三类数据源对预测效果的贡献。",
        "interpretation_note": (
            "当前研究口径已切换为国家流感中心北方省份真实周度监测序列。"
            "若输入数据仍未完成 PDF 解析或搜索指数聚合，消融结果应先作为数据迁移阶段结果解读。"
        ),
        "experiments": results,
    }

    json_path = os.path.join(reports_dir, "ablation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    rows = []
    for item in results:
        row = {
            "experiment": item["name"],
            "label": item["label"],
            "feature_count": item["feature_count"],
            "best_epoch": item["best_epoch"],
            "best_val_loss": item["best_val_loss"],
        }
        row.update(item["metrics"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(reports_dir, "ablation_metrics.csv"), index=False, encoding="utf-8-sig")

    sorted_results = sorted(results, key=lambda item: item["metrics"].get("RMSE", float("inf")))
    md_lines = [
        "# 消融实验报告",
        "",
        "## 实验目的",
        "",
        "本实验固定模型为 iTransformer，通过改变输入特征组合，观察流感历史特征、气象特征和搜索指数特征对 4 周预测任务的贡献。",
        "",
        "## 结果汇总",
        "",
        "| 排名 | 实验 | 特征数 | RMSE | MAE | MAPE | R2 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, item in enumerate(sorted_results, start=1):
        m = item["metrics"]
        md_lines.append(
            f"| {idx} | {item['label']} | {item['feature_count']} | "
            f"{m.get('RMSE', 0):.3f} | {m.get('MAE', 0):.3f} | "
            f"{m.get('MAPE', 0):.2f}% | {m.get('R2', 0):.3f} |"
        )

    md_lines.extend(
        [
            "",
            "## 论文表述建议",
            "",
            f"在当前北方地区真实周度监测数据版本下，`{sorted_results[0]['label']}` 的 RMSE 最低。"
            "该结果可用于讨论不同外生数据源对短期预测误差的影响，但仍需结合数据完整性报告和测试区间流行季背景解释。",
            "",
            "## 实验边界",
            "",
            "- 本消融只固定 iTransformer，不比较不同模型结构。",
            "- 若搜索指数或气象数据仍处于代表城市聚合方案，需在论文中说明聚合方法。",
            "- 每次更新国家流感中心周报数据后，应复跑本脚本并更新论文表格。",
        ]
    )
    with open(os.path.join(reports_dir, "ablation_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    labels = [item["label"] for item in results]
    metrics = ["RMSE", "MAE", "MAPE", "R2"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, metric in zip(axes, metrics):
        values = [item["metrics"].get(metric, 0) for item in results]
        bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        best_idx = int(np.argmax(values) if metric == "R2" else np.argmin(values))
        bars[best_idx].set_edgecolor("#D4AF37")
        bars[best_idx].set_linewidth(3)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("iTransformer 多源特征消融实验")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "ablation_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="运行 iTransformer 多源特征消融实验")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--skip-collect", action="store_true", help="使用已有 merged_dataset.csv")
    parser.add_argument("--debug", action="store_true", help="快速调试模式")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.debug:
        config["training"]["epochs"] = 5
        config["training"]["patience"] = 3
        print("[Debug 模式] epochs=5, patience=3")

    processed_dir = config.get("data", {}).get("processed_dir", "data/processed")
    merged_path = os.path.join(processed_dir, "merged_dataset.csv")
    if args.skip_collect and os.path.exists(merged_path):
        df = pd.read_csv(merged_path)
    else:
        df = collect_data(config)

    df_featured = feature_engineering(config, df)
    ablation_sets = build_ablation_sets(df_featured, config)
    results = []

    for name, spec in ablation_sets.items():
        print("\n" + "=" * 60)
        print(f"消融实验: {spec['label']} ({name})")
        print("=" * 60)
        results.append(train_one_ablation(config, df_featured, name, spec))

    save_ablation_outputs(
        results,
        config.get("reporting", {}).get("reports_dir", "results/reports"),
        config.get("evaluation", {}).get("figures_dir", "results/figures"),
    )
    print("\n消融实验完成:")
    print("  - results/reports/ablation_summary.json")
    print("  - results/reports/ablation_metrics.csv")
    print("  - results/reports/ablation_report.md")
    print("  - results/figures/ablation_comparison.png")


if __name__ == "__main__":
    main()
