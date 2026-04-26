"""
主训练脚本。

用法：
    python scripts/train.py
    python scripts/train.py --model iTransformer
    python scripts/train.py --skip-collect
    python scripts/train.py --debug
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

from src.data.collector import MultiSourceDataCollector
from src.data.dataset import create_dataloaders
from src.data.feature_engineer import FeatureEngineer
from src.data.preprocessor import DataPreprocessor
from src.models.arima_baseline import ARIMABaseline
from src.models.dlinear_baseline import build_dlinear
from src.models.itransformer import build_itransformer
from src.models.lstm_baseline import build_lstm
from src.training.trainer import Trainer
from src.utils.metrics import compute_all_metrics, compute_horizon_metrics, format_metrics
from src.utils.visualization import Visualizer


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_random_seed(config: dict):
    seed = config.get("training", {}).get("random_seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    print(f"[Seed] random_seed={seed}")


def collect_data(config: dict):
    print("\n" + "=" * 60)
    print("Step 1: 多源数据采集")
    print("=" * 60)

    collector = MultiSourceDataCollector(config)
    return collector.collect_all()


def feature_engineering(config: dict, df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Step 2: 特征工程")
    print("=" * 60)

    engineer = FeatureEngineer(config)
    df_featured = engineer.transform(df)

    processed_dir = config.get("data", {}).get("processed_dir", "data/processed")
    os.makedirs(processed_dir, exist_ok=True)
    df_featured.to_csv(
        os.path.join(processed_dir, "featured_dataset.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return df_featured


def preprocess_data(config: dict, df: pd.DataFrame):
    preprocessor = DataPreprocessor(config)
    splits = preprocessor.process(df)
    return splits, preprocessor


def train_itransformer(config: dict, train_loader, val_loader, test_loader, num_vars: int, scaler=None):
    model = build_itransformer(config, num_vars)
    trainer = Trainer(model, config, model_name="iTransformer")
    history = trainer.train(train_loader, val_loader)
    metrics, preds, actuals, horizon_metrics = trainer.evaluate(test_loader, scaler)
    return model, trainer, history, metrics, preds, actuals, horizon_metrics


def train_lstm(config: dict, train_loader, val_loader, test_loader, num_vars: int, scaler=None):
    model = build_lstm(config, num_vars)
    trainer = Trainer(model, config, model_name="LSTM")
    history = trainer.train(train_loader, val_loader)
    metrics, preds, actuals, horizon_metrics = trainer.evaluate(test_loader, scaler)
    return model, trainer, history, metrics, preds, actuals, horizon_metrics


def train_dlinear(config: dict, train_loader, val_loader, test_loader, num_vars: int, scaler=None):
    model = build_dlinear(config, num_vars)
    trainer = Trainer(model, config, model_name="DLinear")
    history = trainer.train(train_loader, val_loader)
    metrics, preds, actuals, horizon_metrics = trainer.evaluate(test_loader, scaler)
    return model, trainer, history, metrics, preds, actuals, horizon_metrics


def run_arima(config: dict, df: pd.DataFrame, split_metadata: dict):
    print(f"\n{'=' * 60}")
    print("ARIMA 基准模型")
    print("=" * 60)

    target_col = config.get("features", {}).get("target_col", "ili_cases")
    series = df[target_col].values

    train_size = int(split_metadata["train_rows"] + split_metadata["val_rows"])
    lookback = config.get("data", {}).get("lookback_window", 16)
    horizon = config.get("data", {}).get("forecast_horizon", 4)

    preds, actuals = [], []
    n_windows = len(series) - train_size - lookback - horizon + 1
    if n_windows <= 0:
        print("[ARIMA] 测试集窗口不足，无法与深度模型对齐评估")
        return {}, np.array([]), np.array([]), {}

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    initial_end = train_size + lookback
    try:
        model = SARIMAX(
            series[:initial_end],
            order=(2, 1, 2),
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=100)
    except Exception as exc:
        print(f"[ARIMA] 初始拟合失败: {exc}")
        return {}, np.array([]), np.array([]), {}

    for i in range(n_windows):
        context_end = train_size + i + lookback
        actual = series[context_end:context_end + horizon]
        try:
            pred = np.array(fitted.forecast(steps=horizon))
        except Exception:
            print(f"[ARIMA] 对齐窗口 {i} 预测失败，跳过该窗口。")
            continue
        preds.append(pred)
        actuals.append(actual)
        if i < n_windows - 1:
            fitted = fitted.append([series[context_end]], refit=False)

    preds = np.array(preds)
    actuals = np.array(actuals)

    if len(preds) == 0:
        print("[ARIMA] 预测失败")
        return {}, np.array([]), np.array([]), {}

    preds_flat = preds.flatten()
    actuals_flat = actuals.flatten()
    metrics = compute_all_metrics(actuals_flat, preds_flat)
    horizon_metrics = compute_horizon_metrics(actuals, preds)

    print(f"\n[ARIMA] 测试集评估结果:")
    print(format_metrics(metrics))
    return metrics, preds_flat, actuals_flat, horizon_metrics


def build_prediction_index(df: pd.DataFrame, split_metadata: dict, lookback: int, horizon: int) -> pd.DataFrame:
    """构造扁平化多步预测结果对应的样本、horizon 与目标日期。"""
    train_rows = int(split_metadata["train_rows"])
    val_rows = int(split_metadata["val_rows"])
    test_start = train_rows + val_rows
    df_test = df.iloc[test_start:].reset_index(drop=True).copy()

    rows = []
    n_windows = len(df_test) - lookback - horizon + 1
    for sample_idx in range(max(n_windows, 0)):
        anchor_date = pd.to_datetime(df_test.loc[sample_idx + lookback - 1, "date"]).date()
        for step in range(horizon):
            target_idx = sample_idx + lookback + step
            row = {
                "sample_index": sample_idx,
                "horizon": step + 1,
                "anchor_date": str(anchor_date),
                "target_date": str(pd.to_datetime(df_test.loc[target_idx, "date"]).date()),
            }
            if "source_type" in df_test.columns:
                row["target_source_type"] = df_test.loc[target_idx, "source_type"]
            rows.append(row)

    return pd.DataFrame(rows)


def summarize_source_types(df: pd.DataFrame) -> dict:
    if "source_type" not in df.columns:
        return {}
    summary = {}
    for source_type, group in df.groupby("source_type", dropna=False):
        dates = pd.to_datetime(group["date"])
        summary[str(source_type)] = {
            "rows": int(len(group)),
            "start": str(dates.min().date()),
            "end": str(dates.max().date()),
        }
    return summary


def write_experiment_brief(summary: dict, reports_dir: str) -> str:
    metrics = summary.get("metrics", {})
    split = summary.get("split_metadata", {})
    source_types = summary.get("source_type_summary", {})
    target_col = summary.get("target_col", "ili_rate")
    best_model = "表现最优模型"

    ranking = sorted(metrics.items(), key=lambda item: item[1].get("RMSE", float("inf")))
    if ranking:
        best_model = ranking[0][0]

    lines = [
        "# 实验结果与答辩说明",
        "",
        "## 一句话结论",
        "",
        "当前实验已完成国家流感中心北方省份周度流感监测序列、北方代表城市气象数据与百度指数聚合序列的三源对齐、特征工程、iTransformer/LSTM/DLinear/ARIMA 对比与多步预测评估。测试区间采用真实周度 ILI% 序列，结果可作为当前北方地区研究口径下的模型对比依据。",
        "",
        "## 数据口径",
        "",
        f"- 目标变量：`{target_col}`，即北方省份哨点医院报告的 ILI%。",
        "- 辅助流感字段：`positive_rate` 保留在原始表和质量报告中；由于部分年份 PDF 表格抽取不稳定，默认训练暂不使用。",
        "- 研究口径：中国国家流感中心北方省份周度流感活动趋势。",
        "- 外生变量：北方代表城市气象数据与百度指数聚合序列。",
        f"- 训练集：{split.get('train_date_range', {}).get('start')} 至 {split.get('train_date_range', {}).get('end')}。",
        f"- 验证集：{split.get('val_date_range', {}).get('start')} 至 {split.get('val_date_range', {}).get('end')}。",
        f"- 测试集：{split.get('test_date_range', {}).get('start')} 至 {split.get('test_date_range', {}).get('end')}。",
        "",
        "## 数据来源分段",
        "",
    ]

    if source_types:
        for source_type, info in source_types.items():
            lines.append(f"- `{source_type}`：{info['rows']} 周，{info['start']} 至 {info['end']}。")
    else:
        lines.append("- 当前合并数据未保留 `source_type` 字段。")

    lines.extend(["", "## 模型对比结果", ""])
    if ranking:
        lines.append("| 排名 | 模型 | RMSE | MAE | MAPE | R2 |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for idx, (model_name, model_metrics) in enumerate(ranking, start=1):
            lines.append(
                f"| {idx} | {model_name} | {model_metrics.get('RMSE', 0):.3f} | "
                f"{model_metrics.get('MAE', 0):.3f} | {model_metrics.get('MAPE', 0):.2f}% | "
                f"{model_metrics.get('R2', 0):.3f} |"
            )

    lines.extend(
        [
            "",
            "## 答辩表述建议",
            "",
            "- 可以说：本项目基于公开可追溯的国家流感中心周报构建了北方省份周度 ILI% 预测链路，完成了严格时间切分、多模型对比和多步预测评估。",
            f"- 可以说：在当前北方地区真实周度监测数据版本下，{best_model} 的整体 RMSE 最低；同时需要说明该排序依赖当前数据版本、特征聚合方法与时间切分。",
            "- 不要说：模型已经证明可以准确预测某个城市或区县的真实流感病例。",
            "- 不要说：搜索指数或气象变量一定提升预测性能；消融实验显示外生变量贡献需要结合数据聚合噪声和流行季背景解释。",
            "",
            "## 后续优先级",
            "",
            "1. 复核 `cnic_weekly_parse_report.json` 与 `final_data_audit.md` 中的 `imputed` 记录，确认论文中的补齐来源、插值字段说明与最终训练数据一致。",
            "2. 若后续更新国家流感中心周报或百度指数聚合文件，应重新运行训练、消融实验并更新图表。",
            "3. 在论文中保留“数据来源与可信性”“PDF 解析质量”“多源变量聚合口径”“消融实验解释”四部分说明。",
        ]
    )

    path = os.path.join(reports_dir, "experiment_brief.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def serialize_metric_dict(metrics_dict: dict) -> dict:
    serialized = {}
    for model_name, metrics in metrics_dict.items():
        serialized[model_name] = {
            key: float(value) if isinstance(value, (np.floating, float)) else int(value)
            for key, value in metrics.items()
        }
    return serialized


def main():
    parser = argparse.ArgumentParser(description="流感预测模型训练")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路径")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "iTransformer", "LSTM", "DLinear", "ARIMA"],
        help="训练的模型",
    )
    parser.add_argument("--skip-collect", action="store_true", help="跳过数据采集")
    parser.add_argument("--debug", action="store_true", help="Debug 模式")
    args = parser.parse_args()

    config = load_config(args.config)
    set_random_seed(config)

    if args.debug:
        config["training"]["epochs"] = 5
        config["training"]["patience"] = 3
        print("[Debug 模式] epochs=5, patience=3")

    processed_dir = config.get("data", {}).get("processed_dir", "data/processed")
    merged_path = os.path.join(processed_dir, "merged_dataset.csv")

    if args.skip_collect and os.path.exists(merged_path):
        print("[跳过] 使用已有数据")
        df = pd.read_csv(merged_path)
    else:
        df = collect_data(config)

    df_featured = feature_engineering(config, df)
    splits, preprocessor = preprocess_data(config, df_featured)
    X_train, y_train, X_val, y_val, X_test, y_test = splits

    num_vars = X_train.shape[1]
    feature_cols = preprocessor.feature_cols
    target_col = config.get("features", {}).get("target_col", "ili_cases")
    target_scaler = preprocessor.scalers.get(target_col)
    print(f"\n输入变量数: {num_vars}")

    batch_size = config.get("training", {}).get("batch_size", 32)
    num_workers = config.get("training", {}).get("num_workers", 0)
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    all_metrics = {}
    all_horizon_metrics = {}
    all_preds = {}
    all_actuals = None

    visualizer = Visualizer(config.get("evaluation", {}).get("figures_dir", "results/figures"))
    reports_dir = config.get("reporting", {}).get("reports_dir", "results/reports")
    os.makedirs(reports_dir, exist_ok=True)

    visualizer.plot_data_overview(df)
    visualizer.plot_correlation_matrix(df_featured, feature_cols[: min(12, len(feature_cols))])

    if args.model in ["all", "iTransformer"]:
        model, trainer, history, metrics, preds, actuals, horizon_metrics = train_itransformer(
            config, train_loader, val_loader, test_loader, num_vars, target_scaler
        )
        all_metrics["iTransformer"] = metrics
        all_horizon_metrics["iTransformer"] = horizon_metrics
        all_preds["iTransformer"] = preds
        all_actuals = actuals

        visualizer.plot_training_history(history, "iTransformer")
        visualizer.plot_predictions(actuals, preds, "iTransformer")

        try:
            import torch

            model.eval()
            with torch.no_grad():
                sample_x = next(iter(test_loader))[0].to(next(model.parameters()).device)
                _ = model(sample_x)
                attn_weights = model.get_attention_weights()
                if attn_weights:
                    avg_attn = attn_weights[-1].mean(dim=0).numpy()
                    short_names = [name[:12] for name in feature_cols[:num_vars]]
                    visualizer.plot_attention_heatmap(avg_attn, short_names)
        except Exception as exc:
            print(f"[注意力可视化失败] {exc}")

    if args.model in ["all", "LSTM"]:
        _, _, history, metrics, preds, actuals, horizon_metrics = train_lstm(
            config, train_loader, val_loader, test_loader, num_vars, target_scaler
        )
        all_metrics["LSTM"] = metrics
        all_horizon_metrics["LSTM"] = horizon_metrics
        all_preds["LSTM"] = preds
        if all_actuals is None:
            all_actuals = actuals

        visualizer.plot_training_history(history, "LSTM")
        visualizer.plot_predictions(actuals, preds, "LSTM")

    if args.model in ["all", "DLinear"]:
        _, _, history, metrics, preds, actuals, horizon_metrics = train_dlinear(
            config, train_loader, val_loader, test_loader, num_vars, target_scaler
        )
        all_metrics["DLinear"] = metrics
        all_horizon_metrics["DLinear"] = horizon_metrics
        all_preds["DLinear"] = preds
        if all_actuals is None:
            all_actuals = actuals

        visualizer.plot_training_history(history, "DLinear")
        visualizer.plot_predictions(actuals, preds, "DLinear")

    if args.model in ["all", "ARIMA"]:
        metrics, preds, actuals_arima, horizon_metrics = run_arima(config, df_featured, preprocessor.split_metadata)
        if metrics:
            all_metrics["ARIMA"] = metrics
            all_horizon_metrics["ARIMA"] = horizon_metrics
            all_preds["ARIMA"] = preds
            if all_actuals is None:
                all_actuals = actuals_arima

    if len(all_metrics) > 1:
        print(f"\n{'=' * 60}")
        print("模型对比汇总")
        print("=" * 60)

        for model_name, metrics in all_metrics.items():
            print(f"\n>>> {model_name}")
            print(format_metrics(metrics))

        visualizer.plot_model_comparison(all_metrics)

        if all_actuals is not None:
            min_len = min(len(all_actuals), min(len(pred) for pred in all_preds.values() if len(pred) > 0))
            trimmed_preds = {name: pred[:min_len] for name, pred in all_preds.items() if len(pred) > 0}
            visualizer.plot_multi_model_predictions(all_actuals[:min_len], trimmed_preds)

    results_dir = config.get("evaluation", {}).get("figures_dir", "results/figures")
    serializable_metrics = serialize_metric_dict(all_metrics)
    serializable_horizon_metrics = serialize_metric_dict(all_horizon_metrics)

    with open(os.path.join(results_dir, "all_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(serializable_metrics, f, ensure_ascii=False, indent=2)

    with open(os.path.join(reports_dir, "horizon_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(serializable_horizon_metrics, f, ensure_ascii=False, indent=2)

    experiment_summary = {
        "config_path": args.config,
        "model": args.model,
        "random_seed": config.get("training", {}).get("random_seed", 42),
        "target_col": target_col,
        "feature_cols": feature_cols,
        "num_variables": int(num_vars),
        "split_metadata": preprocessor.split_metadata,
        "source_type_summary": summarize_source_types(df_featured),
        "interpretation_note": (
            "当前研究口径为中国国家流感中心北方省份真实周度 ILI% 监测序列。"
            "指标可用于该北方地区口径下的模型对比；不应外推为单一城市或区县病例预测结论。"
        ),
        "metrics": serializable_metrics,
        "horizon_metrics": serializable_horizon_metrics,
    }
    with open(os.path.join(reports_dir, "experiment_summary.json"), "w", encoding="utf-8") as f:
        json.dump(experiment_summary, f, ensure_ascii=False, indent=2)
    brief_path = write_experiment_brief(experiment_summary, reports_dir)

    if all_actuals is not None and all_preds:
        prediction_index = build_prediction_index(
            df_featured,
            preprocessor.split_metadata,
            config.get("data", {}).get("lookback_window", 16),
            config.get("data", {}).get("forecast_horizon", 4),
        )
        export_len = min(len(all_actuals), min(len(pred) for pred in all_preds.values()), len(prediction_index))
        prediction_df = prediction_index.iloc[:export_len].to_dict(orient="list")
        prediction_df["actual"] = all_actuals[:export_len]
        for model_name, pred in all_preds.items():
            prediction_df[model_name] = pred[:export_len]
        pd.DataFrame(prediction_df).to_csv(
            os.path.join(reports_dir, "test_predictions.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    print(f"\n{'=' * 60}")
    print("训练完成！所有结果已保存。")
    print(f"{'=' * 60}")
    print("  模型权重: checkpoints/")
    print(f"  可视化图表: {results_dir}/")
    print("  训练日志: results/logs/")
    print(f"  审计报告: {reports_dir}/")
    print(f"  实验简报: {brief_path}")


if __name__ == "__main__":
    main()
