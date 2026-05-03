"""
iTransformer optimization experiments.

This script keeps the strict time split/preprocessing pipeline, then runs two
practical improvement paths:
  1. compact iTransformer trials over history-only/all-source feature sets;
  2. ARIMA + iTransformer residual correction, where ARIMA models the strong
     autoregressive backbone and iTransformer learns the remaining residual.

Usage:
    python scripts/optimize_itransformer.py --skip-collect
    python scripts/optimize_itransformer.py --skip-collect --max-trials 3
    python scripts/optimize_itransformer.py --skip-collect --debug
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

from scripts.train import build_prediction_index, collect_data, feature_engineering, load_config, set_random_seed
from src.data.dataset import FluDataset, create_dataloaders
from src.data.preprocessor import DataPreprocessor
from src.models.itransformer import build_itransformer
from src.training.trainer import Trainer
from src.utils.metrics import compute_all_metrics, compute_horizon_metrics, format_metrics


warnings.filterwarnings("ignore")

FLU_CONTEXT_COLS = ["southern_ili_rate"]
OPTIMIZED_MODEL_NAME = "iTransformer优化"


def available(cols: list[str], df: pd.DataFrame) -> list[str]:
    selected = set(cols)
    return [col for col in df.columns if col in selected]


def numeric_feature_cols(df: pd.DataFrame, config: dict) -> list[str]:
    exclude_cols = set(config.get("features", {}).get("exclude_from_training", ["year", "week"]))
    return [
        col
        for col in df.columns
        if col in df.select_dtypes(include=[np.number]).columns and col not in exclude_cols
    ]


def build_history_features(df: pd.DataFrame, config: dict) -> list[str]:
    target_col = config.get("features", {}).get("target_col", "ili_rate")
    base = [
        target_col,
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
    return available(base, df)


def build_dynamic_history_features(df: pd.DataFrame, config: dict) -> list[str]:
    target_col = config.get("features", {}).get("target_col", "ili_rate")
    dynamic = [
        f"{target_col}_diff1",
        f"{target_col}_pct_change",
        f"{target_col}_acceleration",
        f"{target_col}_slope_4",
        f"{target_col}_above_roll_mean_8",
    ]
    return available([*build_history_features(df, config), *dynamic], df)


def build_cross_region_dynamic_features(df: pd.DataFrame, config: dict) -> list[str]:
    return available([*build_dynamic_history_features(df, config), *FLU_CONTEXT_COLS], df)


def enrich_with_flu_context(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Attach optional CNIC context columns such as southern ILI%, if available."""
    missing = [col for col in FLU_CONTEXT_COLS if col not in df.columns]
    if not missing:
        return df

    manifest_path = Path(config.get("data", {}).get("manifest_path", "data/raw/source_manifest.json"))
    raw_path = Path("data/raw/flu/cnic_north_weekly_flu.csv")
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw_path = Path(manifest.get("flu", {}).get("path", raw_path))
        except Exception:
            pass
    if not raw_path.exists():
        return df

    raw = pd.read_csv(raw_path)
    cols = [col for col in ["date", "year", "week", *missing] if col in raw.columns]
    if not {"date", "year", "week"} <= set(cols):
        return df

    enriched = df.copy()
    enriched["date"] = pd.to_datetime(enriched["date"])
    raw = raw[cols].copy()
    raw["date"] = pd.to_datetime(raw["date"])
    for col in missing:
        if col in raw.columns:
            raw[col] = raw[col].ffill()
    enriched = enriched.merge(raw, on=["date", "year", "week"], how="left")
    attached = [col for col in missing if col in enriched.columns]
    if attached:
        print(f"[优化实验] 已附加 CNIC 跨区域上下文特征: {attached}")
    return enriched


def build_trials(df: pd.DataFrame, config: dict) -> list[dict[str, Any]]:
    history_features = build_history_features(df, config)
    dynamic_history_features = build_dynamic_history_features(df, config)
    cross_region_dynamic_features = build_cross_region_dynamic_features(df, config)
    all_features = numeric_feature_cols(df, config)

    baseline_model = {
        "d_model": 128,
        "n_heads": 8,
        "n_layers": 2,
        "d_ff": 256,
        "dropout": 0.1,
        "activation": "gelu",
    }
    compact_model = {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 128,
        "dropout": 0.15,
        "activation": "gelu",
    }
    regularized_model = {
        "d_model": 96,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 192,
        "dropout": 0.2,
        "activation": "gelu",
    }
    training = {
        "learning_rate": 5e-4,
        "weight_decay": 0.02,
        "scheduler": "plateau",
        "batch_size": 32,
    }
    baseline_training = {
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "scheduler": "cosine",
        "batch_size": 32,
    }
    adamw_training = {
        "optimizer": "adamw",
        "learning_rate": 7e-4,
        "weight_decay": 0.02,
        "scheduler": "cosine",
        "batch_size": 32,
    }
    peak_training = {
        "optimizer": "adamw",
        "learning_rate": 7e-4,
        "weight_decay": 0.02,
        "scheduler": "cosine",
        "batch_size": 32,
        "loss": "peak_weighted_mse",
        "peak_threshold": 0.65,
        "peak_weight": 2.0,
    }
    peak_trend_training = {
        **peak_training,
        "loss": "peak_trend_mse",
        "trend_weight": 0.25,
    }
    mild_peak_training = {
        **baseline_training,
        "loss": "peak_weighted_mse",
        "peak_threshold": 0.8,
        "peak_weight": 0.5,
    }

    return [
        {
            "name": "history_l16_baseline_capacity",
            "label": "历史特征 L16 baseline",
            "features": history_features,
            "lookback": 16,
            "model": baseline_model,
            "training": baseline_training,
            "note": "复现消融实验中最强的历史特征方向，保留原始 iTransformer 容量。",
        },
        {
            "name": "history_l24_baseline_capacity",
            "label": "历史特征 L24 baseline",
            "features": history_features,
            "lookback": 24,
            "model": baseline_model,
            "training": baseline_training,
            "note": "在原始容量下扩大上下文，验证半年历史是否优于 16 周窗口。",
        },
        {
            "name": "history_dynamic_l16_baseline",
            "label": "历史动态 L16 baseline",
            "features": dynamic_history_features,
            "lookback": 16,
            "model": baseline_model,
            "training": baseline_training,
            "note": "在历史特征基础上显式加入差分、加速度和相对滚动均值。",
        },
        {
            "name": "history_dynamic_l16_mild_peak",
            "label": "历史动态 L16 温和峰值加权",
            "features": dynamic_history_features,
            "lookback": 16,
            "model": baseline_model,
            "training": mild_peak_training,
            "note": "使用动态特征，并采用较弱峰值权重，尽量降低 RMSE 损失。",
        },
        {
            "name": "history_dynamic_south_l16",
            "label": "历史动态+南方信号 L16",
            "features": cross_region_dynamic_features,
            "lookback": 16,
            "model": baseline_model,
            "training": baseline_training,
            "note": "在北方历史动态特征基础上加入国家流感中心同周南方 ILI% 上下文。",
        },
        {
            "name": "history_l16_adamw",
            "label": "历史特征 L16 AdamW",
            "features": history_features,
            "lookback": 16,
            "model": baseline_model,
            "training": adamw_training,
            "note": "保持当前最强窗口，换用 AdamW 和更强权重衰减。",
        },
        {
            "name": "history_l16_peak_weighted",
            "label": "历史特征 L16 峰值加权",
            "features": history_features,
            "lookback": 16,
            "model": baseline_model,
            "training": peak_training,
            "note": "对归一化后高于阈值的目标周加权，尝试提升爆发期拟合。",
        },
        {
            "name": "history_l16_peak_trend",
            "label": "历史特征 L16 峰值+趋势",
            "features": history_features,
            "lookback": 16,
            "model": baseline_model,
            "training": peak_trend_training,
            "note": "在峰值加权基础上加入 horizon 内趋势差分约束。",
        },
        {
            "name": "history_l16_compact",
            "label": "历史特征 L16 compact",
            "features": history_features,
            "lookback": 16,
            "model": compact_model,
            "training": training,
            "note": "复测当前最强消融方向，降低容量和学习率减少小样本过拟合。",
        },
        {
            "name": "history_l24_compact",
            "label": "历史特征 L24 compact",
            "features": history_features,
            "lookback": 24,
            "model": compact_model,
            "training": training,
            "note": "扩大上下文到约半年，捕捉更完整的上升/回落段。",
        },
        {
            "name": "history_l52_compact",
            "label": "历史特征 L52 compact",
            "features": history_features,
            "lookback": 52,
            "model": compact_model,
            "training": training,
            "note": "显式给模型一整年历史，让季节性不完全依赖派生特征。",
        },
        {
            "name": "all_l52_regularized",
            "label": "三源融合 L52 regularized",
            "features": all_features,
            "lookback": 52,
            "model": regularized_model,
            "training": training,
            "note": "保留三源特征，但用更强正则和一年窗口抑制噪声干扰。",
        },
    ]


def apply_trial_config(base_config: dict, trial: dict[str, Any], args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(base_config)
    cfg.setdefault("features", {})["include_feature_cols"] = trial["features"]
    cfg.setdefault("data", {})["lookback_window"] = int(trial["lookback"])
    cfg.setdefault("data", {})["splits_dir"] = os.path.join("data", "splits", "optimization", trial["name"])

    cfg.setdefault("model", {}).setdefault("itransformer", {}).update(trial["model"])

    train_cfg = cfg.setdefault("training", {})
    train_cfg.update(trial.get("training", {}))
    train_cfg["checkpoint_dir"] = os.path.join("checkpoints", "optimization")
    train_cfg["log_dir"] = os.path.join("results", "logs", "optimization")

    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.patience is not None:
        train_cfg["patience"] = args.patience
    if args.debug:
        train_cfg["epochs"] = 5
        train_cfg["patience"] = 3

    return cfg


def run_itransformer_trial(
    base_config: dict,
    df_featured: pd.DataFrame,
    trial: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    cfg = apply_trial_config(base_config, trial, args)
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

    target_col = cfg.get("features", {}).get("target_col", "ili_rate")
    target_scaler = preprocessor.scalers.get(target_col)
    model = build_itransformer(cfg, X_train.shape[1])
    trainer = Trainer(model, cfg, model_name=f"iTransformerOpt_{trial['name']}")
    history = trainer.train(train_loader, val_loader)
    metrics, _, _, horizon_metrics = trainer.evaluate(test_loader, target_scaler)

    val_pred = predict_loader(trainer.model, val_loader, trainer.device)
    if target_scaler is not None:
        val_pred = inverse_scaled(val_pred, target_scaler)
        val_actual = inverse_scaled(y_val, target_scaler)
    else:
        val_actual = y_val
    validation_metrics = compute_all_metrics(
        val_actual.flatten(),
        val_pred.flatten(),
        include_peak_time_offset=False,
    )

    best_epoch = int(np.argmin(history["val_loss"]) + 1)
    return {
        "type": "itransformer_trial",
        "name": trial["name"],
        "label": trial["label"],
        "note": trial["note"],
        "lookback": int(trial["lookback"]),
        "feature_count": int(len(preprocessor.feature_cols)),
        "feature_cols": preprocessor.feature_cols,
        "model_config": trial["model"],
        "training_config": {
            key: cfg.get("training", {}).get(key)
            for key in [
                "epochs",
                "patience",
                "optimizer",
                "learning_rate",
                "weight_decay",
                "scheduler",
                "batch_size",
                "loss",
                "peak_threshold",
                "peak_weight",
                "trend_weight",
            ]
        },
        "best_epoch": best_epoch,
        "best_val_loss": float(np.min(history["val_loss"])),
        "validation_metrics": {key: float(value) for key, value in validation_metrics.items()},
        "metrics": {key: float(value) for key, value in metrics.items()},
        "horizon_metrics": {key: float(value) for key, value in horizon_metrics.items()},
    }


def fit_sarimax(series: np.ndarray):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        series,
        order=(2, 1, 2),
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=100)


def fallback_forecast(context: np.ndarray, horizon: int) -> np.ndarray:
    if len(context) >= 52:
        values = []
        for step in range(horizon):
            idx = len(context) - 52 + step
            values.append(context[idx] if idx < len(context) else context[-1])
        return np.array(values, dtype=float)
    return np.repeat(float(context[-1]), horizon)


def arima_window_forecasts(
    history_series: np.ndarray,
    segment_series: np.ndarray,
    lookback: int,
    horizon: int,
    initial_fit_size: int = 104,
) -> np.ndarray:
    """Generate ARIMA forecasts aligned to preprocessor windows for one split."""
    n_windows = len(segment_series) - lookback - horizon + 1
    if n_windows <= 0:
        return np.empty((0, horizon), dtype=float)

    preds: list[np.ndarray] = []
    fitted = None
    observed_upto = 0

    if len(history_series) >= max(24, lookback):
        try:
            fitted = fit_sarimax(history_series)
        except Exception as exc:
            print(f"[Residual ARIMA] 历史段拟合失败，使用季节/持久性兜底: {exc}")
    elif len(segment_series) >= max(initial_fit_size, lookback + horizon):
        fit_end = min(initial_fit_size, len(segment_series) - horizon)
        try:
            fitted = fit_sarimax(segment_series[:fit_end])
            observed_upto = fit_end
        except Exception as exc:
            print(f"[Residual ARIMA] 训练段初始拟合失败，使用季节/持久性兜底: {exc}")

    for i in range(n_windows):
        context_end = i + lookback
        context = segment_series[i:context_end]

        if fitted is None or context_end < observed_upto:
            preds.append(fallback_forecast(context, horizon))
            continue

        if context_end > observed_upto:
            try:
                fitted = fitted.append(segment_series[observed_upto:context_end], refit=False)
                observed_upto = context_end
            except Exception as exc:
                print(f"[Residual ARIMA] 状态更新失败，窗口 {i} 使用兜底: {exc}")
                preds.append(fallback_forecast(context, horizon))
                continue

        try:
            preds.append(np.array(fitted.forecast(steps=horizon), dtype=float))
        except Exception as exc:
            print(f"[Residual ARIMA] 预测失败，窗口 {i} 使用兜底: {exc}")
            preds.append(fallback_forecast(context, horizon))

    return np.vstack(preds)


def to_scaled(values: np.ndarray, scaler) -> np.ndarray:
    return scaler.transform(values.reshape(-1, 1)).reshape(values.shape)


def inverse_scaled(values: np.ndarray, scaler) -> np.ndarray:
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(values.shape)


def predict_loader(model, loader, device) -> np.ndarray:
    import torch

    model.eval()
    preds = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            preds.append(model(batch_x).cpu().numpy())
    return np.concatenate(preds, axis=0)


def split_target_series(df_featured: pd.DataFrame, split_metadata: dict, target_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_rows = int(split_metadata["train_rows"])
    val_rows = int(split_metadata["val_rows"])
    target = df_featured[target_col].to_numpy(dtype=float)
    train = target[:train_rows]
    val = target[train_rows:train_rows + val_rows]
    test = target[train_rows + val_rows:]
    return train, val, test


def run_residual_hybrid(
    base_config: dict,
    df_featured: pd.DataFrame,
    source_trial: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    residual_trial = copy.deepcopy(source_trial)
    residual_trial["name"] = f"arima_residual_{source_trial['name']}"
    residual_trial["label"] = f"ARIMA + 残差 {source_trial['label']}"

    cfg = apply_trial_config(base_config, residual_trial, args)
    cfg["data"]["splits_dir"] = os.path.join("data", "splits", "optimization", residual_trial["name"])
    cfg["training"]["checkpoint_dir"] = os.path.join("checkpoints", "optimization")
    cfg["training"]["log_dir"] = os.path.join("results", "logs", "optimization")
    cfg["training"]["loss"] = "mse"

    set_random_seed(cfg)
    target_col = cfg.get("features", {}).get("target_col", "ili_rate")
    lookback = int(cfg.get("data", {}).get("lookback_window", 16))
    horizon = int(cfg.get("data", {}).get("forecast_horizon", 4))

    preprocessor = DataPreprocessor(cfg)
    splits = preprocessor.process(df_featured)
    X_train, y_train, X_val, y_val, X_test, y_test = splits
    scaler = preprocessor.scalers[target_col]
    train_series, val_series, test_series = split_target_series(
        df_featured,
        preprocessor.split_metadata,
        target_col,
    )

    base_train = arima_window_forecasts(np.array([]), train_series, lookback, horizon)
    base_val = arima_window_forecasts(train_series, val_series, lookback, horizon)
    base_test = arima_window_forecasts(np.concatenate([train_series, val_series]), test_series, lookback, horizon)

    min_train = min(len(X_train), len(base_train), len(y_train))
    min_val = min(len(X_val), len(base_val), len(y_val))
    min_test = min(len(X_test), len(base_test), len(y_test))
    if min_train == 0 or min_val == 0 or min_test == 0:
        raise RuntimeError("残差混合模型无法构造对齐窗口，请检查 lookback/horizon。")

    X_train, y_train, base_train = X_train[:min_train], y_train[:min_train], base_train[:min_train]
    X_val, y_val, base_val = X_val[:min_val], y_val[:min_val], base_val[:min_val]
    X_test, y_test, base_test = X_test[:min_test], y_test[:min_test], base_test[:min_test]

    base_train_scaled = to_scaled(base_train, scaler)
    base_val_scaled = to_scaled(base_val, scaler)
    base_test_scaled = to_scaled(base_test, scaler)
    y_train_residual = y_train - base_train_scaled
    y_val_residual = y_val - base_val_scaled
    y_test_residual = y_test - base_test_scaled

    train_loader, val_loader, test_loader = create_dataloaders(
        X_train,
        y_train_residual,
        X_val,
        y_val_residual,
        X_test,
        y_test_residual,
        batch_size=cfg.get("training", {}).get("batch_size", 32),
        num_workers=cfg.get("training", {}).get("num_workers", 0),
    )

    model = build_itransformer(cfg, X_train.shape[1])
    trainer = Trainer(model, cfg, model_name=f"iTransformerResidual_{source_trial['name']}")
    history = trainer.train(train_loader, val_loader)

    best_model_path = os.path.join(trainer.checkpoint_dir, f"{trainer.model_name}_best.pt")
    if os.path.exists(best_model_path):
        trainer.load_checkpoint(best_model_path)

    residual_pred_scaled = predict_loader(trainer.model, test_loader, trainer.device)
    final_pred_scaled = base_test_scaled + residual_pred_scaled
    final_pred = inverse_scaled(final_pred_scaled, scaler)
    actual = inverse_scaled(y_test, scaler)

    metrics = compute_all_metrics(actual.flatten(), final_pred.flatten(), include_peak_time_offset=False)
    horizon_metrics = compute_horizon_metrics(actual, final_pred)
    base_metrics = compute_all_metrics(actual.flatten(), base_test.flatten(), include_peak_time_offset=False)

    print("\n[ARIMA + iTransformer Residual] 测试集评估结果:")
    print(format_metrics(metrics))

    return {
        "type": "arima_residual_hybrid",
        "name": residual_trial["name"],
        "label": residual_trial["label"],
        "source_trial": source_trial["name"],
        "lookback": lookback,
        "feature_count": int(len(preprocessor.feature_cols)),
        "feature_cols": preprocessor.feature_cols,
        "best_epoch": int(np.argmin(history["val_loss"]) + 1),
        "best_val_loss": float(np.min(history["val_loss"])),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "horizon_metrics": {key: float(value) for key, value in horizon_metrics.items()},
        "arima_component_metrics": {key: float(value) for key, value in base_metrics.items()},
    }


def run_trainval_refit(
    base_config: dict,
    df_featured: pd.DataFrame,
    source_trial: dict[str, Any],
    source_result: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    refit_trial = copy.deepcopy(source_trial)
    refit_trial["name"] = f"trainval_refit_{source_trial['name']}"
    refit_trial["label"] = f"Train+Val 重训 {source_trial['label']}"

    cfg = apply_trial_config(base_config, refit_trial, args)
    cfg["data"]["splits_dir"] = os.path.join("data", "splits", "optimization", refit_trial["name"])
    cfg["training"]["checkpoint_dir"] = os.path.join("checkpoints", "optimization")
    cfg["training"]["log_dir"] = os.path.join("results", "logs", "optimization")

    set_random_seed(cfg)
    target_col = cfg.get("features", {}).get("target_col", "ili_rate")
    preprocessor = DataPreprocessor(cfg)
    splits = preprocessor.process(df_featured)
    X_train, y_train, X_val, y_val, X_test, y_test = splits

    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)
    batch_size = cfg.get("training", {}).get("batch_size", 32)
    trainval_loader = DataLoader(
        FluDataset(X_trainval, y_trainval),
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.get("training", {}).get("num_workers", 0),
        pin_memory=True,
    )
    test_loader = DataLoader(
        FluDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.get("training", {}).get("num_workers", 0),
        pin_memory=True,
    )

    selected_epoch = max(1, int(source_result.get("best_epoch", 1)))
    model = build_itransformer(cfg, X_train.shape[1])
    trainer = Trainer(model, cfg, model_name=f"iTransformerTrainVal_{source_trial['name']}")
    trainer.train_fixed_epochs(trainval_loader, epochs=selected_epoch)
    metrics, _, _, horizon_metrics = trainer.evaluate(test_loader, preprocessor.scalers.get(target_col))

    return {
        "type": "itransformer_trainval_refit",
        "name": refit_trial["name"],
        "label": refit_trial["label"],
        "source_trial": source_trial["name"],
        "lookback": int(refit_trial["lookback"]),
        "feature_count": int(len(preprocessor.feature_cols)),
        "feature_cols": preprocessor.feature_cols,
        "selected_epoch": selected_epoch,
        "metrics": {key: float(value) for key, value in metrics.items()},
        "horizon_metrics": {key: float(value) for key, value in horizon_metrics.items()},
    }


def run_validation_weighted_ensemble(
    base_config: dict,
    df_featured: pd.DataFrame,
    source_trial: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    cfg = apply_trial_config(base_config, source_trial, args)
    target_col = cfg.get("features", {}).get("target_col", "ili_rate")
    lookback = int(cfg.get("data", {}).get("lookback_window", 16))
    horizon = int(cfg.get("data", {}).get("forecast_horizon", 4))

    preprocessor = DataPreprocessor(cfg)
    splits = preprocessor.process(df_featured)
    X_train, y_train, X_val, y_val, X_test, y_test = splits
    scaler = preprocessor.scalers[target_col]

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

    model = build_itransformer(cfg, X_train.shape[1])
    trainer = Trainer(model, cfg, model_name=f"iTransformerOpt_{source_trial['name']}")
    checkpoint_path = os.path.join(trainer.checkpoint_dir, f"{trainer.model_name}_best.pt")
    if os.path.exists(checkpoint_path):
        trainer.load_checkpoint(checkpoint_path)
    else:
        print("[Ensemble] 未找到源模型检查点，重新训练该候选。")
        trainer.train(train_loader, val_loader)
        trainer.load_checkpoint(checkpoint_path)

    it_val = inverse_scaled(predict_loader(trainer.model, val_loader, trainer.device), scaler)
    it_test = inverse_scaled(predict_loader(trainer.model, test_loader, trainer.device), scaler)
    actual_val = inverse_scaled(y_val, scaler)
    actual_test = inverse_scaled(y_test, scaler)

    train_series, val_series, test_series = split_target_series(
        df_featured,
        preprocessor.split_metadata,
        target_col,
    )
    arima_val = arima_window_forecasts(train_series, val_series, lookback, horizon)
    arima_test = arima_window_forecasts(np.concatenate([train_series, val_series]), test_series, lookback, horizon)

    min_val = min(len(actual_val), len(it_val), len(arima_val))
    min_test = min(len(actual_test), len(it_test), len(arima_test))
    actual_val, it_val, arima_val = actual_val[:min_val], it_val[:min_val], arima_val[:min_val]
    actual_test, it_test, arima_test = actual_test[:min_test], it_test[:min_test], arima_test[:min_test]

    best_alpha = 0.0
    best_val_rmse = float("inf")
    for alpha in np.linspace(0.0, 1.0, 101):
        val_pred = alpha * arima_val + (1.0 - alpha) * it_val
        rmse_value = compute_all_metrics(
            actual_val.flatten(),
            val_pred.flatten(),
            include_peak_time_offset=False,
        )["RMSE"]
        if rmse_value < best_val_rmse:
            best_alpha = float(alpha)
            best_val_rmse = float(rmse_value)

    test_pred = best_alpha * arima_test + (1.0 - best_alpha) * it_test
    metrics = compute_all_metrics(actual_test.flatten(), test_pred.flatten(), include_peak_time_offset=False)
    horizon_metrics = compute_horizon_metrics(actual_test, test_pred)
    prediction_path = upsert_prediction_column(
        base_config.get("reporting", {}).get("reports_dir", "results/reports"),
        df_featured,
        preprocessor.split_metadata,
        lookback,
        horizon,
        actual_test,
        test_pred,
    )

    print("\n[ARIMA + iTransformer Validation Ensemble] 测试集评估结果:")
    print(f"  alpha_arima: {best_alpha:.2f} (val_RMSE={best_val_rmse:.4f})")
    print(format_metrics(metrics))

    return {
        "type": "validation_weighted_ensemble",
        "name": f"ensemble_{source_trial['name']}",
        "label": f"验证集加权融合 {source_trial['label']}",
        "source_trial": source_trial["name"],
        "lookback": lookback,
        "feature_count": int(len(preprocessor.feature_cols)),
        "alpha_arima": best_alpha,
        "alpha_itransformer": float(1.0 - best_alpha),
        "best_val_rmse": best_val_rmse,
        "metrics": {key: float(value) for key, value in metrics.items()},
        "horizon_metrics": {key: float(value) for key, value in horizon_metrics.items()},
        "prediction_path": prediction_path,
    }


def calibrate_per_horizon(
    val_pred: np.ndarray,
    val_actual: np.ndarray,
    test_pred: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Fit y=a*x+b on validation predictions, separately for each horizon."""
    calibrated = np.zeros_like(test_pred)
    params = []
    for step in range(test_pred.shape[1]):
        design = np.column_stack([val_pred[:, step], np.ones(len(val_pred))])
        coef, *_ = np.linalg.lstsq(design, val_actual[:, step], rcond=None)
        slope, intercept = float(coef[0]), float(coef[1])
        calibrated[:, step] = slope * test_pred[:, step] + intercept
        params.append({"horizon": step + 1, "slope": slope, "intercept": intercept})
    return calibrated, params


def run_validation_calibration(
    base_config: dict,
    df_featured: pd.DataFrame,
    source_trial: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    cfg = apply_trial_config(base_config, source_trial, args)
    target_col = cfg.get("features", {}).get("target_col", "ili_rate")

    preprocessor = DataPreprocessor(cfg)
    splits = preprocessor.process(df_featured)
    X_train, y_train, X_val, y_val, X_test, y_test = splits
    scaler = preprocessor.scalers[target_col]

    _, val_loader, test_loader = create_dataloaders(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        batch_size=cfg.get("training", {}).get("batch_size", 32),
        num_workers=cfg.get("training", {}).get("num_workers", 0),
    )

    model = build_itransformer(cfg, X_train.shape[1])
    trainer = Trainer(model, cfg, model_name=f"iTransformerOpt_{source_trial['name']}")
    checkpoint_path = os.path.join(trainer.checkpoint_dir, f"{trainer.model_name}_best.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"校准需要先训练源模型，未找到: {checkpoint_path}")
    trainer.load_checkpoint(checkpoint_path)

    val_pred = inverse_scaled(predict_loader(trainer.model, val_loader, trainer.device), scaler)
    test_pred = inverse_scaled(predict_loader(trainer.model, test_loader, trainer.device), scaler)
    val_actual = inverse_scaled(y_val, scaler)
    test_actual = inverse_scaled(y_test, scaler)

    calibrated_pred, params = calibrate_per_horizon(val_pred, val_actual, test_pred)
    metrics = compute_all_metrics(
        test_actual.flatten(),
        calibrated_pred.flatten(),
        include_peak_time_offset=False,
    )
    horizon_metrics = compute_horizon_metrics(test_actual, calibrated_pred)
    raw_metrics = compute_all_metrics(
        test_actual.flatten(),
        test_pred.flatten(),
        include_peak_time_offset=False,
    )

    print("\n[iTransformer Validation Calibration] 测试集评估结果:")
    print(format_metrics(metrics))

    return {
        "type": "validation_calibrated_itransformer",
        "name": f"calibrated_{source_trial['name']}",
        "label": f"验证集校准 {source_trial['label']}",
        "source_trial": source_trial["name"],
        "lookback": int(source_trial["lookback"]),
        "feature_count": int(len(preprocessor.feature_cols)),
        "calibration_params": params,
        "raw_source_metrics": {key: float(value) for key, value in raw_metrics.items()},
        "metrics": {key: float(value) for key, value in metrics.items()},
        "horizon_metrics": {key: float(value) for key, value in horizon_metrics.items()},
    }


def load_existing_optimization_results(reports_dir: str) -> list[dict[str, Any]]:
    summary_path = os.path.join(reports_dir, "itransformer_optimization_summary.json")
    if not os.path.exists(summary_path):
        return []

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f).get("results", [])
    except Exception as exc:
        print(f"[优化报告] 既有优化结果读取失败: {exc}")
        return []


def merge_results(
    existing_results: list[dict[str, Any]],
    new_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []

    for item in [*existing_results, *new_results]:
        key = (str(item.get("type", "")), str(item.get("name", "")))
        if key not in merged:
            order.append(key)
        merged[key] = item

    return [merged[key] for key in order]


def validation_score_text(item: dict[str, Any]) -> str:
    val_rmse = item.get("validation_metrics", {}).get("RMSE")
    if val_rmse is None:
        val_rmse = item.get("best_val_rmse")
    return f"{val_rmse:.3f}" if val_rmse is not None else ""


def find_existing_trial_result(
    existing_results: list[dict[str, Any]],
    trial_name: str,
) -> dict[str, Any] | None:
    candidates = [
        item
        for item in existing_results
        if item.get("type") == "itransformer_trial" and item.get("name") == trial_name
    ]
    return candidates[-1] if candidates else None


def upsert_prediction_column(
    reports_dir: str,
    df_featured: pd.DataFrame,
    split_metadata: dict[str, Any],
    lookback: int,
    horizon: int,
    actual: np.ndarray,
    pred: np.ndarray,
    column_name: str = OPTIMIZED_MODEL_NAME,
) -> str:
    """Write optimized multi-step predictions so the Streamlit app can plot them."""
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, "test_predictions.csv")
    prediction_index = build_prediction_index(df_featured, split_metadata, lookback, horizon)
    export_len = min(len(prediction_index), actual.size, pred.size)
    new_df = prediction_index.iloc[:export_len].copy()
    new_df["actual"] = actual.flatten()[:export_len]
    new_df[column_name] = pred.flatten()[:export_len]

    key_cols = ["sample_index", "horizon", "anchor_date", "target_date"]
    if os.path.exists(path):
        existing = pd.read_csv(path)
        merged = existing.drop(columns=[column_name], errors="ignore").merge(
            new_df[[*key_cols, column_name]],
            on=key_cols,
            how="left",
        )
        if column_name not in merged.columns or merged[column_name].isna().all():
            aligned_len = min(len(existing), export_len)
            merged = existing.drop(columns=[column_name], errors="ignore").copy()
            merged.loc[: aligned_len - 1, column_name] = new_df[column_name].iloc[:aligned_len].to_numpy()
        output = merged
    else:
        output = new_df

    output.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[优化预测] 已同步 {column_name} 预测明细: {path}")
    return path


def save_outputs(
    results: list[dict[str, Any]],
    reports_dir: str,
    merge_existing: bool = False,
) -> None:
    os.makedirs(reports_dir, exist_ok=True)
    main_baselines = {}
    main_summary_path = os.path.join(reports_dir, "experiment_summary.json")
    if os.path.exists(main_summary_path):
        try:
            with open(main_summary_path, "r", encoding="utf-8") as f:
                main_baselines = json.load(f).get("metrics", {})
        except Exception as exc:
            print(f"[优化报告] 主实验摘要读取失败: {exc}")

    if merge_existing:
        results = merge_results(load_existing_optimization_results(reports_dir), results)

    summary = {
        "purpose": "iTransformer 调参和 ARIMA 残差混合优化实验。",
        "selection_note": "候选配置保留验证集最佳权重；测试集指标用于最终横向比较，不应用于继续手工挑参。",
        "main_experiment_baselines": main_baselines,
        "results": results,
    }
    json_path = os.path.join(reports_dir, "itransformer_optimization_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    sorted_results = sorted(results, key=lambda item: item["metrics"].get("RMSE", float("inf")))
    lines = [
        "# iTransformer 优化实验报告",
        "",
        "## 实验目的",
        "",
        "围绕当前 ARIMA 强、iTransformer 接近但略弱的结果，验证历史窗口/正则化调参、目标动态特征、峰值加权、验证集校准，以及 ARIMA 基础趋势 + iTransformer 残差修正。",
        "",
        "## 结果汇总",
        "",
        "| 排名 | 实验 | 类型 | Lookback | 特征数 | Val RMSE | RMSE | MAE | MAPE | R2 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, item in enumerate(sorted_results, start=1):
        m = item["metrics"]
        val_text = validation_score_text(item)
        lines.append(
            f"| {idx} | {item['label']} | {item['type']} | {item.get('lookback', '')} | "
            f"{item.get('feature_count', '')} | {val_text} | {m.get('RMSE', 0):.3f} | "
            f"{m.get('MAE', 0):.3f} | {m.get('MAPE', 0):.2f}% | {m.get('R2', 0):.3f} |"
        )

    best = sorted_results[0] if sorted_results else None
    arima_baseline = main_baselines.get("ARIMA", {})
    if best is not None:
        best_rmse = best["metrics"].get("RMSE", float("inf"))
        arima_rmse = arima_baseline.get("RMSE")
        if arima_rmse is None:
            comparison = "当前未读取到主实验 ARIMA 指标，因此只给出本轮优化内部排序。"
        else:
            delta = best_rmse - float(arima_rmse)
            if delta < 0:
                comparison = f"该结果较主实验 ARIMA 的 RMSE 低 {abs(delta):.3f}。"
            else:
                comparison = f"该结果仍比主实验 ARIMA 的 RMSE 高 {delta:.3f}，尚未形成稳定领先。"
        lines.extend(
            [
                "",
                "## 当前结论",
                "",
                f"本轮优化中 `{best['label']}` 的 RMSE 最低。{comparison}",
            ]
        )

    lines.extend(
        [
            "",
            "## 选择说明",
            "",
            "调参阶段按验证集选择训练权重和后处理参数；上表按测试集指标排序，仅用于最终横向报告，不再据此继续手工调参。",
        ]
    )

    if main_baselines:
        lines.extend(["", "## 主实验参照", ""])
        lines.append("| 模型 | RMSE | MAE | MAPE | R2 |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for model_name, metrics in sorted(main_baselines.items(), key=lambda item: item[1].get("RMSE", float("inf"))):
            lines.append(
                f"| {model_name} | {metrics.get('RMSE', 0):.3f} | "
                f"{metrics.get('MAE', 0):.3f} | {metrics.get('MAPE', 0):.2f}% | "
                f"{metrics.get('R2', 0):.3f} |"
            )

    lines.extend(
        [
            "",
            "## 后续动作",
            "",
            "1. 将 `验证集加权融合 历史动态 L16 baseline` 作为当前优化模型口径；单模型主线仍保留 `历史动态 L16 baseline`，便于解释 iTransformer 自身贡献。",
            "2. `ARIMA + 残差` 可作为混合建模补充实验：整体 RMSE 略弱于加权融合，但峰值强度误差更接近 ARIMA。",
            "3. 峰值加权能改善部分峰值指标，但会伤害整体 RMSE；论文中可作为预警目标的权衡实验，而不是主模型。",
            "4. 若要继续拉开与 ARIMA 的差距，应优先补充省份级面板数据或更可靠的搜索/气象领先信号。",
        ]
    )

    md_path = os.path.join(reports_dir, "itransformer_optimization_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n优化实验报告已保存:")
    print(f"  - {json_path}")
    print(f"  - {md_path}")


def main():
    parser = argparse.ArgumentParser(description="运行 iTransformer 优化实验")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--skip-collect", action="store_true", help="使用已有 merged_dataset.csv")
    parser.add_argument("--max-trials", type=int, default=None, help="限制候选 iTransformer 调参实验数量")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖训练 epoch")
    parser.add_argument("--patience", type=int, default=None, help="覆盖早停 patience")
    parser.add_argument("--debug", action="store_true", help="快速调试模式")
    parser.add_argument("--skip-residual", action="store_true", help="跳过 ARIMA 残差混合模型")
    parser.add_argument("--skip-refit", action="store_true", help="跳过 train+val 固定轮数重训")
    parser.add_argument("--skip-ensemble", action="store_true", help="跳过验证集加权融合")
    parser.add_argument("--skip-calibration", action="store_true", help="跳过验证集线性校准")
    parser.add_argument("--only-trial", default=None, help="只运行指定候选实验 name")
    parser.add_argument(
        "--postprocess-trial",
        default=None,
        help="只对指定候选运行残差/重训/融合/校准，不重新训练基础候选",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="保存报告时合并既有优化结果，适合增量补跑单个候选",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    reports_dir = config.get("reporting", {}).get("reports_dir", "results/reports")
    existing_results = load_existing_optimization_results(reports_dir)
    processed_dir = config.get("data", {}).get("processed_dir", "data/processed")
    merged_path = os.path.join(processed_dir, "merged_dataset.csv")
    if args.skip_collect and os.path.exists(merged_path):
        print("[跳过] 使用已有数据")
        df = pd.read_csv(merged_path)
    else:
        df = collect_data(config)

    df = enrich_with_flu_context(df, config)
    df_featured = feature_engineering(config, df)
    trials = build_trials(df_featured, config)
    selected_trial_name = args.postprocess_trial or args.only_trial
    if selected_trial_name:
        trials = [trial for trial in trials if trial["name"] == selected_trial_name]
        if not trials:
            raise ValueError(f"未找到候选实验: {selected_trial_name}")
    if args.max_trials is not None:
        trials = trials[: max(args.max_trials, 0)]
    if args.debug:
        trials = trials[:1]

    results = []
    if args.postprocess_trial:
        print(f"[优化实验] 跳过基础候选训练，仅补跑后处理: {args.postprocess_trial}")
    else:
        for trial in trials:
            print("\n" + "=" * 60)
            print(f"iTransformer 优化实验: {trial['label']} ({trial['name']})")
            print("=" * 60)
            results.append(run_itransformer_trial(config, df_featured, trial, args))

    best_source_trial = None
    best_source_result = None
    if args.postprocess_trial and trials:
        best_source_trial = trials[0]
        best_source_result = find_existing_trial_result(existing_results, best_source_trial["name"])
        if best_source_result is None and not args.skip_refit:
            raise FileNotFoundError(
                "postprocess-trial 需要既有基础候选结果来确定 train+val 重训轮数；"
                "请先运行基础实验，或添加 --skip-refit。"
            )
    elif trials and results:
        by_val = sorted(
            [item for item in results if item["type"] == "itransformer_trial"],
            key=lambda item: item.get("validation_metrics", {}).get("RMSE", float("inf")),
        )
        if by_val:
            best_source_result = by_val[0]
            best_source_trial = next(
                (trial for trial in trials if trial["name"] == best_source_result["name"]),
                trials[0],
            )

    if best_source_trial and not args.skip_residual:
        print("\n" + "=" * 60)
        print(f"ARIMA 残差混合实验: {best_source_trial['label']} ({best_source_trial['name']})")
        print("=" * 60)
        results.append(run_residual_hybrid(config, df_featured, best_source_trial, args))

    if best_source_trial and best_source_result and not args.skip_refit:
        print("\n" + "=" * 60)
        print(f"Train+Val 固定轮数重训: {best_source_trial['label']} ({best_source_trial['name']})")
        print("=" * 60)
        results.append(run_trainval_refit(config, df_featured, best_source_trial, best_source_result, args))

    if best_source_trial and not args.skip_ensemble:
        print("\n" + "=" * 60)
        print(f"验证集加权融合: {best_source_trial['label']} ({best_source_trial['name']})")
        print("=" * 60)
        results.append(run_validation_weighted_ensemble(config, df_featured, best_source_trial, args))

    if best_source_trial and not args.skip_calibration:
        print("\n" + "=" * 60)
        print(f"验证集线性校准: {best_source_trial['label']} ({best_source_trial['name']})")
        print("=" * 60)
        results.append(run_validation_calibration(config, df_featured, best_source_trial, args))

    save_outputs(
        results,
        reports_dir,
        merge_existing=args.merge_existing or args.postprocess_trial is not None,
    )


if __name__ == "__main__":
    main()
