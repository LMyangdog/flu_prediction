"""
评价指标计算模块

支持的指标：
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - MAPE (Mean Absolute Percentage Error)
    - R² (Coefficient of Determination)

Author: flu_prediction project
"""

import numpy as np
from typing import Dict


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根误差"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均绝对误差"""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """平均绝对百分比误差"""
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """决定系数 R²"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def peak_accuracy(y_true: np.ndarray, y_pred: np.ndarray, 
                  threshold_percentile: float = 90) -> Dict[str, float]:
    """
    峰值预测准确性评估
    
    评估模型对流感高峰期的捕捉能力。
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        threshold_percentile: 定义"高峰"的百分位阈值
        
    Returns:
        包含峰值相关指标的字典
    """
    threshold = np.percentile(y_true, threshold_percentile)
    
    # 真实高峰期
    true_peaks = y_true >= threshold
    pred_peaks = y_pred >= threshold
    
    # 高峰期命中率
    if true_peaks.sum() > 0:
        hit_rate = (true_peaks & pred_peaks).sum() / true_peaks.sum()
    else:
        hit_rate = 0.0
    
    # 峰值时间偏移（预测最大值与真实最大值的时间差）
    true_peak_idx = np.argmax(y_true)
    pred_peak_idx = np.argmax(y_pred)
    time_offset = abs(int(true_peak_idx) - int(pred_peak_idx))
    
    # 峰值强度误差
    peak_value_error = abs(float(y_true[true_peak_idx]) - float(y_pred[pred_peak_idx]))
    
    return {
        'peak_hit_rate': float(hit_rate),
        'peak_time_offset': time_offset,
        'peak_value_error': peak_value_error,
    }


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算所有评价指标
    
    Args:
        y_true: 真实值 (1D)
        y_pred: 预测值 (1D)
        
    Returns:
        指标字典
    """
    metrics = {
        'RMSE': rmse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
    }
    
    peak_metrics = peak_accuracy(y_true, y_pred)
    metrics.update(peak_metrics)
    
    return metrics


def compute_horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算多步预测中每个 horizon 的误差，用于论文中的分步分析。
    输入必须为二维数组: (num_samples, horizon)
    """
    if y_true.ndim != 2 or y_pred.ndim != 2:
        return {}

    horizon_metrics: Dict[str, float] = {}
    horizon = min(y_true.shape[1], y_pred.shape[1])
    for step in range(horizon):
        suffix = f"H{step + 1}"
        horizon_metrics[f"{suffix}_RMSE"] = rmse(y_true[:, step], y_pred[:, step])
        horizon_metrics[f"{suffix}_MAE"] = mae(y_true[:, step], y_pred[:, step])
        horizon_metrics[f"{suffix}_MAPE"] = mape(y_true[:, step], y_pred[:, step])

    return horizon_metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """格式化指标输出"""
    lines = []
    for name, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {name}: {value:.4f}")
        else:
            lines.append(f"  {name}: {value}")
    return "\n".join(lines)
