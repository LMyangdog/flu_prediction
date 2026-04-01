"""
主训练脚本 — 完整流水线

Usage:
    python scripts/train.py                              # 全部模型
    python scripts/train.py --model iTransformer          # 仅 iTransformer
    python scripts/train.py --model LSTM                  # 仅 LSTM
    python scripts/train.py --skip-collect                # 跳过数据采集
    python scripts/train.py --debug                       # Debug 模式（5 epochs）

Author: flu_prediction project
"""

import os
import sys
import json
import argparse
import pickle
from pathlib import Path

import numpy as np
import yaml

# 将项目根目录加入 Python 路径
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

from src.data.collector import MultiSourceDataCollector
from src.data.feature_engineer import FeatureEngineer
from src.data.preprocessor import DataPreprocessor
from src.data.dataset import create_dataloaders
from src.models.itransformer import build_itransformer
from src.models.lstm_baseline import build_lstm
from src.models.dlinear_baseline import build_dlinear
from src.models.arima_baseline import ARIMABaseline
from src.training.trainer import Trainer
from src.utils.visualization import Visualizer
from src.utils.metrics import compute_all_metrics, format_metrics


def load_config(config_path: str = "config/config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def collect_data(config: dict):
    """第1步：数据采集"""
    print("\n" + "=" * 60)
    print("Step 1: 多源数据采集")
    print("=" * 60)
    
    collector = MultiSourceDataCollector(config)
    merged_df = collector.collect_all()
    return merged_df


def feature_engineering(config: dict, df):
    """第2步：特征工程"""
    print("\n" + "=" * 60)
    print("Step 2: 特征工程")
    print("=" * 60)
    
    import pandas as pd
    engineer = FeatureEngineer(config)
    df_featured = engineer.transform(df)
    
    # 保存特征工程后的数据
    processed_dir = config.get('data', {}).get('processed_dir', 'data/processed')
    os.makedirs(processed_dir, exist_ok=True)
    df_featured.to_csv(os.path.join(processed_dir, 'featured_dataset.csv'),
                       index=False, encoding='utf-8-sig')
    
    return df_featured


def preprocess_data(config: dict, df):
    """第3步：数据预处理"""
    preprocessor = DataPreprocessor(config)
    splits = preprocessor.process(df)
    return splits, preprocessor


def train_itransformer(config: dict, train_loader, val_loader, test_loader, 
                       num_vars: int, scaler=None):
    """训练 iTransformer"""
    model = build_itransformer(config, num_vars)
    trainer = Trainer(model, config, model_name="iTransformer")
    history = trainer.train(train_loader, val_loader)
    metrics, preds, actuals = trainer.evaluate(test_loader, scaler)
    return model, trainer, history, metrics, preds, actuals


def train_lstm(config: dict, train_loader, val_loader, test_loader,
               num_vars: int, scaler=None):
    """训练 LSTM 基准模型"""
    model = build_lstm(config, num_vars)
    trainer = Trainer(model, config, model_name="LSTM")
    history = trainer.train(train_loader, val_loader)
    metrics, preds, actuals = trainer.evaluate(test_loader, scaler)
    return model, trainer, history, metrics, preds, actuals


def train_dlinear(config: dict, train_loader, val_loader, test_loader,
                  num_vars: int, scaler=None):
    """训练 DLinear 基准模型"""
    model = build_dlinear(config, num_vars)
    trainer = Trainer(model, config, model_name="DLinear")
    history = trainer.train(train_loader, val_loader)
    metrics, preds, actuals = trainer.evaluate(test_loader, scaler)
    return model, trainer, history, metrics, preds, actuals


def run_arima(config: dict, preprocessor: DataPreprocessor):
    """运行 ARIMA 基准模型"""
    print(f"\n{'=' * 60}")
    print("ARIMA 基准模型")
    print("=" * 60)
    
    import pandas as pd
    
    # 加载原始 ILI 数据
    processed_dir = config.get('data', {}).get('processed_dir', 'data/processed')
    df = pd.read_csv(os.path.join(processed_dir, 'merged_dataset.csv'))
    
    target_col = config.get('features', {}).get('target_col', 'ili_rate')
    series = df[target_col].values
    
    # 划分训练/测试
    train_ratio = config.get('data', {}).get('train_ratio', 0.7)
    val_ratio = config.get('data', {}).get('val_ratio', 0.15)
    train_size = int(len(series) * (train_ratio + val_ratio))
    horizon = config.get('data', {}).get('forecast_horizon', 4)
    
    arima = ARIMABaseline(order=(2, 1, 2), seasonal_order=None)
    preds, actuals = arima.fit_predict_rolling(series, train_size, horizon)
    
    if len(preds) > 0:
        preds_flat = preds.flatten()
        actuals_flat = actuals.flatten()
        metrics = compute_all_metrics(actuals_flat, preds_flat)
        print(f"\n[ARIMA] 测试集评估结果:")
        print(format_metrics(metrics))
        return metrics, preds_flat, actuals_flat
    else:
        print("[ARIMA] 预测失败")
        return {}, np.array([]), np.array([])


def main():
    parser = argparse.ArgumentParser(description='流感预测模型训练')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['all', 'iTransformer', 'LSTM', 'DLinear', 'ARIMA'],
                       help='训练的模型')
    parser.add_argument('--skip-collect', action='store_true', help='跳过数据采集')
    parser.add_argument('--debug', action='store_true', help='Debug 模式')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    if args.debug:
        config['training']['epochs'] = 5
        config['training']['patience'] = 3
        print("[Debug 模式] epochs=5, patience=3")
    
    # ============================================================
    # Step 1: 数据采集
    # ============================================================
    import pandas as pd
    
    processed_dir = config.get('data', {}).get('processed_dir', 'data/processed')
    merged_path = os.path.join(processed_dir, 'merged_dataset.csv')
    
    if args.skip_collect and os.path.exists(merged_path):
        print("[跳过] 使用已有数据")
        df = pd.read_csv(merged_path)
    else:
        df = collect_data(config)
    
    # ============================================================
    # Step 2: 特征工程
    # ============================================================
    df_featured = feature_engineering(config, df)
    
    # ============================================================
    # Step 3: 数据预处理
    # ============================================================
    # 更新配置中的特征列（加入衍生特征）
    all_numeric_cols = df_featured.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['year', 'week']
    feature_cols = [c for c in all_numeric_cols if c not in exclude_cols]
    
    # 确保目标列在首位
    target_col = config.get('features', {}).get('target_col', 'ili_rate')
    if target_col in feature_cols:
        feature_cols.remove(target_col)
        feature_cols.insert(0, target_col)
    
    # 更新配置
    config['features']['flu_cols'] = [c for c in feature_cols if c in 
                                       config['features'].get('flu_cols', []) or 
                                       'ili' in c or 'positive' in c or 'lag' in c or 
                                       'roll' in c]
    
    # 用所有数值列作为特征
    config['features']['flu_cols'] = []
    config['features']['weather_cols'] = []
    config['features']['search_cols'] = []
    
    # 重新创建一个简单的config，将所有特征放在flu_cols中
    all_feature_config = config.copy()
    all_feature_config['features'] = {
        'target_col': target_col,
        'flu_cols': feature_cols,
        'weather_cols': [],
        'search_cols': [],
    }
    
    splits, preprocessor = preprocess_data(all_feature_config, df_featured)
    X_train, y_train, X_val, y_val, X_test, y_test = splits
    
    num_vars = X_train.shape[1]
    print(f"\n输入变量数: {num_vars}")
    
    # 创建 DataLoader
    batch_size = config.get('training', {}).get('batch_size', 32)
    num_workers = config.get('training', {}).get('num_workers', 0)
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=batch_size, num_workers=num_workers
    )
    
    # 获取目标变量的 scaler
    target_scaler = preprocessor.scalers.get(target_col)
    
    # ============================================================
    # Step 4: 模型训练与评估
    # ============================================================
    all_metrics = {}
    all_preds = {}
    all_actuals = None
    
    visualizer = Visualizer(config.get('evaluation', {}).get('figures_dir', 'results/figures'))
    
    # 数据概览可视化
    visualizer.plot_data_overview(df)
    
    # 特征相关性
    visualizer.plot_correlation_matrix(df_featured, feature_cols[:10])
    
    # --- iTransformer ---
    if args.model in ['all', 'iTransformer']:
        model, trainer, history, metrics, preds, actuals = train_itransformer(
            config, train_loader, val_loader, test_loader, num_vars, target_scaler
        )
        all_metrics['iTransformer'] = metrics
        all_preds['iTransformer'] = preds
        all_actuals = actuals
        
        visualizer.plot_training_history(history, 'iTransformer')
        visualizer.plot_predictions(actuals, preds, 'iTransformer')
        
        # 注意力热力图
        try:
            import torch
            model.eval()
            with torch.no_grad():
                sample_x = next(iter(test_loader))[0].to(next(model.parameters()).device)
                _ = model(sample_x)
                attn_weights = model.get_attention_weights()
                if attn_weights:
                    avg_attn = attn_weights[-1][0].numpy()  # 取最后一层第一个样本
                    var_names = preprocessor.feature_cols[:num_vars]
                    # 截短变量名
                    short_names = [n[:12] for n in var_names]
                    visualizer.plot_attention_heatmap(avg_attn, short_names)
        except Exception as e:
            print(f"[注意力可视化失败] {e}")
    
    # --- LSTM ---
    if args.model in ['all', 'LSTM']:
        _, _, history, metrics, preds, actuals = train_lstm(
            config, train_loader, val_loader, test_loader, num_vars, target_scaler
        )
        all_metrics['LSTM'] = metrics
        all_preds['LSTM'] = preds
        if all_actuals is None:
            all_actuals = actuals
        
        visualizer.plot_training_history(history, 'LSTM')
        visualizer.plot_predictions(actuals, preds, 'LSTM')
    
    # --- DLinear ---
    if args.model in ['all', 'DLinear']:
        _, _, history, metrics, preds, actuals = train_dlinear(
            config, train_loader, val_loader, test_loader, num_vars, target_scaler
        )
        all_metrics['DLinear'] = metrics
        all_preds['DLinear'] = preds
        if all_actuals is None:
            all_actuals = actuals
        
        visualizer.plot_training_history(history, 'DLinear')
        visualizer.plot_predictions(actuals, preds, 'DLinear')
    
    # --- ARIMA ---
    if args.model in ['all', 'ARIMA']:
        metrics, preds, actuals_arima = run_arima(config, preprocessor)
        if metrics:
            all_metrics['ARIMA'] = metrics
            all_preds['ARIMA'] = preds
    
    # ============================================================
    # Step 5: 对比分析与可视化
    # ============================================================
    if len(all_metrics) > 1:
        print(f"\n{'=' * 60}")
        print("模型对比汇总")
        print("=" * 60)
        
        for model_name, metrics in all_metrics.items():
            print(f"\n>>> {model_name}")
            print(format_metrics(metrics))
        
        # 对比柱状图
        visualizer.plot_model_comparison(all_metrics)
        
        # 多模型预测对比
        if all_actuals is not None:
            # 取最短的预测长度
            min_len = min(len(all_actuals), 
                         min(len(p) for p in all_preds.values() if len(p) > 0))
            trimmed_preds = {k: v[:min_len] for k, v in all_preds.items() if len(v) > 0}
            visualizer.plot_multi_model_predictions(all_actuals[:min_len], trimmed_preds)
    
    # 保存所有指标
    results_dir = config.get('evaluation', {}).get('figures_dir', 'results/figures')
    with open(os.path.join(results_dir, 'all_metrics.json'), 'w', encoding='utf-8') as f:
        # 将 numpy 类型转换为 Python 原生类型
        serializable_metrics = {}
        for model_name, metrics in all_metrics.items():
            serializable_metrics[model_name] = {
                k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                for k, v in metrics.items()
            }
        json.dump(serializable_metrics, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'=' * 60}")
    print("训练完成！所有结果已保存。")
    print(f"{'=' * 60}")
    print(f"  模型权重: checkpoints/")
    print(f"  可视化图表: {results_dir}/")
    print(f"  训练日志: results/logs/")


if __name__ == "__main__":
    main()
