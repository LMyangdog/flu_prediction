"""
可视化模块 — 生成论文级别的高质量图表

图表类型：
    1. 训练损失曲线 (Train vs Val Loss)
    2. 预测对比图 (真实值 vs 预测值)
    3. 多模型对比柱状图
    4. 注意力热力图
    5. 特征相关性矩阵
    6. 数据探索性分析图

Author: flu_prediction project
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['savefig.dpi'] = 200
matplotlib.rcParams['savefig.bbox'] = 'tight'

# 配色方案
COLORS = {
    'iTransformer': '#2196F3',
    'LSTM': '#FF9800',
    'ARIMA': '#4CAF50',
    'DLinear': '#9C27B0',
    'actual': '#E53935',
    'train': '#1976D2',
    'val': '#FF7043',
}


class Visualizer:
    """高质量图表生成器"""
    
    def __init__(self, save_dir: str = "results/figures"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                              model_name: str = "Model") -> str:
        """
        绘制训练损失曲线
        
        Args:
            history: 包含 'train_loss' 和 'val_loss' 的字典
            model_name: 模型名称
            
        Returns:
            保存的文件路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # 损失曲线
        ax1.plot(epochs, history['train_loss'], 
                color=COLORS['train'], linewidth=2, label='训练损失', alpha=0.8)
        ax1.plot(epochs, history['val_loss'], 
                color=COLORS['val'], linewidth=2, label='验证损失', alpha=0.8)
        
        # 标注最低验证损失
        min_val_idx = np.argmin(history['val_loss'])
        min_val = history['val_loss'][min_val_idx]
        ax1.scatter(min_val_idx + 1, min_val, color=COLORS['val'], 
                   s=100, zorder=5, edgecolors='black', linewidth=1.5)
        ax1.annotate(f'最优: {min_val:.4f}\n(Epoch {min_val_idx + 1})',
                    xy=(min_val_idx + 1, min_val),
                    xytext=(min_val_idx + 20, min_val * 1.5),
                    arrowprops=dict(arrowstyle='->', color='gray'),
                    fontsize=10, color='gray')
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (MSE)', fontsize=12)
        ax1.set_title(f'{model_name} — 训练过程', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 学习率曲线
        if 'lr' in history:
            ax2.plot(epochs, history['lr'], color='#673AB7', linewidth=2)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Learning Rate', fontsize=12)
            ax2.set_title('学习率调度', fontsize=14, fontweight='bold')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(self.save_dir, f'{model_name}_training_history.png')
        plt.savefig(path)
        plt.close()
        print(f"  [保存] {path}")
        return path
    
    def plot_predictions(self, actuals: np.ndarray, predictions: np.ndarray,
                         model_name: str = "Model",
                         dates: Optional[np.ndarray] = None) -> str:
        """
        绘制预测对比图
        
        Args:
            actuals: 真实值序列
            predictions: 预测值序列
            model_name: 模型名称
            dates: 日期序列 (可选)
            
        Returns:
            保存的文件路径
        """
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), 
                                  gridspec_kw={'height_ratios': [3, 1]})
        
        x = range(len(actuals))
        
        # 主图：真实值 vs 预测值
        ax1 = axes[0]
        ax1.plot(x, actuals, color=COLORS['actual'], linewidth=2, 
                label='真实值 (ILI%)', alpha=0.9)
        ax1.plot(x, predictions, color=COLORS.get(model_name, '#2196F3'), 
                linewidth=2, label=f'{model_name} 预测', alpha=0.8, linestyle='--')
        
        # 填充误差区域
        ax1.fill_between(x, actuals, predictions, 
                         alpha=0.15, color=COLORS.get(model_name, '#2196F3'))
        
        ax1.set_ylabel('ILI 率 (%)', fontsize=12)
        ax1.set_title(f'{model_name} — 流感趋势预测对比', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 子图：预测误差
        ax2 = axes[1]
        errors = predictions - actuals
        ax2.bar(x, errors, color=np.where(errors >= 0, '#EF5350', '#66BB6A'), alpha=0.7)
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.set_xlabel('时间步', fontsize=12)
        ax2.set_ylabel('预测误差', fontsize=12)
        ax2.set_title('预测误差分布', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(self.save_dir, f'{model_name}_predictions.png')
        plt.savefig(path)
        plt.close()
        print(f"  [保存] {path}")
        return path
    
    def plot_model_comparison(self, metrics_dict: Dict[str, Dict[str, float]]) -> str:
        """
        绘制多模型对比柱状图
        
        Args:
            metrics_dict: {model_name: {metric_name: value}}
            
        Returns:
            保存的文件路径
        """
        models = list(metrics_dict.keys())
        base_metrics = ['RMSE', 'MAE', 'MAPE', 'R²']
        available_metrics = [m for m in base_metrics 
                            if all(m in metrics_dict[model] for model in models)]
        
        if not available_metrics:
            print("[警告] 没有可用的对比指标")
            return ""
        
        fig, axes = plt.subplots(1, len(available_metrics), 
                                  figsize=(5 * len(available_metrics), 6))
        
        if len(available_metrics) == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, available_metrics):
            values = [metrics_dict[model][metric] for model in models]
            colors = [COLORS.get(model, '#999') for model in models]
            
            bars = ax.bar(models, values, color=colors, alpha=0.85, edgecolor='white', 
                         linewidth=1.5)
            
            # 数值标注
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_title(metric, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.2, axis='y')
            
            # R² 指标越高越好，其他指标越低越好
            if metric == 'R²':
                best_idx = np.argmax(values)
            else:
                best_idx = np.argmin(values)
            bars[best_idx].set_edgecolor('#FFD700')
            bars[best_idx].set_linewidth(3)
        
        fig.suptitle('模型性能对比', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = os.path.join(self.save_dir, 'model_comparison.png')
        plt.savefig(path)
        plt.close()
        print(f"  [保存] {path}")
        return path
    
    def plot_attention_heatmap(self, attention_weights: np.ndarray,
                                variable_names: List[str],
                                layer_idx: int = -1) -> str:
        """
        绘制注意力权重热力图
        
        可视化 iTransformer 中变量间的注意力分布。
        
        Args:
            attention_weights: 注意力权重矩阵 (num_vars, num_vars)
            variable_names: 变量名列表
            layer_idx: 层索引
            
        Returns:
            保存的文件路径
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(attention_weights, 
                   annot=True, fmt='.3f',
                   xticklabels=variable_names,
                   yticklabels=variable_names,
                   cmap='YlOrRd',
                   ax=ax,
                   linewidths=0.5,
                   square=True)
        
        ax.set_title(f'iTransformer 变量间注意力权重 (Layer {layer_idx + 1})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Key 变量', fontsize=12)
        ax.set_ylabel('Query 变量', fontsize=12)
        
        plt.tight_layout()
        path = os.path.join(self.save_dir, f'attention_heatmap_layer{layer_idx + 1}.png')
        plt.savefig(path)
        plt.close()
        print(f"  [保存] {path}")
        return path
    
    def plot_data_overview(self, df: pd.DataFrame) -> str:
        """
        绘制多源数据概览图
        
        展示流感、气象、搜索指数三类数据的时序趋势。
        """
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        
        dates = pd.to_datetime(df['date']) if 'date' in df.columns else range(len(df))
        
        # 流感数据
        ax1 = axes[0]
        if 'ili_rate' in df.columns:
            ax1.plot(dates, df['ili_rate'], color='#E53935', linewidth=1.5, label='ILI 率 (%)')
        if 'positive_rate' in df.columns:
            ax1_twin = ax1.twinx()
            ax1_twin.plot(dates, df['positive_rate'], color='#AB47BC', 
                         linewidth=1.5, alpha=0.7, label='阳性率 (%)')
            ax1_twin.set_ylabel('阳性率 (%)', fontsize=11, color='#AB47BC')
            ax1_twin.legend(loc='upper left', fontsize=10)
        ax1.set_ylabel('ILI 率 (%)', fontsize=11, color='#E53935')
        ax1.set_title('流感监测数据', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 气象数据
        ax2 = axes[1]
        if 'temperature' in df.columns:
            ax2.plot(dates, df['temperature'], color='#FF7043', linewidth=1.2, label='温度 (°C)')
        if 'humidity' in df.columns:
            ax2_twin = ax2.twinx()
            ax2_twin.plot(dates, df['humidity'], color='#42A5F5', 
                         linewidth=1.2, alpha=0.7, label='湿度 (%)')
            ax2_twin.set_ylabel('湿度 (%)', fontsize=11, color='#42A5F5')
            ax2_twin.legend(loc='upper left', fontsize=10)
        ax2.set_ylabel('温度 (°C)', fontsize=11, color='#FF7043')
        ax2.set_title('气象数据', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 搜索指数
        ax3 = axes[2]
        search_cols = ['flu_search_index', 'cold_search_index', 'fever_search_index']
        search_colors = ['#26A69A', '#7E57C2', '#FFA726']
        search_labels = ['流感搜索', '感冒搜索', '发烧搜索']
        for col, color, label in zip(search_cols, search_colors, search_labels):
            if col in df.columns:
                ax3.plot(dates, df[col], color=color, linewidth=1.2, alpha=0.8, label=label)
        ax3.set_ylabel('搜索指数', fontsize=11)
        ax3.set_title('搜索指数数据', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        fig.suptitle('多源数据时序概览', fontsize=16, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(self.save_dir, 'data_overview.png')
        plt.savefig(path)
        plt.close()
        print(f"  [保存] {path}")
        return path
    
    def plot_correlation_matrix(self, df: pd.DataFrame, 
                                 feature_cols: List[str]) -> str:
        """绘制特征相关性矩阵"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        cols_available = [c for c in feature_cols if c in df.columns]
        corr = df[cols_available].corr()
        
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                   cmap='RdBu_r', center=0, ax=ax,
                   square=True, linewidths=0.5,
                   vmin=-1, vmax=1)
        
        ax.set_title('特征相关性矩阵', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        path = os.path.join(self.save_dir, 'correlation_matrix.png')
        plt.savefig(path)
        plt.close()
        print(f"  [保存] {path}")
        return path
    
    def plot_multi_model_predictions(self, actuals: np.ndarray,
                                      predictions_dict: Dict[str, np.ndarray]) -> str:
        """
        绘制多模型预测对比图
        
        Args:
            actuals: 真实值
            predictions_dict: {model_name: predictions_array}
        """
        fig, ax = plt.subplots(figsize=(16, 7))
        
        x = range(len(actuals))
        ax.plot(x, actuals, color=COLORS['actual'], linewidth=2.5, 
               label='真实值', alpha=0.9, zorder=5)
        
        for model_name, preds in predictions_dict.items():
            color = COLORS.get(model_name, '#999')
            ax.plot(x, preds[:len(actuals)], color=color, linewidth=1.8, 
                   label=model_name, alpha=0.7, linestyle='--')
        
        ax.set_xlabel('时间步', fontsize=12)
        ax.set_ylabel('ILI 率 (%)', fontsize=12)
        ax.set_title('多模型预测结果对比', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(self.save_dir, 'multi_model_predictions.png')
        plt.savefig(path)
        plt.close()
        print(f"  [保存] {path}")
        return path
