"""
DLinear 基准模型 — 简单线性分解方法

基于 "Are Transformers Effective for Time Series Forecasting?" (Zeng et al., 2023)
将时间序列分解为趋势项和季节项，分别用线性层预测。

作为轻量基准，验证 iTransformer 的非线性建模优势。

Author: flu_prediction project
"""

import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """滑动平均 — 提取趋势成分"""
    
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) or (batch, channels, seq_len)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out = self.avg(x)
        return out.squeeze(1) if out.shape[1] == 1 else out


class DLinear(nn.Module):
    """
    DLinear 基准模型
    
    核心思想：
    1. 将输入序列分解为趋势项 (Trend) 和残差项 (Residual)
    2. 对两部分分别用独立的线性层进行预测
    3. 将预测结果相加
    
    与 iTransformer 对比：
    - DLinear 不学习变量间的交互关系
    - DLinear 是纯线性映射，无法捕捉非线性模式
    
    Args:
        num_variables: 输入变量数
        lookback_window: 历史窗口
        forecast_horizon: 预测步长
        individual: 是否对每个变量使用独立的线性层
        kernel_size: 滑动平均窗口大小
    """
    
    def __init__(self,
                 num_variables: int,
                 lookback_window: int,
                 forecast_horizon: int,
                 individual: bool = True,
                 kernel_size: int = 3):
        super().__init__()
        
        self.num_variables = num_variables
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.individual = individual
        
        # 趋势提取
        self.moving_avg = MovingAvg(kernel_size)
        
        if individual:
            # 每个变量独立的线性层
            self.trend_linear = nn.ModuleList([
                nn.Linear(lookback_window, forecast_horizon)
                for _ in range(num_variables)
            ])
            self.residual_linear = nn.ModuleList([
                nn.Linear(lookback_window, forecast_horizon)
                for _ in range(num_variables)
            ])
        else:
            # 共享线性层
            self.trend_linear = nn.Linear(lookback_window, forecast_horizon)
            self.residual_linear = nn.Linear(lookback_window, forecast_horizon)
        
        # 最终投影（从所有变量到目标）
        self.output_fc = nn.Linear(num_variables * forecast_horizon, forecast_horizon)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_variables, lookback_window)
        Returns:
            predictions: (batch, forecast_horizon)
        """
        B, C, L = x.shape
        
        # 分解：趋势 + 残差
        trend = self.moving_avg(x)       # (B, C, L)
        residual = x - trend             # (B, C, L)
        
        if self.individual:
            trend_out = torch.stack([
                self.trend_linear[i](trend[:, i, :])
                for i in range(C)
            ], dim=1)  # (B, C, horizon)
            
            residual_out = torch.stack([
                self.residual_linear[i](residual[:, i, :])
                for i in range(C)
            ], dim=1)  # (B, C, horizon)
        else:
            trend_out = self.trend_linear(trend)
            residual_out = self.residual_linear(residual)
        
        # 合并趋势和残差预测
        combined = trend_out + residual_out  # (B, C, horizon)
        
        # 展平并投影到最终输出
        flat = combined.reshape(B, -1)  # (B, C * horizon)
        predictions = self.output_fc(flat)  # (B, horizon)
        
        return predictions
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_dlinear(config: dict, num_variables: int) -> DLinear:
    """根据配置构建 DLinear 基准模型"""
    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {}).get('dlinear', {})
    
    model = DLinear(
        num_variables=num_variables,
        lookback_window=data_cfg.get('lookback_window', 12),
        forecast_horizon=data_cfg.get('forecast_horizon', 4),
        individual=model_cfg.get('individual', True),
    )
    
    print(f"[DLinear] 模型构建完成, 参数量: {model.count_parameters():,}")
    return model
