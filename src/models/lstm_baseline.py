"""
LSTM 基准模型 — 用于与 iTransformer 进行对比实验

传统 LSTM 方法：
    将所有变量在同一时间步拼接输入，按时间步序列化处理。
    这正是开题报告中提到的"强行将同一时间步的异质特征绑定拼接"的做法。

Author: flu_prediction project
"""

import torch
import torch.nn as nn
from typing import Optional


class LSTMBaseline(nn.Module):
    """
    LSTM 基准模型
    
    与 iTransformer 的关键区别：
    - LSTM: 按时间步序列处理，同一时间步的所有变量拼接
    - iTransformer: 按变量独立处理，每个变量的完整时序作为独立 Token
    
    Args:
        num_variables: 输入变量数
        lookback_window: 历史回看窗口
        forecast_horizon: 预测时域
        hidden_dim: LSTM 隐藏维度
        num_layers: LSTM 层数
        dropout: Dropout 概率
        bidirectional: 是否双向 LSTM
    """
    
    def __init__(self,
                 num_variables: int,
                 lookback_window: int,
                 forecast_horizon: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 bidirectional: bool = False):
        super().__init__()
        
        self.num_variables = num_variables
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=num_variables,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 全连接输出层
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_horizon)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_variables, lookback_window) — iTransformer 格式
        Returns:
            predictions: (batch, forecast_horizon)
        """
        # 转换为 LSTM 输入格式: (batch, seq_len, input_size)
        x = x.transpose(1, 2)  # (B, L, C)
        
        # LSTM 前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (B, L, H)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # (B, H)
        
        # 全连接投影
        predictions = self.fc(last_output)  # (B, horizon)
        
        return predictions
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_lstm(config: dict, num_variables: int) -> LSTMBaseline:
    """根据配置构建 LSTM 基准模型"""
    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {}).get('lstm', {})
    
    model = LSTMBaseline(
        num_variables=num_variables,
        lookback_window=data_cfg.get('lookback_window', 12),
        forecast_horizon=data_cfg.get('forecast_horizon', 4),
        hidden_dim=model_cfg.get('hidden_dim', 128),
        num_layers=model_cfg.get('num_layers', 2),
        dropout=model_cfg.get('dropout', 0.2),
        bidirectional=model_cfg.get('bidirectional', False),
    )
    
    print(f"[LSTM] 模型构建完成, 参数量: {model.count_parameters():,}")
    return model
