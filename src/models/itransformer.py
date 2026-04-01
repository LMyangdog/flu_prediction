"""
iTransformer — 倒置 Transformer 用于多变量时间序列预测

核心创新：
    1. 通道独立嵌入 (Channel-Independent Embedding):
       将每个变量的完整时间序列作为独立 Token 嵌入到高维空间
       
    2. 倒置自注意力 (Inverted Self-Attention):
       在变量维度（而非时间维度）上计算注意力权重，
       学习不同数据源（流感、气象、搜索指数）间的跨维度关联
       
    3. 天然适配多源异构数据:
       不同类型的变量（数值型气象、文本衍生搜索数据、统计型监测数据）
       通过独立嵌入规避了异质特征间的相互干扰

架构示意:
    Input (B, C, L) → Channel Embedding (B, C, D) → 
    Transformer Encoder (B, C, D) → Projection (B, horizon)

    B: batch size
    C: num_variables (通道数)
    L: lookback_window (历史窗口)
    D: d_model (嵌入维度)

参考论文:
    "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"
    Yong Liu et al., ICLR 2024

Author: flu_prediction project
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ChannelEmbedding(nn.Module):
    """
    通道独立嵌入层
    
    将每个变量的完整时间序列 (L维) 映射到 d_model 维空间。
    每个变量独立嵌入，保持通道独立性。
    """
    
    def __init__(self, lookback_window: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(lookback_window, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_vars, lookback_window)
        Returns:
            tokens: (batch, num_vars, d_model)
        """
        tokens = self.linear(x)      # (B, C, D)
        tokens = self.norm(tokens)
        tokens = self.dropout(tokens)
        return tokens


class InvertedMultiHeadAttention(nn.Module):
    """
    倒置多头自注意力
    
    在变量维度上计算注意力：
    - Query、Key、Value 都来自变量 Token
    - 注意力权重表示变量间的相互影响程度
    - 例如：温度变量通过注意力影响 ILI 率的预测
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model({d_model}) 必须能被 n_heads({n_heads}) 整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # 保存注意力权重（用于可视化）
        self.attn_weights = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_vars, d_model)
        Returns:
            output: (batch, num_vars, d_model)
        """
        B, C, D = x.shape
        
        # 计算 Q, K, V
        Q = self.W_Q(x).view(B, C, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, C, dk)
        K = self.W_K(x).view(B, C, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, C, dk)
        V = self.W_V(x).view(B, C, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, C, dk)
        
        # 注意力计算 — 在变量维度上
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale     # (B, H, C, C)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 保存注意力权重
        self.attn_weights = attn_probs.detach().cpu()
        
        # 加权聚合
        context = torch.matmul(attn_probs, V)                               # (B, H, C, dk)
        context = context.transpose(1, 2).contiguous().view(B, C, D)         # (B, C, D)
        
        output = self.W_O(context)
        return output


class FeedForward(nn.Module):
    """前馈网络 — 逐通道特征变换"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, 
                 activation: str = 'gelu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class iTransformerBlock(nn.Module):
    """
    iTransformer 编码器块
    
    结构：倒置多头自注意力 → Add & Norm → 前馈网络 → Add & Norm
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        
        self.attention = InvertedMultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_vars, d_model)
        Returns:
            output: (batch, num_vars, d_model)
        """
        # 倒置自注意力 + 残差
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 前馈网络 + 残差
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class iTransformer(nn.Module):
    """
    iTransformer — 倒置 Transformer 用于多变量时间序列预测
    
    核心流程：
    1. 通道独立嵌入：每个变量的历史序列 (L) → Token (D)
    2. N 层倒置自注意力：学习变量间跨维度关联
    3. 输出投影：仅取目标变量 Token 投影到预测长度
    
    Args:
        num_variables: 输入变量数 (通道数)
        lookback_window: 历史回看窗口长度
        forecast_horizon: 预测时域长度
        d_model: 嵌入维度
        n_heads: 注意力头数
        n_layers: Transformer 层数
        d_ff: 前馈网络隐藏维度
        dropout: Dropout 概率
        activation: 激活函数 ('gelu' 或 'relu')
    """
    
    def __init__(self, 
                 num_variables: int,
                 lookback_window: int,
                 forecast_horizon: int,
                 d_model: int = 128,
                 n_heads: int = 8,
                 n_layers: int = 3,
                 d_ff: int = 256,
                 dropout: float = 0.1,
                 activation: str = 'gelu'):
        super().__init__()
        
        self.num_variables = num_variables
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model
        
        # 1. 通道独立嵌入
        self.embedding = ChannelEmbedding(lookback_window, d_model, dropout)
        
        # 2. 倒置 Transformer 编码器
        self.encoder_layers = nn.ModuleList([
            iTransformerBlock(d_model, n_heads, d_ff, dropout, activation)
            for _ in range(n_layers)
        ])
        
        # 3. 输出投影 — 独立抽取目标序列特征(索引为0)进行映射，防止多变量全连接层过拟合
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, forecast_horizon)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 均匀初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (batch, num_variables, lookback_window)
            
        Returns:
            predictions: (batch, forecast_horizon)
        """
        # Step 1: 通道独立嵌入
        tokens = self.embedding(x)  # (B, C, D)
        
        # Step 2: 倒置 Transformer 编码
        for layer in self.encoder_layers:
            tokens = layer(tokens)   # (B, C, D)
        
        # Step 3: 根据 iTransformer 论文机制，独立抽取目标变量（固定为特征列的第0位）投影
        # 而不是把含有噪声的所有通道 (B, C*D) 全部展平
        target_token = tokens[:, 0, :]                   # (B, D)
        predictions = self.output_projection(target_token) # (B, horizon)
        
        return predictions
    
    def get_attention_weights(self) -> list:
        """获取所有层的注意力权重（用于可视化）"""
        weights = []
        for layer in self.encoder_layers:
            if layer.attention.attn_weights is not None:
                # 取所有头的平均
                avg_attn = layer.attention.attn_weights.mean(dim=1)  # (B, C, C)
                weights.append(avg_attn)
        return weights
    
    def count_parameters(self) -> int:
        """计算可训练参数总数"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        return (f"iTransformer(\n"
                f"  num_variables={self.num_variables},\n"
                f"  lookback={self.lookback_window},\n"
                f"  horizon={self.forecast_horizon},\n"
                f"  d_model={self.d_model},\n"
                f"  params={self.count_parameters():,}\n"
                f")")


def build_itransformer(config: dict, num_variables: int) -> iTransformer:
    """
    根据配置文件构建 iTransformer 模型
    
    Args:
        config: 配置字典
        num_variables: 输入变量数
        
    Returns:
        iTransformer 模型实例
    """
    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {}).get('itransformer', {})
    
    model = iTransformer(
        num_variables=num_variables,
        lookback_window=data_cfg.get('lookback_window', 12),
        forecast_horizon=data_cfg.get('forecast_horizon', 4),
        d_model=model_cfg.get('d_model', 128),
        n_heads=model_cfg.get('n_heads', 8),
        n_layers=model_cfg.get('n_layers', 3),
        d_ff=model_cfg.get('d_ff', 256),
        dropout=model_cfg.get('dropout', 0.1),
        activation=model_cfg.get('activation', 'gelu'),
    )
    
    print(f"\n[iTransformer] 模型构建完成")
    print(f"  参数量: {model.count_parameters():,}")
    print(model)
    
    return model


if __name__ == "__main__":
    # 快速测试
    batch_size = 16
    num_vars = 9
    lookback = 12
    horizon = 4
    
    model = iTransformer(
        num_variables=num_vars,
        lookback_window=lookback,
        forecast_horizon=horizon,
    )
    
    print(model)
    
    # 前向传播测试
    x = torch.randn(batch_size, num_vars, lookback)
    y = model(x)
    print(f"\n输入: {x.shape}")
    print(f"输出: {y.shape}")
    
    # 注意力权重
    attn_weights = model.get_attention_weights()
    print(f"注意力层数: {len(attn_weights)}")
    for i, w in enumerate(attn_weights):
        print(f"  Layer {i}: {w.shape}")
