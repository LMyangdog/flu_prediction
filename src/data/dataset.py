"""
PyTorch Dataset 定义 — 适配 iTransformer 通道独立机制

核心设计：
    输入张量形状为 (num_variables, lookback_window)，
    即每个变量的完整历史序列作为独立 Token 输入 iTransformer。

Author: flu_prediction project
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


class FluDataset(Dataset):
    """
    流感多变量时间序列数据集
    
    适配 iTransformer 的通道独立机制:
        - 输入 x: (num_variables, lookback_window) — 每个变量的完整历史
        - 目标 y: (forecast_horizon,) — 未来流感趋势
    
    Args:
        X: 输入数据 (num_samples, num_variables, lookback_window)
        y: 目标数据 (num_samples, forecast_horizon)
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
        assert len(self.X) == len(self.y), \
            f"X 和 y 长度不一致: {len(self.X)} vs {len(self.y)}"
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
    
    @property
    def num_variables(self) -> int:
        """输入变量（通道）数"""
        return self.X.shape[1]
    
    @property
    def lookback_window(self) -> int:
        """历史回看窗口长度"""
        return self.X.shape[2]
    
    @property
    def forecast_horizon(self) -> int:
        """预测时域长度"""
        return self.y.shape[1]


def create_dataloaders(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       batch_size: int = 32,
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练/验证/测试 DataLoader
    
    Args:
        batch_size: 批大小
        num_workers: 数据加载线程数
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = FluDataset(X_train, y_train)
    val_dataset = FluDataset(X_val, y_val)
    test_dataset = FluDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n[DataLoader] 创建完成:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"  变量数: {train_dataset.num_variables}, "
          f"回看窗口: {train_dataset.lookback_window}, "
          f"预测步长: {train_dataset.forecast_horizon}")
    
    return train_loader, val_loader, test_loader
