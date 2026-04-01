"""
训练器模块 — 统一的模型训练流水线

功能：
    - 支持 iTransformer、LSTM、DLinear 等模型的统一训练
    - 早停机制 (Early Stopping)
    - 学习率调度 (Cosine Annealing / ReduceOnPlateau)
    - TensorBoard 日志
    - 模型检查点保存与恢复
    - 梯度裁剪

Author: flu_prediction project
"""

import os
import time
import json
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

from src.utils.metrics import compute_all_metrics, format_metrics


class EarlyStopping:
    """
    早停机制 — 当验证损失不再下降时停止训练
    """
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-5, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss: float, epoch: int) -> bool:
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose and self.counter % 5 == 0:
                print(f"  [EarlyStopping] {self.counter}/{self.patience} "
                      f"(最佳 epoch: {self.best_epoch})")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        
        return False


class Trainer:
    """
    统一训练器
    
    支持多种模型的训练、验证和测试评估。
    
    Args:
        model: PyTorch 模型
        config: 配置字典
        model_name: 模型名称（用于日志和保存）
    """
    
    def __init__(self, model: nn.Module, config: dict, model_name: str = "model"):
        self.model = model
        self.config = config
        self.model_name = model_name
        
        train_cfg = config.get('training', {})
        
        # 设备
        device_str = train_cfg.get('device', 'cuda')
        if device_str == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"[Trainer] 使用 GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print(f"[Trainer] 使用 CPU")
        
        self.model = self.model.to(self.device)
        
        # 优化器
        self.optimizer = self._build_optimizer(train_cfg)
        
        # 学习率调度器
        self.scheduler = self._build_scheduler(train_cfg)
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 早停
        self.early_stopping = EarlyStopping(
            patience=train_cfg.get('patience', 15)
        )
        
        # 训练配置
        self.epochs = train_cfg.get('epochs', 200)
        self.max_grad_norm = train_cfg.get('max_grad_norm', 1.0)
        
        # 保存路径
        self.checkpoint_dir = train_cfg.get('checkpoint_dir', 'checkpoints')
        self.log_dir = train_cfg.get('log_dir', 'results/logs')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
        }
    
    def _build_optimizer(self, train_cfg: dict) -> torch.optim.Optimizer:
        """构建优化器"""
        lr = train_cfg.get('learning_rate', 0.001)
        wd = train_cfg.get('weight_decay', 0.0001)
        opt_name = train_cfg.get('optimizer', 'adam').lower()
        
        if opt_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=lr, 
                                    weight_decay=wd, momentum=0.9)
        else:
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
    
    def _build_scheduler(self, train_cfg: dict):
        """构建学习率调度器"""
        scheduler_name = train_cfg.get('scheduler', 'cosine').lower()
        epochs = train_cfg.get('epochs', 200)
        
        if scheduler_name == 'cosine':
            return CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
        elif scheduler_name == 'plateau':
            return ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)
        elif scheduler_name == 'step':
            return StepLR(self.optimizer, step_size=30, gamma=0.5)
        else:
            return CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        Returns:
            训练历史字典
        """
        print(f"\n{'=' * 60}")
        print(f"开始训练: {self.model_name}")
        print(f"{'=' * 60}")
        print(f"Epochs: {self.epochs}, 设备: {self.device}")
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(1, self.epochs + 1):
            # 训练
            train_loss = self._train_epoch(train_loader)
            
            # 验证
            val_loss = self._validate(val_loader)
            
            # 当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(current_lr)
            
            # 学习率调度
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, is_best=True)
            
            # 打印进度
            if epoch % 10 == 0 or epoch == 1:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:4d}/{self.epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Time: {elapsed:.0f}s")
            
            # 早停检查
            if self.early_stopping(val_loss, epoch):
                print(f"\n[Early Stopping] 在 Epoch {epoch} 停止训练")
                print(f"  最佳验证损失: {best_val_loss:.6f} "
                      f"(Epoch {self.early_stopping.best_epoch})")
                break
        
        total_time = time.time() - start_time
        print(f"\n训练完成！总时间: {total_time:.1f}s")
        print(f"最佳验证损失: {best_val_loss:.6f}")
        
        # 保存训练历史
        self._save_history()
        
        return self.history
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """单个 epoch 训练"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)
            
            loss.backward()
            
            # 梯度裁剪
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def _validate(self, val_loader: DataLoader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def evaluate(self, test_loader: DataLoader, scaler=None) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """
        测试集评估
        
        Args:
            test_loader: 测试数据加载器
            scaler: 目标变量的反归一化 scaler
            
        Returns:
            (metrics_dict, predictions, actuals)
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                
                predictions = self.model(batch_x)
                
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(batch_y.numpy())
        
        predictions = np.concatenate(all_preds, axis=0)
        actuals = np.concatenate(all_targets, axis=0)
        
        # 反归一化
        if scaler is not None:
            predictions_flat = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            actuals_flat = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
        else:
            predictions_flat = predictions.flatten()
            actuals_flat = actuals.flatten()
        
        # 计算指标
        metrics = compute_all_metrics(actuals_flat, predictions_flat)
        
        print(f"\n[{self.model_name}] 测试集评估结果:")
        print(format_metrics(metrics))
        
        return metrics, predictions_flat, actuals_flat
    
    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'model_name': self.model_name,
        }
        
        filename = f"{self.model_name}_best.pt" if is_best else f"{self.model_name}_epoch{epoch}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """加载模型检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[Trainer] 加载检查点: {path} (Epoch {checkpoint.get('epoch', '?')})")
    
    def _save_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.log_dir, f"{self.model_name}_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
