"""
数据预处理模块 — 清洗、对齐、归一化

功能：
    1. 缺失值处理（线性插值 + 前向填充）
    2. 异常值检测与剔除（IQR 方法）
    3. 归一化（Min-Max / Z-Score）
    4. 滑动窗口切片
    5. 训练/验证/测试集划分

Author: flu_prediction project
"""

import os
import pickle
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataPreprocessor:
    """
    多源数据预处理器
    
    处理流程：
        原始数据 → 缺失值填充 → 异常值处理 → 归一化 → 滑动窗口 → 数据集划分
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.scalers: Dict[str, object] = {}
        self.feature_cols = []
        self.target_col = config.get('features', {}).get('target_col', 'ili_rate')
    
    def process(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                   np.ndarray, np.ndarray, np.ndarray]:
        """
        完整预处理流水线
        
        Args:
            df: 原始合并数据 DataFrame
            
        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test) 
            X shape: (num_samples, num_variables, lookback_window)
            y shape: (num_samples, forecast_horizon)
        """
        print("\n" + "=" * 60)
        print("数据预处理流水线")
        print("=" * 60)
        
        # Step 1: 选择特征列
        df = self._select_features(df)
        print(f"[Step 1] 特征选择完成: {self.feature_cols}")
        
        # Step 2: 缺失值处理
        df = self._handle_missing(df)
        print(f"[Step 2] 缺失值处理完成. 剩余缺失: {df[self.feature_cols].isnull().sum().sum()}")
        
        # Step 3: 异常值处理
        df = self._handle_outliers(df)
        print(f"[Step 3] 异常值处理完成. 数据量: {len(df)}")
        
        # Step 4: 归一化
        df_scaled = self._normalize(df)
        print(f"[Step 4] 归一化完成. 使用 Min-Max Scaling")
        
        # Step 5: 滑动窗口切片
        X, y = self._create_sequences(df_scaled)
        print(f"[Step 5] 滑动窗口完成. X shape: {X.shape}, y shape: {y.shape}")
        
        # Step 6: 划分数据集
        splits = self._split_data(X, y)
        X_train, y_train, X_val, y_val, X_test, y_test = splits
        print(f"[Step 6] 数据划分完成:")
        print(f"  - Train: {X_train.shape[0]} samples")
        print(f"  - Val:   {X_val.shape[0]} samples")
        print(f"  - Test:  {X_test.shape[0]} samples")
        
        # 保存预处理后的数据
        self._save_splits(splits, df)
        
        return splits
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """选择用于建模的特征列"""
        feat_cfg = self.config.get('features', {})
        
        flu_cols = feat_cfg.get('flu_cols', ['ili_rate', 'positive_rate'])
        weather_cols = feat_cfg.get('weather_cols', ['temperature', 'humidity', 'wind_speed', 'pressure'])
        search_cols = feat_cfg.get('search_cols', ['flu_search_index', 'cold_search_index', 'fever_search_index'])
        
        # 收集存在于数据中的特征列
        all_feature_cols = flu_cols + weather_cols + search_cols
        self.feature_cols = [col for col in all_feature_cols if col in df.columns]
        
        # 确保目标列在特征列的第一个位置
        if self.target_col in self.feature_cols:
            self.feature_cols.remove(self.target_col)
            self.feature_cols.insert(0, self.target_col)
        
        return df
    
    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """缺失值处理"""
        df = df.copy()
        
        for col in self.feature_cols:
            if col in df.columns:
                # 先用线性插值
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
                # 再用前向/后向填充处理边界
                df[col] = df[col].ffill().bfill()
        
        # 删除仍有缺失的行
        df = df.dropna(subset=self.feature_cols).reset_index(drop=True)
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, factor: float = 3.0) -> pd.DataFrame:
        """
        异常值处理 — IQR 方法
        
        对每个特征，将超出 [Q1 - factor*IQR, Q3 + factor*IQR] 范围的值
        裁剪到边界值（Winsorize），而非直接删除行。
        """
        df = df.copy()
        
        for col in self.feature_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - factor * IQR
                upper = Q3 + factor * IQR
                
                n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                if n_outliers > 0:
                    df[col] = df[col].clip(lower, upper)
                    print(f"  [异常值] {col}: 裁剪 {n_outliers} 个极端值")
        
        return df
    
    def _normalize(self, df: pd.DataFrame) -> np.ndarray:
        """
        归一化 — 每个特征独立进行 Min-Max 缩放到 [0, 1]
        
        保存 scaler 以便后续反归一化
        """
        data = df[self.feature_cols].values.copy()
        
        for i, col in enumerate(self.feature_cols):
            scaler = MinMaxScaler(feature_range=(0, 1))
            data[:, i:i+1] = scaler.fit_transform(data[:, i:i+1])
            self.scalers[col] = scaler
        
        return data
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        滑动窗口切片 — 创建输入-目标序列对
        
        输出格式适配 iTransformer 通道独立机制：
        - X: (num_samples, num_variables, lookback_window)
        - y: (num_samples, forecast_horizon)
        """
        data_cfg = self.config.get('data', {})
        lookback = data_cfg.get('lookback_window', 12)
        horizon = data_cfg.get('forecast_horizon', 4)
        
        X_list, y_list = [], []
        
        for i in range(len(data) - lookback - horizon + 1):
            # 输入: 所有变量的历史窗口
            # shape: (lookback, num_vars) → 转置为 (num_vars, lookback)
            x = data[i:i + lookback, :].T
            
            # 目标: 仅预测目标变量（ILI率，第0列）
            y = data[i + lookback:i + lookback + horizon, 0]
            
            X_list.append(x)
            y_list.append(y)
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        
        return X, y
    
    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """
        时间序列划分 — 按时间顺序划分（不打乱）
        
        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        data_cfg = self.config.get('data', {})
        train_ratio = data_cfg.get('train_ratio', 0.7)
        val_ratio = data_cfg.get('val_ratio', 0.15)
        
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def _save_splits(self, splits: Tuple, original_df: pd.DataFrame):
        """保存预处理后的数据和 scaler"""
        splits_dir = self.config.get('data', {}).get('splits_dir', 'data/splits')
        os.makedirs(splits_dir, exist_ok=True)
        
        X_train, y_train, X_val, y_val, X_test, y_test = splits
        
        np.save(os.path.join(splits_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(splits_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(splits_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(splits_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(splits_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(splits_dir, 'y_test.npy'), y_test)
        
        # 保存 scaler
        with open(os.path.join(splits_dir, 'scalers.pkl'), 'wb') as f:
            pickle.dump(self.scalers, f)
        
        # 保存特征列名
        with open(os.path.join(splits_dir, 'feature_cols.pkl'), 'wb') as f:
            pickle.dump(self.feature_cols, f)
        
        print(f"\n[保存] 预处理数据已保存至 {splits_dir}/")
    
    def inverse_transform_target(self, normalized_values: np.ndarray) -> np.ndarray:
        """将归一化的目标变量反变换为原始尺度"""
        if self.target_col in self.scalers:
            scaler = self.scalers[self.target_col]
            return scaler.inverse_transform(normalized_values.reshape(-1, 1)).flatten()
        return normalized_values


if __name__ == "__main__":
    import yaml
    
    with open("config/config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载合并后的数据
    df = pd.read_csv("data/processed/merged_dataset.csv")
    
    preprocessor = DataPreprocessor(config)
    X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.process(df)
    
    print(f"\n最终数据形状:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
