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

import json
import os
import pickle
from typing import Tuple, Dict

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
        self.target_col = config.get('features', {}).get('target_col', 'ili_cases')
        self.split_metadata: Dict[str, object] = {}
    
    def process(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                   np.ndarray, np.ndarray, np.ndarray]:
        """
        完整预处理流水线
        
        处理流程：
            选择特征 → 缺失值/异常值处理 → 时序全分割(防泄露) → 归一化(仅拟合Train) → 分别滑动窗口切片
        """
        print("\n" + "=" * 60)
        print("数据预处理流水线 (严格无泄露版)")
        print("=" * 60)
        
        if 'date' in df.columns:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

        # Step 1: 选择特征列
        df = self._select_features(df)
        print(f"[Step 1] 特征选择完成: {self.feature_cols}")
        
        # Step 2: 缺失值处理
        df = self._handle_missing(df)
        
        # Step 3: 异常值处理
        df = self._handle_outliers(df)
        print(f"[Step 3] 异常值处理完成. 数据量: {len(df)}")

        # Step 4: [修复] 物理隔离切分 DataFrame
        data_cfg = self.config.get('data', {})
        train_ratio = data_cfg.get('train_ratio', 0.7)
        val_ratio = data_cfg.get('val_ratio', 0.15)
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        df_train = df.iloc[:train_end].copy()
        df_val = df.iloc[train_end:val_end].copy()
        df_test = df.iloc[val_end:].copy()

        self.split_metadata = {
            'rows': int(n),
            'feature_count': int(len(self.feature_cols)),
            'train_rows': int(len(df_train)),
            'val_rows': int(len(df_val)),
            'test_rows': int(len(df_test)),
            'lookback_window': int(data_cfg.get('lookback_window', 12)),
            'forecast_horizon': int(data_cfg.get('forecast_horizon', 4)),
        }
        if 'date' in df.columns:
            self.split_metadata['train_date_range'] = {
                'start': str(pd.to_datetime(df_train['date']).min().date()),
                'end': str(pd.to_datetime(df_train['date']).max().date()),
            }
            self.split_metadata['val_date_range'] = {
                'start': str(pd.to_datetime(df_val['date']).min().date()),
                'end': str(pd.to_datetime(df_val['date']).max().date()),
            }
            self.split_metadata['test_date_range'] = {
                'start': str(pd.to_datetime(df_test['date']).min().date()),
                'end': str(pd.to_datetime(df_test['date']).max().date()),
            }
        
        # Step 5: [修复] 归一化 (必须基于 df_train 拟合)
        data_train, data_val, data_test = self._strict_normalize(df_train, df_val, df_test)

        # Step 6: 独立滑动窗口切片 (彻底阻断边缘重叠泄露)
        X_train, y_train = self._create_sequences(data_train)
        X_val, y_val = self._create_sequences(data_val)
        X_test, y_test = self._create_sequences(data_test)
        
        splits = (X_train, y_train, X_val, y_val, X_test, y_test)
        
        print(f"[Step 6] 严格切分完成:")
        print(f"  - Train: {X_train.shape[0]} samples")
        print(f"  - Val:   {X_val.shape[0]} samples")
        print(f"  - Test:  {X_test.shape[0]} samples")
        
        # 保存预处理后的数据
        self._save_splits(splits, df)
        
        return splits
        
    def _strict_normalize(self, df_train, df_val, df_test):
        """严格基于 Train 拟合归一化，并应用于其它集"""
        train_data = df_train[self.feature_cols].values.copy()
        val_data = df_val[self.feature_cols].values.copy()
        test_data = df_test[self.feature_cols].values.copy()
        
        for i, col in enumerate(self.feature_cols):
            scaler = MinMaxScaler(feature_range=(0, 1))
            # 仅在 Train 上 .fit
            scaler.fit(train_data[:, i:i+1])
            
            # 分别 .transform
            train_data[:, i:i+1] = scaler.transform(train_data[:, i:i+1])
            val_data[:, i:i+1] = scaler.transform(val_data[:, i:i+1])
            test_data[:, i:i+1] = scaler.transform(test_data[:, i:i+1])
            
            self.scalers[col] = scaler
        
        return train_data, val_data, test_data
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """选择用于建模的特征列"""
        feat_cfg = self.config.get('features', {})
        
        use_engineered_features = feat_cfg.get('use_engineered_features', True)
        exclude_cols = set(feat_cfg.get('exclude_from_training', ['year', 'week']))
        include_cols = feat_cfg.get('include_feature_cols')

        if include_cols:
            numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
            self.feature_cols = [
                col for col in include_cols
                if col in df.columns and col in numeric_cols and col not in exclude_cols
            ]
        elif use_engineered_features:
            numeric_cols = [
                col for col in df.select_dtypes(include=[np.number]).columns
                if col not in exclude_cols
            ]
            self.feature_cols = [col for col in df.columns if col in numeric_cols]
        else:
            flu_cols = feat_cfg.get('flu_cols', ['ili_cases', 'positive_count_monthly'])
            weather_cols = feat_cfg.get('weather_cols', ['temperature', 'humidity', 'wind_speed', 'pressure'])
            search_cols = feat_cfg.get('search_cols', ['flu_search_index', 'cold_search_index', 'fever_search_index'])

            all_feature_cols = flu_cols + weather_cols + search_cols
            self.feature_cols = [col for col in all_feature_cols if col in df.columns]

        # 确保目标列在特征列的第一个位置
        if self.target_col in self.feature_cols:
            self.feature_cols.remove(self.target_col)
            self.feature_cols.insert(0, self.target_col)
        else:
            raise ValueError(f"目标列 `{self.target_col}` 不在训练特征中，无法继续训练。")

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
        归一化 — 基于训练集拟合，防止未来数据泄漏
        """
        data = df[self.feature_cols].values.copy()
        
        data_cfg = self.config.get('data', {})
        train_ratio = data_cfg.get('train_ratio', 0.7)
        # 近似计算训练集终点，仅使用这部分进行归一化拟合，避免使用测试集分布造假
        train_end_idx = int(len(data) * train_ratio)
        
        for i, col in enumerate(self.feature_cols):
            scaler = MinMaxScaler(feature_range=(0, 1))
            # 仅在训练集上拟合
            scaler.fit(data[:train_end_idx, i:i+1])
            # 全局变换
            data[:, i:i+1] = scaler.transform(data[:, i:i+1])
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
            
            # 目标: 仅预测目标变量（默认 ili_cases，第0列）
            y = data[i + lookback:i + lookback + horizon, 0]
            
            X_list.append(x)
            y_list.append(y)
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        if len(X) == 0 or len(y) == 0:
            raise ValueError(
                "当前数据量不足以切出有效时序样本，请检查 lookback_window / forecast_horizon 设置，"
                "以及真实数据是否完整。"
            )
        
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

        with open(os.path.join(splits_dir, 'split_metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(self.split_metadata, f, ensure_ascii=False, indent=2)
        
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
