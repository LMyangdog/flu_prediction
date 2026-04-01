"""
特征工程模块 — 构造衍生特征以增强模型表达能力

构造的特征类型：
    1. 时间特征：周数编码、月份编码、季节标志
    2. 滞后特征：ILI 的 1/2/4 周滞后
    3. 滚动统计：滚动均值、标准差
    4. 交叉特征：温度×湿度交互
    5. 搜索衍生：环比变化率

Author: flu_prediction project
"""

import numpy as np
import pandas as pd
from typing import List, Optional


class FeatureEngineer:
    """
    特征工程器
    
    在原始多源数据基础上构造衍生特征，
    增强模型对流感季节性和非线性关系的建模能力。
    """
    
    def __init__(self, config: dict):
        self.config = config
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行完整特征工程流水线
        
        Args:
            df: 原始合并数据
            
        Returns:
            添加衍生特征后的 DataFrame
        """
        print("\n[特征工程] 开始构建衍生特征...")
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # 1. 时间特征
        df = self._add_temporal_features(df)
        
        # 2. 滞后特征
        df = self._add_lag_features(df)
        
        # 3. 滚动统计特征
        df = self._add_rolling_features(df)
        
        # 4. 交叉特征
        df = self._add_cross_features(df)
        
        # 5. 搜索指数衍生特征
        df = self._add_search_derivatives(df)
        
        # 删除因滞后/滚动产生的 NaN 行
        initial_len = len(df)
        df = df.dropna().reset_index(drop=True)
        print(f"[特征工程] 完成！新增特征后删除 {initial_len - len(df)} 行 NaN")
        print(f"[特征工程] 最终特征数: {len(df.columns)}, 样本数: {len(df)}")
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        时间特征编码
        
        - week_sin/cos: 周数的正弦/余弦编码（捕捉周期性）
        - month_sin/cos: 月份编码
        - is_flu_season: 流感季标志（11月~次年3月）
        """
        # 周数正弦/余弦编码
        week = df['date'].dt.isocalendar().week.astype(int)
        df['week_sin'] = np.sin(2 * np.pi * week / 52)
        df['week_cos'] = np.cos(2 * np.pi * week / 52)
        
        # 月份编码
        month = df['date'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)
        
        # 流感季标志 (11月-次年3月)
        df['is_flu_season'] = ((month >= 11) | (month <= 3)).astype(float)
        
        print(f"  [时间特征] 添加 5 个: week_sin/cos, month_sin/cos, is_flu_season")
        return df
    
    def _add_lag_features(self, df: pd.DataFrame, 
                          lags: Optional[List[int]] = None) -> pd.DataFrame:
        """
        滞后特征 — ILI 率的历史值
        
        Args:
            lags: 滞后周数列表，默认 [1, 2, 4]
        """
        if lags is None:
            lags = [1, 2, 4]
        
        target = self.config.get('features', {}).get('target_col', 'ili_rate')
        
        for lag in lags:
            if target in df.columns:
                df[f'{target}_lag{lag}'] = df[target].shift(lag)
        
        print(f"  [滞后特征] 添加 {len(lags)} 个: {[f'{target}_lag{l}' for l in lags]}")
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame, 
                               windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        滚动统计特征 — 捕捉短期趋势
        
        Args:
            windows: 滚动窗口大小列表，默认 [4, 8]
        """
        if windows is None:
            windows = [4, 8]
        
        target = self.config.get('features', {}).get('target_col', 'ili_rate')
        count = 0
        
        for w in windows:
            if target in df.columns:
                df[f'{target}_roll_mean_{w}'] = df[target].rolling(w, min_periods=1).mean()
                df[f'{target}_roll_std_{w}'] = df[target].rolling(w, min_periods=1).std()
                count += 2
        
        print(f"  [滚动特征] 添加 {count} 个: 窗口大小 {windows}")
        return df
    
    def _add_cross_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        交叉特征 — 捕捉变量间的非线性交互
        
        - 温度×湿度交互（低温高湿利于流感传播）
        - 温湿指数
        """
        count = 0
        
        if 'temperature' in df.columns and 'humidity' in df.columns:
            # 温度×湿度交互
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
            count += 1
            
            # 温湿指数（简化的体感指标）
            df['comfort_index'] = 0.72 * df['temperature'] + 0.08 * df['humidity']
            count += 1
        
        if 'temperature' in df.columns and 'wind_speed' in df.columns:
            # 风寒指数（低温+大风）
            df['wind_chill'] = (
                13.12 + 0.6215 * df['temperature'] 
                - 11.37 * df['wind_speed'].clip(lower=0.1) ** 0.16 
                + 0.3965 * df['temperature'] * df['wind_speed'].clip(lower=0.1) ** 0.16
            )
            count += 1
        
        print(f"  [交叉特征] 添加 {count} 个")
        return df
    
    def _add_search_derivatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        搜索指数衍生特征 — 捕捉搜索热度的变化趋势
        
        - 环比变化率
        - 搜索加速度（变化率的变化）
        """
        search_cols = self.config.get('features', {}).get('search_cols', [])
        count = 0
        
        for col in search_cols:
            if col in df.columns:
                # 环比变化率
                df[f'{col}_pct_change'] = df[col].pct_change().fillna(0)
                df[f'{col}_pct_change'] = df[f'{col}_pct_change'].clip(-5, 5)  # 限制极端值
                count += 1
                
                # 搜索加速度
                df[f'{col}_acceleration'] = df[f'{col}_pct_change'].diff().fillna(0)
                df[f'{col}_acceleration'] = df[f'{col}_acceleration'].clip(-5, 5)
                count += 1
        
        print(f"  [搜索衍生] 添加 {count} 个")
        return df


if __name__ == "__main__":
    import yaml
    
    with open("config/config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    df = pd.read_csv("data/processed/merged_dataset.csv")
    
    engineer = FeatureEngineer(config)
    df_featured = engineer.transform(df)
    
    print(f"\n特征工程后的列：")
    print(df_featured.columns.tolist())
    print(f"\n数据形状: {df_featured.shape}")
