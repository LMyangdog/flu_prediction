"""
ARIMA 基准模型 — 传统统计方法对比

单变量 ARIMA 模型，仅使用 ILI 病例数的历史数据进行预测。
作为深度学习方法的传统统计基准。

Author: flu_prediction project
"""

import numpy as np
import pandas as pd
import warnings
from typing import Tuple, Optional

warnings.filterwarnings('ignore')


class ARIMABaseline:
    """
    ARIMA 基准模型
    
    特点：
    - 仅使用目标变量（ILI 病例数）的单变量历史序列
    - 不利用多源数据（气象、搜索指数）
    - 适用于对比展示多源数据融合的价值
    
    参数：
        order: ARIMA(p, d, q) 参数
        seasonal_order: 季节性参数 (P, D, Q, s)
    """
    
    def __init__(self, order: Tuple[int, int, int] = (2, 1, 2),
                 seasonal_order: Optional[Tuple[int, int, int, int]] = (1, 1, 1, 52)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted = None
    
    def fit(self, train_series: np.ndarray):
        """
        拟合 ARIMA 模型
        
        Args:
            train_series: 训练序列 (1D array)
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        try:
            self.model = SARIMAX(
                train_series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.fitted = self.model.fit(disp=False, maxiter=200)
            print(f"[ARIMA] 模型拟合完成. AIC: {self.fitted.aic:.2f}")
        except Exception as e:
            print(f"[ARIMA] SARIMAX 拟合失败，降级为简单 ARIMA: {e}")
            from statsmodels.tsa.arima.model import ARIMA as SimpleARIMA
            self.model = SimpleARIMA(train_series, order=self.order)
            self.fitted = self.model.fit()
    
    def predict(self, steps: int) -> np.ndarray:
        """
        预测未来 steps 步
        
        Args:
            steps: 预测步数
            
        Returns:
            预测值数组
        """
        if self.fitted is None:
            raise RuntimeError("模型尚未拟合，请先调用 fit()")
        
        forecast = self.fitted.forecast(steps=steps)
        return np.array(forecast)
    
    def fit_predict_rolling(self, full_series: np.ndarray, 
                            train_size: int,
                            horizon: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        滚动预测 — 模拟实时预测场景
        
        每次用历史数据拟合模型，预测未来 horizon 步，
        然后滚动前移，适用于与深度学习模型的公平对比。
        
        Args:
            full_series: 完整历史序列
            train_size: 初始训练集大小
            horizon: 预测步长
            
        Returns:
            (predictions, actuals) 
        """
        predictions = []
        actuals = []
        
        from statsmodels.tsa.arima.model import ARIMA as SimpleARIMA
        
        n_predictions = (len(full_series) - train_size) // horizon
        
        for i in range(n_predictions):
            start = 0
            end = train_size + i * horizon
            
            train = full_series[:end]
            actual = full_series[end:end + horizon]
            
            if len(actual) < horizon:
                break
            
            try:
                model = SimpleARIMA(train, order=self.order)
                fitted = model.fit()
                pred = fitted.forecast(steps=horizon)
                predictions.append(pred)
                actuals.append(actual)
            except Exception as e:
                # 尝试降级为极简 ARIMA(0,1,1)
                try:
                    fallback_model = SimpleARIMA(train, order=(0, 1, 1))
                    fallback_fitted = fallback_model.fit()
                    pred = fallback_fitted.forecast(steps=horizon)
                    predictions.append(pred)
                    actuals.append(actual)
                except Exception:
                    # 如果仍失败，该窗口预测跳过，不应强行用最后一天真实值造假填充
                    print(f"[ARIMA] 窗口 {i} 预测失败，跳过该窗口。")
                    pass
        
        if predictions:
            return np.array(predictions), np.array(actuals)
        return np.array([]), np.array([])
    
    @staticmethod
    def auto_select_order(series: np.ndarray, max_p: int = 5, max_q: int = 5) -> Tuple[int, int, int]:
        """
        自动选择 ARIMA 阶数 (最小 AIC)
        
        Args:
            series: 时间序列
            max_p, max_q: 搜索范围上界
            
        Returns:
            最优 (p, d, q) 参数
        """
        from statsmodels.tsa.arima.model import ARIMA as SimpleARIMA
        from statsmodels.tsa.stattools import adfuller
        
        # 确定差分阶数 d
        d = 0
        temp = series.copy()
        for _ in range(2):
            result = adfuller(temp, autolag='AIC')
            if result[1] < 0.05:
                break
            temp = np.diff(temp)
            d += 1
        
        # 网格搜索 p, q
        best_aic = np.inf
        best_order = (1, d, 1)
        
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    model = SimpleARIMA(series, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except Exception:
                    continue
        
        print(f"[ARIMA Auto] 最优阶数: {best_order}, AIC: {best_aic:.2f}")
        return best_order
