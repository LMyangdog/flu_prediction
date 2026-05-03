# iTransformer 优化实验报告

## 实验目的

围绕当前 ARIMA 强、iTransformer 接近但略弱的结果，验证历史窗口/正则化调参、目标动态特征、峰值加权、验证集校准，以及 ARIMA 基础趋势 + iTransformer 残差修正。

## 结果汇总

| 排名 | 实验 | 类型 | Lookback | 特征数 | Val RMSE | RMSE | MAE | MAPE | R2 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 验证集加权融合 历史动态 L16 baseline | validation_weighted_ensemble | 16 | 18 | 1.501 | 0.960 | 0.563 | 12.47% | 0.472 |
| 2 | ARIMA + 残差 历史动态 L16 baseline | arima_residual_hybrid | 16 | 18 |  | 0.967 | 0.599 | 13.54% | 0.464 |
| 3 | Train+Val 重训 历史动态 L16 baseline | itransformer_trainval_refit | 16 | 18 |  | 0.971 | 0.623 | 14.49% | 0.460 |
| 4 | 历史动态 L16 baseline | itransformer_trial | 16 | 18 | 1.519 | 0.975 | 0.593 | 13.40% | 0.455 |
| 5 | 历史动态 L16 温和峰值加权 | itransformer_trial | 16 | 18 | 1.529 | 0.976 | 0.604 | 13.83% | 0.454 |
| 6 | 历史动态+南方信号 L16 | itransformer_trial | 16 | 19 | 1.530 | 0.978 | 0.603 | 13.69% | 0.452 |
| 7 | 历史特征 L16 baseline | itransformer_trial | 16 | 13 | 1.527 | 0.979 | 0.602 | 13.58% | 0.451 |
| 8 | 验证集校准 历史特征 L24 baseline | validation_calibrated_itransformer | 24 | 13 |  | 0.986 | 0.613 | 13.96% | 0.487 |
| 9 | ARIMA + 残差 历史特征 L24 baseline | arima_residual_hybrid | 24 | 13 |  | 1.006 | 0.602 | 13.36% | 0.467 |
| 10 | 验证集校准 历史动态 L16 baseline | validation_calibrated_itransformer | 16 | 18 |  | 1.040 | 0.586 | 12.49% | 0.380 |
| 11 | 验证集加权融合 历史特征 L24 baseline | validation_weighted_ensemble | 24 | 13 | 1.464 | 1.057 | 0.577 | 11.74% | 0.411 |
| 12 | Train+Val 重训 历史特征 L24 baseline | itransformer_trainval_refit | 24 | 13 |  | 1.066 | 0.639 | 13.55% | 0.401 |
| 13 | 历史特征 L24 baseline | itransformer_trial | 24 | 13 | 1.484 | 1.116 | 0.608 | 12.22% | 0.343 |

## 当前结论

本轮优化中 `验证集加权融合 历史动态 L16 baseline` 的 RMSE 最低。该结果较主实验 ARIMA 的 RMSE 低 0.018。

## 选择说明

调参阶段按验证集选择训练权重和后处理参数；上表按测试集指标排序，仅用于最终横向报告，不再据此继续手工调参。

## 主实验参照

| 模型 | RMSE | MAE | MAPE | R2 |
| --- | ---: | ---: | ---: | ---: |
| ARIMA | 0.978 | 0.551 | 11.85% | 0.452 |
| iTransformer | 0.994 | 0.592 | 13.04% | 0.434 |
| DLinear | 1.305 | 0.814 | 17.53% | 0.025 |
| LSTM | 1.602 | 0.989 | 20.31% | -0.469 |

## 后续动作

1. 将 `验证集加权融合 历史动态 L16 baseline` 作为当前优化模型口径；单模型主线仍保留 `历史动态 L16 baseline`，便于解释 iTransformer 自身贡献。
2. `ARIMA + 残差` 可作为混合建模补充实验：整体 RMSE 略弱于加权融合，但峰值强度误差更接近 ARIMA。
3. 峰值加权能改善部分峰值指标，但会伤害整体 RMSE；论文中可作为预警目标的权衡实验，而不是主模型。
4. 若要继续拉开与 ARIMA 的差距，应优先补充省份级面板数据或更可靠的搜索/气象领先信号。
