# 最终数据复核说明

## 一句话结论

当前正式实验数据已切换为国家流感中心北方省份真实周度 ILI% 序列，气象与百度指数均使用北方代表城市聚合口径。当前 CSV 已完成缺失补齐：`ili_rate` 剩余缺失 0 周，`positive_rate` 剩余缺失 0 周；`ili_rate` 有来源补齐 3 周，`positive_rate` 插值补齐 108 周。

## 流感周报数据

- 原始行数：849
- 原始时间范围：2010-01-04 至 2026-04-13
- 解析状态：{'ok': 739, 'imputed': 110}
- 补齐记录：ili_rate=3 周，positive_rate=108 周
- 周度合并数据行数：797
- 周度合并数据时间范围：2011-01-03 至 2026-04-13
- 特征工程后进入训练切分的行数：793
- 特征工程后训练/验证/测试覆盖：2011-01-31 至 2026-04-13

### ili_rate 缺失周

- 无

### ili_rate 补齐来源

- date=2017-10-09；year=2017；week=41；ili_rate=2.4；imputation_method=ili_rate_cross_report；imputation_source=https://ivdc.chinacdc.cn/cnic/zyzx/lgzb/202010/t20201018_222148.htm；imputation_note=CNIC 2020 week 41 report: 2017 same-period northern ILI% = 2.4%
- date=2017-10-16；year=2017；week=42；ili_rate=2.6；imputation_method=ili_rate_cross_report；imputation_source=https://ivdc.chinacdc.cn/cnic/zyzx/lgzb/201711/t20171107_154704.htm；imputation_note=CNIC 2017 week 43 report: previous-week northern ILI% = 2.6%
- date=2021-12-13；year=2021；week=50；ili_rate=3.5；imputation_method=ili_rate_cross_report; positive_rate_linear_interpolation；imputation_source=https://ivdc.chinacdc.cn/cnic/zyzx/lgzb/202212/t20221222_263091.htm; neighboring CNIC positive_rate values；imputation_note=CNIC 2022/2023 week 50 reports: 2021 same-period northern ILI% = 3.5%; positive_rate is an audit-only field and is excluded from model features

### positive_rate 缺失说明

- `positive_rate` 剩余缺失 0 周；已通过相邻周线性插值补齐 108 周。当前训练默认不使用该字段，仅作为原始监测留痕与数据质量说明。

## 百度指数聚合复核

- 行数：5593
- 时间范围：2011-01-01 至 2026-04-24
- `fever_search_index_region_count`：min=8，max=8，nunique=1
- `cold_search_index_region_count`：min=8，max=8，nunique=1
- `flu_search_index_region_count`：min=8，max=8，nunique=1

结论：三个关键词每日均为 8 个代表地区参与聚合，当前 CSV 与 README/论文中的北方代表城市口径一致。

## 气象数据复核

- 行数：5594
- 时间范围：2011-01-01 至 2026-04-25
- 字段：temperature, humidity, wind_speed, pressure

## 论文写作边界

- 可以表述为北方省份周度 ILI% 趋势预测，不能表述为单一城市或区县病例预测。
- `positive_rate` 应说明为原始留痕字段，默认不进入训练特征。
- 百度指数聚合方式当前为 8 个北方代表地区简单平均；若后续改为人口权重，需要重新生成数据和结果。
