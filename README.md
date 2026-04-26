# 基于深度学习和多元数据的流感爆发趋势预测

## 项目介绍（答辩版）

本项目面向本科毕业设计答辩场景，构建了一个“真实数据采集 - 多源数据融合 - 深度学习预测 - 结果可视化展示”的流感趋势预测系统。研究对象为中国北方省份周度流感活动水平，预测目标为国家流感中心周报中的 `ili_rate`（哨点医院报告的流感样病例占比，ILI%）。系统最终使用流感监测、气象环境和搜索行为三类数据，训练 iTransformer 多变量时间序列模型，并与 LSTM、DLinear、ARIMA 等基线模型进行对比，用于说明模型在未来 1-4 周流感趋势预测中的表现。

### 数据集规模与来源

项目当前采用严格真实数据模式，所有正式实验均不使用模拟数据。数据来源和规模如下：

| 数据类型 | 数据来源 | 原始粒度 | 当前规模 | 用途 |
| --- | --- | --- | ---: | --- |
| 流感监测数据 | 中国国家流感中心《中国流感监测周报》 | 周度 | 849 周，2010-01-04 至 2026-04-13 | 提供预测目标 `ili_rate`，并保留 `positive_rate` 作为质量审计字段 |
| 气象数据 | Open-Meteo Archive API | 日度 | 5594 天，2011-01-01 至 2026-04-25 | 提供温度、湿度、风速、气压等外生变量 |
| 搜索指数数据 | 百度指数 | 日度 | 5593 天，2011-01-01 至 2026-04-24 | 提供“流感”“感冒”“发烧”等公众关注度变量 |
| 周度融合数据 | 三类数据按周对齐融合 | 周度 | 797 周，2011-01-03 至 2026-04-13 | 作为建模前的统一时序数据 |
| 特征工程后数据 | 滞后、滚动、季节和交互特征处理后 | 周度 | 793 周，2011-01-31 至 2026-04-13 | 进入训练、验证、测试切分 |

其中，气象和百度指数均采用北方代表城市聚合口径，覆盖北京、天津、济南、石家庄、沈阳、哈尔滨、西安、太原 8 个地区。这样既能与“北方省份流感监测”口径保持一致，又能引入气候变化和公众搜索行为对流感活动的辅助解释信息。

### 数据采集与补齐流程

流感数据通过 `scripts/fetch_flu_cn_weekly.py` 从国家流感中心周报页面自动采集。脚本优先解析官方 HTML 周报页面；当近年页面只提供附件或表格不完整时，继续解析 PDF 附件，并将页面和解析摘要缓存到本地，便于后续复核。气象数据由 Open-Meteo Archive API 按代表城市逐日拉取，再按日期聚合为北方地区均值。百度指数由于需要登录态或人工导出，项目将导出的日度关键词指数统一整理为标准 CSV，再按 8 个代表地区聚合。

数据清洗阶段主要完成四件事：第一，统一日期格式并按 ISO 周对齐；第二，将日度气象和搜索指数聚合到周度；第三，对流感周报解析缺口进行可追溯补齐；第四，生成数据质量报告用于论文和答辩说明。当前 `ili_rate` 已无缺失，其中 3 周缺失值根据国家流感中心后续或相邻周报中的明确数值补齐；`positive_rate` 已通过相邻周线性插值补齐 108 周，但该字段仅作为原始监测留痕和数据质量审计字段，默认不进入模型训练。

### 特征工程与训练方法

模型输入由 29 个特征组成，主要包括四类信息：流感历史序列、气象变量、百度搜索指数和时间序列衍生特征。衍生特征包括周/月周期编码、流感季标记、`ili_rate` 的 1 周、2 周、4 周滞后值，4 周和 8 周滚动均值/标准差，以及气象交互项和搜索指数变化率/加速度等。模型使用过去 16 周数据作为输入窗口，预测未来 4 周 `ili_rate`，适合展示“提前预警”和“趋势复盘”的毕业设计目标。

训练阶段采用严格时间顺序切分，避免未来信息泄漏：

| 划分 | 行数 | 时间范围 |
| --- | ---: | --- |
| 训练集 | 555 | 2011-01-31 至 2021-09-20 |
| 验证集 | 119 | 2021-09-27 至 2024-01-01 |
| 测试集 | 119 | 2024-01-08 至 2026-04-13 |

核心模型为 iTransformer。该模型将多变量时间序列中的变量维度作为注意力建模对象，更适合学习流感历史、气象和搜索行为之间的跨变量关系。训练配置为随机种子 42、最大 300 轮、批大小 32、Adam 优化器、初始学习率 0.001、权重衰减 0.01、余弦学习率调度和早停机制。训练完成后，系统输出多模型对比、多步预测误差、消融实验、预测曲线和 Web 可视化所需的全部结果文件。

### 实验结果与展示方式

最终测试集上，iTransformer 的整体结果为 RMSE 0.772、MAE 0.491、MAPE 11.51%、R2 0.538，在当前数据版本下优于 LSTM、DLinear，并与 ARIMA 形成清晰对照。项目还完成了特征消融实验，用于解释“仅流感历史”“流感+气象”“流感+搜索”“三源融合”四种输入组合的差异。Web 页面位于 `web/app.py`，面向答辩展示提供数据概览、模型指标、预测曲线、预警复盘和论文结论支撑，便于导师快速理解项目的数据来源、建模方法和实验可信度。

中国海洋大学信息科学与工程学部计算机科学与技术本科毕业设计项目。

本项目围绕开题报告中的目标，使用 `iTransformer` 对流感监测、气象与搜索指数三类多源时序数据进行融合预测。当前版本已将研究口径从“北京市平谷区代理序列”切换为：

- 流感主序列：`中国国家流感中心北方省份周度流感监测数据`
- 目标变量：`ili_rate`，即北方省份哨点医院报告的 ILI%
- 辅助流感字段：`positive_rate` 保留在原始表中；由于部分年份 PDF 表格不可稳定抽取，默认训练暂不使用该字段
- 外生变量：北方代表城市气象数据与百度指数聚合序列

## 当前状态

- 默认启用“严格真实数据模式”，不再回退到模拟数据或占位常数。
- 训练前必须提供 `data/raw/source_manifest.json`，登记三类数据的来源、区域、粒度与文件路径。
- 已新增国家流感中心周报采集脚本：`scripts/fetch_flu_cn_weekly.py`。
- 旧版平谷 `synthetic_extension` 数据仅保留为工程历史，不进入最终论文实验结论。
- 训练过程会输出：
  - `results/reports/data_quality_report.json`
  - `results/reports/horizon_metrics.json`
  - `results/reports/experiment_summary.json`
  - `results/reports/experiment_brief.md`
  - `results/reports/ablation_summary.json`
  - `results/reports/ablation_report.md`
  - `results/reports/final_data_audit.md`
  - `results/reports/test_predictions.csv`

## 真实数据准备

### 1. 流感数据

- 文件路径：`data/raw/flu/cnic_north_weekly_flu.csv`
- 必要字段：`date, year, week, ili_rate`
- 推荐生成方式：

```bash
python scripts/fetch_flu_cn_weekly.py --start-year 2010 --end-year 2026
```

脚本会从国家流感中心新旧周报列表抓取官方 HTML，并在近年页面缺少完整表格时解析 PDF 附件。解析摘要会保存到：

- `results/reports/cnic_weekly_parse_report.json`

### 2. 气象数据

- 文件路径：`data/raw/weather/north_representative_city_weather.csv`
- 必要字段：`date, temperature, humidity, wind_speed, pressure`
- 默认由 `Open-Meteo Archive API` 自动拉取北京、天津、济南、石家庄、沈阳、哈尔滨、西安、太原的日度气象数据，并按日期取均值。
- 训练前会自动按 ISO 周聚合为周度特征。

### 3. 搜索指数数据

- 文件路径：`data/raw/search/north_baidu_index.csv`
- 必要字段：`date, flu_search_index, cold_search_index, fever_search_index`
- 推荐口径：使用北方代表城市或省份的百度指数，按日均值或人口权重聚合。
- 百度指数需要登录态 Cookie，无法无凭证公开批量下载；抓取或人工导出后需统一为上述字段。

## 当前研究口径

- 研究对象是“北方省份流感活动周度趋势”，不是北京市或平谷区代理趋势。
- 目标序列来自中国国家流感中心《中国流感监测周报》，原始流感序列覆盖 `2010-2026`；考虑百度指数可用起点，默认多源建模区间为 `2011-2026`。
- `positive_rate` 作为原始留痕字段保存；若个别周报字段需要 PDF 解析或存在早期口径变化，应在 `cnic_weekly_parse_report.json` 和论文数据说明中留痕。
- 不再将 `synthetic_extension` 测试结果作为最终论文结论。

## 运行方式

```bash
pip install -r requirements.txt
python scripts/fetch_flu_cn_weekly.py --start-year 2010 --end-year 2026
python scripts/train.py
streamlit run web/app.py
```

如果只想使用已经生成的 `data/processed/merged_dataset.csv`：

```bash
python scripts/train.py --skip-collect
```

运行 iTransformer 多源特征消融实验：

```bash
python scripts/run_ablation.py --skip-collect
```

## 当前收尾检查

1. `scripts/fetch_flu_cn_weekly.py` 已生成北方省份周度流感序列；`scripts/patch_flu_missing_values.py` 已对剩余缺失做可追溯补齐，需在论文中说明 `cnic_weekly_parse_report.json` 与 `final_data_audit.md` 记录的 `imputed` 来源。
2. `north_baidu_index.csv` 已切换为北方代表城市聚合序列；若更新城市范围或聚合方式，应同步更新 `source_manifest.json` 与论文数据来源表。
3. 每次更新原始数据后，重新运行 `python scripts/train.py --skip-collect` 和 `python scripts/run_ablation.py --skip-collect`，并用新产物覆盖论文图表与指标表。
4. 运行 `python scripts/audit_final_data.py` 生成最终数据复核说明，用于答辩和论文数据质量章节。
5. 最终论文与 Web 展示均应使用“国家流感中心北方省份周度流感监测数据”口径，不再使用旧平谷代理与 synthetic_extension 结论。

## 数据补齐与最终状态

当前数据补齐与边界已汇总在 `results/reports/final_data_audit.md`，核心情况如下：

1. 原始国家流感中心周报共 `849` 行，时间范围为 `2010-01-04` 至 `2026-04-13`；解析状态为 `ok=739`、`imputed=110`。
2. `ili_rate` 剩余缺失为 `0` 周；原先缺失的 `3` 周已按国家流感中心后续/相邻周报中的明确数值补齐：
   - `2017-10-09`，2017 年第 41 周，补齐为 `2.4%`，来源为 2020 年第 41 周周报的 2017 同期北方 ILI%。
   - `2017-10-16`，2017 年第 42 周，补齐为 `2.6%`，来源为 2017 年第 43 周周报的“前一周”北方 ILI%。
   - `2021-12-13`，2021 年第 50 周，补齐为 `3.5%`，来源为 2022/2023 年第 50 周周报的 2021 同期北方 ILI%。
3. `positive_rate` 剩余缺失为 `0` 周；原先缺失的 `108` 周已用相邻周线性插值补齐，仅作为原始监测留痕字段，默认不进入训练特征。
4. 百度指数 `flu_search_index`、`cold_search_index`、`fever_search_index` 三个关键词每日均由 `8` 个北方代表地区聚合，当前不存在地区数量缺口；若后续改为人口权重聚合，需要重新生成搜索数据、训练结果和论文图表。
5. 当前周度合并数据为 `797` 行；特征工程删除滞后/滚动窗口产生的前置缺失后，进入训练切分的数据为 `793` 行。

补齐脚本会保留原始备份 `data/raw/flu/cnic_north_weekly_flu.before_imputation.csv`，并在 CSV 的 `imputation_method`、`imputation_source`、`imputation_note` 字段中记录补齐依据。

## 目录结构

```text
flu_prediction/
├── config/config.yaml
├── data/
│   ├── raw/
│   │   ├── source_manifest.example.json
│   │   ├── flu/cnic_north_weekly_flu.csv
│   │   ├── weather/north_representative_city_weather.csv
│   │   └── search/north_baidu_index.csv
│   ├── processed/
│   └── splits/
├── results/
│   ├── figures/
│   ├── logs/
│   └── reports/
├── scripts/
├── src/
└── web/
```
