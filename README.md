# 基于深度学习和多元数据的流感爆发趋势预测

> 中国海洋大学 · 信息科学与工程学部 · 计算机科学与技术 2022 级 本科毕业设计

## 📖 项目概述

本项目基于 Python 构建了一个融合多源异构数据的流感爆发趋势预测系统，采用前沿的 **iTransformer** 深度学习架构，通过创新的通道独立（Channel Independence）机制，高效融合流感监测数据、气象数据和网络搜索关键词数据，实现流感趋势的高精度预测与可视化。

### 核心创新点

- **iTransformer 倒置自注意力**：将每个变量的完整时间序列作为独立 Token，在变量维度上计算注意力，天然适配多源异构数据融合
- **多源数据融合**：整合流感监测（ILI%）、气象（温湿度）、搜索指数三类数据
- **全链路系统**：从数据采集 → 特征工程 → 模型训练 → 评估对比 → Web 可视化的完整流水线

## 🏗️ 技术架构

```
数据采集层          预处理层          模型层           评估层          展示层
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 流感中心  │    │ 缺失值填充│    │iTransformer│    │ RMSE/MAE │    │ Streamlit│
│ 气象 API  │──→│ 异常值处理│──→│  LSTM     │──→│  R²/MAPE │──→│ Web 仪表板│
│ 搜索指数  │    │ 归一化   │    │  DLinear  │    │ 峰值准确率│    │ 可视化图表│
│           │    │ 滑动窗口 │    │  ARIMA   │    │          │    │          │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境（推荐）
conda create -n flu python=3.10
conda activate flu

# 安装 PyTorch（CUDA 版本）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
pip install -r requirements.txt
```

### 2. 训练模型

本项目实现了**端到端的自动化全流程架构**。针对您的不同需求，提供两种训练方式：

**方法一：全自动采集并训练（默认方式）**
程序会自动从气象局、流感中心采集最新到今天的前瞻性数据，清洗并进行特征工程后开始深度学习。保证模型使用的是最新鲜的数据。
```bash
# 完整跑通一次最新数据（约需十几分钟）
python scripts/train.py
```

**方法二：纯脱机本地训练（调试模型用，带 --skip-collect 参数）**
如果您只想验证某几个超参数或是需要断网答辩，且本地已经有过 `data/processed/merged_dataset.csv` 时，加上跳过参数即可 1 秒钟跳过爬虫阶段直接炼丹：
```bash
# 纯本地断网高速训练
python scripts/train.py --skip-collect

# 如果你只想调参跑 5 个 Epoch 赶快看个响
python scripts/train.py --debug --skip-collect

# 如果你只想单独训练主角模型 iTransformer
python scripts/train.py --model iTransformer --skip-collect
```

### 3. 启动 Web 仪表板

```bash
streamlit run web/app.py
```

## 📁 项目结构

```
flu_prediction/
├── config/config.yaml          # 全局配置（超参数、路径等）
├── data/                       # 数据目录
│   ├── raw/                    # 原始数据（流感/气象/搜索）
│   ├── processed/              # 预处理后数据
│   └── splits/                 # 训练/验证/测试集
├── src/                        # 核心源代码
│   ├── data/                   # 数据模块
│   │   ├── collector.py        # 多源数据采集器
│   │   ├── preprocessor.py     # 数据预处理
│   │   ├── feature_engineer.py # 特征工程
│   │   └── dataset.py          # PyTorch Dataset
│   ├── models/                 # 模型模块
│   │   ├── itransformer.py     # iTransformer（核心）
│   │   ├── lstm_baseline.py    # LSTM 基准
│   │   ├── dlinear_baseline.py # DLinear 基准
│   │   └── arima_baseline.py   # ARIMA 基准
│   ├── training/               # 训练模块
│   │   └── trainer.py          # 统一训练器
│   └── utils/                  # 工具模块
│       ├── metrics.py          # 评价指标
│       └── visualization.py    # Matplotlib 可视化
├── scripts/train.py            # 训练入口脚本
├── web/app.py                  # Streamlit Web 仪表板
├── checkpoints/                # 模型权重
├── results/                    # 实验结果
│   ├── figures/                # 可视化图表
│   └── logs/                   # 训练日志
└── requirements.txt            # 依赖清单
```

## 📊 模型对比

| 模型 | 特点 | 多源数据 | 非线性 |
|------|------|----------|--------|
| **iTransformer** | 通道独立 + 倒置自注意力 | ✅ | ✅ |
| LSTM | 时间步序列化处理 | ✅ | ✅ |
| DLinear | 趋势-残差分解 + 线性映射 | ✅ | ❌ |
| ARIMA | 经典统计方法 | ❌ | ❌ |

## 📝 评价指标

- **RMSE** — 均方根误差
- **MAE** — 平均绝对误差
- **MAPE** — 平均绝对百分比误差
- **R²** — 决定系数
- **峰值准确率** — 流感高峰期预测命中率

## 🛠️ 技术栈

- **深度学习**: PyTorch 2.0+
- **数据处理**: Pandas, NumPy, Scikit-learn
- **可视化**: Matplotlib, Seaborn, Plotly
- **Web**: Streamlit
- **统计模型**: Statsmodels
- **数据采集**: Requests, BeautifulSoup

启动命令：
C:/ProgramData/anaconda3/envs/ocean_torch/python.exe -m streamlit run d:/flu_prediction/web/app.py
