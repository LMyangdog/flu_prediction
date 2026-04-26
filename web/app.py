"""
流感爆发趋势预测系统 - Streamlit 答辩展示仪表板

启动命令:
    streamlit run web/app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"
DATA_DIR = PROJECT_ROOT / "data"

TARGET_LABEL = "北方省份哨点医院 ILI%"
MODEL_ORDER = ["iTransformer", "ARIMA", "DLinear", "LSTM"]
MODEL_COLORS = {
    "iTransformer": "#2563EB",
    "ARIMA": "#16A34A",
    "DLinear": "#9333EA",
    "LSTM": "#EA580C",
    "Actual": "#111827",
}


st.set_page_config(
    page_title="流感趋势预测答辩展示",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
    :root {
        --ink: #172033;
        --muted: #667085;
        --line: #E4E7EC;
        --panel: #FFFFFF;
        --soft: #F6F8FB;
        --blue: #2563EB;
        --green: #16A34A;
        --amber: #D97706;
        --red: #DC2626;
    }

    .stApp {
        background: #F7F8FA;
        color: var(--ink);
    }

    .main .block-container {
        max-width: 1360px;
        padding: 1.1rem 1.8rem 2.6rem;
    }

    header,
    header[data-testid="stHeader"],
    footer,
    #MainMenu,
    .stDeployButton,
    .stAppDeployButton,
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }

    h1, h2, h3 {
        letter-spacing: 0;
    }

    .top-band {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-left: 6px solid var(--blue);
        border-radius: 8px;
        padding: 1.15rem 1.25rem;
        margin-bottom: 1rem;
    }

    .eyebrow {
        color: var(--blue);
        font-size: 0.84rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .page-title {
        color: var(--ink);
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 0.35rem 0;
    }

    .page-subtitle {
        color: var(--muted);
        font-size: 0.98rem;
        line-height: 1.62;
        margin: 0;
    }

    .section-title {
        color: var(--ink);
        font-size: 1.18rem;
        font-weight: 760;
        margin: 1.15rem 0 0.65rem;
    }

    .kpi-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.95rem 1rem;
        min-height: 118px;
    }

    .kpi-label {
        color: var(--muted);
        font-size: 0.82rem;
        font-weight: 650;
        margin-bottom: 0.42rem;
    }

    .kpi-value {
        color: var(--ink);
        font-size: 1.42rem;
        font-weight: 820;
        line-height: 1.22;
    }

    .kpi-note {
        color: var(--muted);
        font-size: 0.79rem;
        margin-top: 0.45rem;
        line-height: 1.38;
    }

    .story-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        min-height: 132px;
    }

    .story-index {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.65rem;
        height: 1.65rem;
        border-radius: 50%;
        background: #EFF6FF;
        color: var(--blue);
        font-weight: 800;
        margin-bottom: 0.55rem;
    }

    .story-title {
        color: var(--ink);
        font-weight: 780;
        margin-bottom: 0.32rem;
    }

    .story-text {
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.48;
    }

    .note-box {
        background: #F8FAFC;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.82rem 0.95rem;
        color: #475467;
        font-size: 0.9rem;
        line-height: 1.55;
    }

    .verdict {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem 1.05rem;
    }

    .verdict strong {
        color: var(--ink);
    }

    .status-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.16rem 0.58rem;
        font-size: 0.78rem;
        font-weight: 750;
        margin-bottom: 0.35rem;
    }

    .pill-blue { background: #EFF6FF; color: #1D4ED8; }
    .pill-green { background: #ECFDF3; color: #15803D; }
    .pill-amber { background: #FFF7ED; color: #B45309; }
    .pill-red { background: #FEF2F2; color: #B91C1C; }

    .small-table div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 8px;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.85rem 0.9rem;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.2rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.55rem 0.95rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_data() -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    paths = {
        "merged": DATA_DIR / "processed" / "merged_dataset.csv",
        "featured": DATA_DIR / "processed" / "featured_dataset.csv",
    }
    for name, path in paths.items():
        if path.exists():
            df = pd.read_csv(path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            data[name] = df
    return data


@st.cache_data
def load_metrics() -> dict[str, dict[str, float]]:
    summary = read_json(REPORTS_DIR / "experiment_summary.json", {})
    if summary.get("metrics"):
        return summary["metrics"]
    return read_json(FIGURES_DIR / "all_metrics.json", {})


@st.cache_data
def load_experiment_summary() -> dict[str, Any]:
    return read_json(REPORTS_DIR / "experiment_summary.json", {})


@st.cache_data
def load_horizon_metrics() -> dict[str, dict[str, float]]:
    summary = load_experiment_summary()
    if summary.get("horizon_metrics"):
        return summary["horizon_metrics"]
    return read_json(REPORTS_DIR / "horizon_metrics.json", {})


@st.cache_data
def load_audit() -> dict[str, Any]:
    return read_json(REPORTS_DIR / "final_data_audit.json", {})


@st.cache_data
def load_ablation() -> pd.DataFrame:
    path = REPORTS_DIR / "ablation_metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_predictions() -> pd.DataFrame:
    path = REPORTS_DIR / "test_predictions.csv"
    if path.exists():
        df = pd.read_csv(path)
        for col in ["anchor_date", "target_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        return df
    return pd.DataFrame()


def fmt_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "暂无"
    try:
        if pd.isna(value):
            return "暂无"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_date(value: Any, fmt: str = "%Y-%m-%d") -> str:
    if value is None or pd.isna(value):
        return "暂无"
    return pd.to_datetime(value).strftime(fmt)


def metric_value(metrics: dict[str, dict[str, float]], model: str, key: str) -> float | None:
    value = metrics.get(model, {}).get(key)
    if value is None and key == "R2":
        value = metrics.get(model, {}).get("R²")
    return value


def best_model(metrics: dict[str, dict[str, float]], key: str, higher_is_better: bool) -> tuple[str, float] | None:
    rows = []
    for model, vals in metrics.items():
        value = vals.get(key, vals.get("R²") if key == "R2" else None)
        if value is not None and not pd.isna(value):
            rows.append((model, float(value)))
    if not rows:
        return None
    return sorted(rows, key=lambda item: item[1], reverse=higher_is_better)[0]


def kpi_card(label: str, value: str, note: str, color: str = "#2563EB") -> None:
    st.markdown(
        f"""
        <div class="kpi-card" style="border-top: 4px solid {color};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def story_card(index: int, title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="story-card">
            <div class="story-index">{index}</div>
            <div class="story-title">{title}</div>
            <div class="story-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_title() -> None:
    st.markdown(
        """
        <div class="top-band">
            <div class="eyebrow">毕业设计答辩展示 · 多源时序预测</div>
            <div class="page-title">基于深度学习和多元数据的流感爆发趋势预测</div>
            <p class="page-subtitle">
                研究对象为中国国家流感中心北方省份周度 ILI% 监测序列，融合气象与搜索指数，
                对未来 4 周流感活动趋势进行预测，并用 ARIMA、DLinear、LSTM 等基线模型进行验证。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    summary = load_experiment_summary()
    audit = load_audit()

    with st.sidebar:
        st.markdown("### 展示导航")
        page = st.radio(
            "选择内容",
            ["答辩总览", "数据与特征", "模型结果", "预警复盘", "附录信息"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### 研究口径")
        st.caption("中国北方省份真实周度流感监测序列")
        st.markdown("**目标变量**：`ili_rate`")
        st.markdown("**预测任务**：输入 16 周，预测未来 4 周")

        if summary.get("split_metadata"):
            meta = summary["split_metadata"]
            st.markdown("**训练/验证/测试**")
            st.caption(f"{meta.get('train_rows', 'n/a')} / {meta.get('val_rows', 'n/a')} / {meta.get('test_rows', 'n/a')} 条窗口样本")

        if audit:
            st.markdown("**数据覆盖**")
            st.caption(f"{audit.get('feature_engineered_date_start', 'n/a')} 至 {audit.get('feature_engineered_date_end', 'n/a')}")

        st.markdown("---")
        st.markdown("### 答辩讲法")
        st.caption("先讲数据口径，再讲方法设计，随后展示模型优于基线，最后说明预警流程与局限。")

    return page


def get_target_df() -> tuple[pd.DataFrame | None, str]:
    data = load_data()
    if "featured" in data:
        df = data["featured"].copy()
    elif "merged" in data:
        df = data["merged"].copy()
    else:
        return None, "ili_rate"
    target_col = "ili_rate" if "ili_rate" in df.columns else df.select_dtypes(include=[np.number]).columns[0]
    return df, target_col


def make_signal_chart(df: pd.DataFrame, target_col: str) -> go.Figure:
    plot_df = df[["date", target_col]].dropna().copy()
    plot_df["rolling_8w"] = plot_df[target_col].rolling(8, min_periods=2).mean()
    high_threshold = plot_df[target_col].quantile(0.85)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["date"],
            y=plot_df[target_col],
            name="周度 ILI%",
            mode="lines",
            line=dict(color="#111827", width=1.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["date"],
            y=plot_df["rolling_8w"],
            name="8周滑动均值",
            mode="lines",
            line=dict(color="#2563EB", width=2.6),
        )
    )
    fig.add_hline(
        y=high_threshold,
        line_dash="dash",
        line_color="#DC2626",
        annotation_text="历史85分位",
        annotation_position="top left",
    )
    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=45, b=10),
        title="真实监测趋势：季节性波动与高位周可直接辨认",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title=TARGET_LABEL,
        xaxis_title="",
    )
    return fig


def make_prediction_chart(preds: pd.DataFrame, horizon: int = 1, models: list[str] | None = None) -> go.Figure:
    models = models or ["iTransformer", "ARIMA", "DLinear", "LSTM"]
    if preds.empty:
        return go.Figure()

    plot_df = preds[preds["horizon"] == horizon].sort_values("target_date").copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["target_date"],
            y=plot_df["actual"],
            name="真实值",
            line=dict(color=MODEL_COLORS["Actual"], width=2.8),
        )
    )
    for model in models:
        if model in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df["target_date"],
                    y=plot_df[model],
                    name=model,
                    line=dict(color=MODEL_COLORS.get(model, "#64748B"), width=2),
                )
            )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=45, b=10),
        title=f"测试集预测对比：提前 {horizon} 周预测",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title=TARGET_LABEL,
        xaxis_title="",
    )
    return fig


def metrics_frame(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame()
    df = pd.DataFrame(metrics).T
    keep = [c for c in ["RMSE", "MAE", "MAPE", "R2", "peak_hit_rate", "peak_value_error"] if c in df.columns]
    df = df[keep].copy()
    df.index.name = "模型"
    return df


def make_metric_bar(metrics: dict[str, dict[str, float]], metric: str) -> go.Figure:
    rows = []
    for model in MODEL_ORDER:
        value = metric_value(metrics, model, metric)
        if value is not None:
            rows.append((model, float(value)))
    if not rows:
        return go.Figure()
    if metric in {"RMSE", "MAE", "MAPE", "peak_value_error"}:
        rows.sort(key=lambda item: item[1])
    else:
        rows.sort(key=lambda item: item[1], reverse=True)

    fig = go.Figure(
        data=[
            go.Bar(
                x=[r[0] for r in rows],
                y=[r[1] for r in rows],
                marker_color=[MODEL_COLORS.get(r[0], "#64748B") for r in rows],
                text=[fmt_num(r[1], 3) for r in rows],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        height=330,
        template="plotly_white",
        margin=dict(l=10, r=10, t=35, b=10),
        title=f"{metric} 指标对比",
        yaxis_title=metric,
        showlegend=False,
    )
    return fig


def make_horizon_chart(horizon: dict[str, dict[str, float]], metric: str = "RMSE") -> go.Figure:
    fig = go.Figure()
    for model in MODEL_ORDER:
        values = []
        for h in range(1, 5):
            values.append(horizon.get(model, {}).get(f"H{h}_{metric}"))
        if any(v is not None for v in values):
            fig.add_trace(
                go.Scatter(
                    x=["H1", "H2", "H3", "H4"],
                    y=values,
                    name=model,
                    mode="lines+markers",
                    line=dict(color=MODEL_COLORS.get(model, "#64748B"), width=2.4),
                )
            )
    fig.update_layout(
        height=350,
        template="plotly_white",
        margin=dict(l=10, r=10, t=35, b=10),
        title=f"预测步长稳定性：{metric}",
        yaxis_title=metric,
        xaxis_title="预测步长",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def make_ablation_chart(ablation: pd.DataFrame) -> go.Figure:
    if ablation.empty:
        return go.Figure()
    plot_df = ablation.sort_values("RMSE").copy()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=plot_df["label"],
            y=plot_df["RMSE"],
            name="RMSE",
            marker_color="#2563EB",
            text=[fmt_num(v, 3) for v in plot_df["RMSE"]],
            textposition="outside",
        ),
        secondary_y=False,
    )
    if "peak_value_error" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["label"],
                y=plot_df["peak_value_error"],
                name="峰值幅度误差",
                mode="lines+markers",
                line=dict(color="#DC2626", width=2.6),
            ),
            secondary_y=True,
        )
    fig.update_layout(
        height=360,
        template="plotly_white",
        margin=dict(l=10, r=10, t=35, b=10),
        title="消融实验：不同数据源组合的贡献",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="RMSE", secondary_y=False)
    fig.update_yaxes(title_text="峰值幅度误差", secondary_y=True)
    return fig


def render_overview() -> None:
    df, target_col = get_target_df()
    metrics = load_metrics()
    summary = load_experiment_summary()
    audit = load_audit()
    preds = load_predictions()
    ablation = load_ablation()

    if df is None:
        st.warning("尚未找到处理后的数据，请先运行 `python scripts/train.py --skip-collect`。")
        return

    best_rmse = best_model(metrics, "RMSE", higher_is_better=False)
    best_r2 = best_model(metrics, "R2", higher_is_better=True)
    meta = summary.get("split_metadata", {})
    latest = df.dropna(subset=[target_col]).sort_values("date").iloc[-1]
    recent = df.dropna(subset=[target_col]).tail(4)[target_col].mean()
    previous = df.dropna(subset=[target_col]).tail(8).head(4)[target_col].mean()
    delta = recent - previous if not pd.isna(previous) else np.nan

    cols = st.columns(4)
    with cols[0]:
        kpi_card(
            "数据跨度",
            f"{fmt_date(df['date'].min(), '%Y-%m')} 至 {fmt_date(df['date'].max(), '%Y-%m')}",
            f"合并周度样本 {len(df)} 条，特征工程后 {audit.get('feature_engineered_rows', meta.get('rows', 'n/a'))} 条。",
            "#2563EB",
        )
    with cols[1]:
        kpi_card(
            "预测任务",
            "16周输入 -> 4周输出",
            "面向答辩展示的短期趋势预测，既看误差，也看峰值识别。",
            "#16A34A",
        )
    with cols[2]:
        if best_rmse:
            kpi_card(
                "综合误差最优",
                f"{best_rmse[0]}",
                f"RMSE={fmt_num(best_rmse[1], 3)}；R2最优为 {best_r2[0] if best_r2 else '暂无'}。",
                "#9333EA",
            )
        else:
            kpi_card("综合误差最优", "暂无结果", "训练完成后自动读取评估指标。", "#9333EA")
    with cols[3]:
        kpi_card(
            "最新监测值",
            f"{fmt_num(latest[target_col], 2)}%",
            f"{fmt_date(latest['date'])}；近4周较前4周 {'+' if delta >= 0 else ''}{fmt_num(delta, 2)}。",
            "#D97706",
        )

    st.markdown('<div class="section-title">答辩主线</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        story_card(1, "真实数据口径", "统一采用国家流感中心北方省份周度 ILI%，研究对象清晰可追溯。")
    with s2:
        story_card(2, "多源特征融合", "历史流感、气象、搜索指数与季节性特征共同进入模型。")
    with s3:
        story_card(3, "iTransformer 建模", "把变量当作 token，学习变量间关联，输出未来 4 周趋势。")
    with s4:
        story_card(4, "基线与消融验证", "与 ARIMA、DLinear、LSTM 对比，并解释数据源组合的边界。")

    st.markdown('<div class="section-title">核心图表</div>', unsafe_allow_html=True)
    left, right = st.columns([1.55, 1])
    with left:
        st.plotly_chart(make_signal_chart(df, target_col), use_container_width=True)
    with right:
        rmse_text = "暂无"
        r2_text = "暂无"
        if "iTransformer" in metrics:
            rmse_text = fmt_num(metric_value(metrics, "iTransformer", "RMSE"), 3)
            r2_text = fmt_num(metric_value(metrics, "iTransformer", "R2"), 3)
        st.markdown(
            f"""
            <div class="verdict">
                <span class="status-pill pill-blue">一句话结论</span>
                <p><strong>本系统能基于真实周度 ILI% 数据完成未来 4 周流感活动趋势预测。</strong></p>
                <p>当前实验中，iTransformer 在多模型综合误差上表现较优：RMSE={rmse_text}，R2={r2_text}。</p>
                <p>消融结果显示，多源融合对峰值幅度误差更友好；仅历史流感序列在 RMSE 上也很强，说明目标序列自相关是主要信息来源。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if audit:
            st.markdown("")
            st.markdown(
                f"""
                <div class="note-box">
                    <strong>数据质量说明：</strong> 原始周报 {audit.get('flu_rows', 'n/a')} 行，
                    ILI% 缺失 {audit.get('missing_ili_rate_rows', 'n/a')} 周；
                    搜索指数覆盖 {audit.get('search_region_count_summary', {}).get('flu_search_index_region_count', {}).get('min', 'n/a')}
                    个北方代表地区。该说明适合答辩时主动回应数据可信度问题。
                </div>
                """,
                unsafe_allow_html=True,
            )

    if not preds.empty:
        st.plotly_chart(make_prediction_chart(preds, horizon=1, models=["iTransformer", "ARIMA", "DLinear"]), use_container_width=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(make_metric_bar(metrics, "RMSE"), use_container_width=True)
    with c2:
        if not ablation.empty:
            st.plotly_chart(make_ablation_chart(ablation), use_container_width=True)
        else:
            st.info("暂无消融实验结果。")


def render_data_story() -> None:
    df, target_col = get_target_df()
    audit = load_audit()

    if df is None:
        st.warning("尚未找到处理后的数据。")
        return

    st.markdown('<div class="section-title">数据来源与处理链路</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    with cols[0]:
        kpi_card("流感主序列", "国家流感中心周报", "北方省份周度 ILI%，作为唯一预测目标。", "#2563EB")
    with cols[1]:
        kpi_card("气象外生变量", "8个代表城市", "温度、湿度、风速、气压按周聚合。", "#16A34A")
    with cols[2]:
        kpi_card("搜索行为变量", "3个关键词", "流感、感冒、发烧搜索指数聚合。", "#D97706")

    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.plotly_chart(make_signal_chart(df, target_col), use_container_width=True)
    with c2:
        st.markdown('<div class="section-title">数据审计</div>', unsafe_allow_html=True)
        if audit:
            audit_rows = pd.DataFrame(
                [
                    ("原始流感周报行数", audit.get("flu_rows")),
                    ("周报解析完整/部分", f"{audit.get('parse_status_counts', {}).get('ok', 'n/a')} / {audit.get('parse_status_counts', {}).get('partial', 'n/a')}"),
                    ("ILI% 缺失周数", audit.get("missing_ili_rate_rows")),
                    ("周度合并样本", audit.get("merged_rows")),
                    ("特征工程样本", audit.get("feature_engineered_rows")),
                    ("气象日度记录", audit.get("weather_rows")),
                    ("搜索指数日度记录", audit.get("search_rows")),
                ],
                columns=["项目", "数值"],
            )
            audit_rows["数值"] = audit_rows["数值"].astype(str)
            st.dataframe(audit_rows, hide_index=True, use_container_width=True)
        else:
            st.info("未找到数据审计报告。")

    st.markdown('<div class="section-title">特征相关性</div>', unsafe_allow_html=True)
    numeric = df.select_dtypes(include=[np.number]).copy()
    prefer = [
        target_col,
        "temperature",
        "humidity",
        "wind_speed",
        "pressure",
        "flu_search_index",
        "cold_search_index",
        "fever_search_index",
        "is_flu_season",
    ]
    cols = [c for c in prefer if c in numeric.columns]
    if len(cols) >= 2:
        corr = numeric[cols].corr()
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
                hovertemplate="%{y} vs %{x}<br>相关系数=%{z:.2f}<extra></extra>",
            )
        )
        fig.update_layout(height=470, template="plotly_white", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("查看处理后数据前 30 行"):
        st.dataframe(df.head(30), use_container_width=True, hide_index=True)


def render_model_results() -> None:
    metrics = load_metrics()
    horizon = load_horizon_metrics()
    preds = load_predictions()
    ablation = load_ablation()

    if not metrics:
        st.warning("尚未找到模型评估结果，请先运行训练脚本。")
        return

    st.markdown('<div class="section-title">模型指标总览</div>', unsafe_allow_html=True)
    mdf = metrics_frame(metrics)
    if not mdf.empty:
        st.dataframe(
            mdf.style.format("{:.4f}"),
            use_container_width=True,
        )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(make_metric_bar(metrics, "RMSE"), use_container_width=True)
    with right:
        st.plotly_chart(make_metric_bar(metrics, "R2"), use_container_width=True)

    st.markdown('<div class="section-title">预测步长表现</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(make_horizon_chart(horizon, "RMSE"), use_container_width=True)
    with c2:
        st.plotly_chart(make_horizon_chart(horizon, "MAPE"), use_container_width=True)

    st.markdown('<div class="section-title">测试集预测曲线</div>', unsafe_allow_html=True)
    if not preds.empty:
        horizon_choice = st.radio(
            "预测提前周数",
            options=[1, 2, 3, 4],
            format_func=lambda x: f"提前{x}周",
            horizontal=True,
        )
        st.plotly_chart(make_prediction_chart(preds, int(horizon_choice)), use_container_width=True)
    else:
        st.info("未找到测试集预测明细。")

    st.markdown('<div class="section-title">消融实验解释</div>', unsafe_allow_html=True)
    if not ablation.empty:
        c3, c4 = st.columns([1.2, 1])
        with c3:
            st.plotly_chart(make_ablation_chart(ablation), use_container_width=True)
        with c4:
            show_cols = [c for c in ["label", "feature_count", "RMSE", "MAE", "MAPE", "R2", "peak_value_error"] if c in ablation.columns]
            st.dataframe(
                ablation[show_cols].sort_values("RMSE").style.format(
                    {"RMSE": "{:.4f}", "MAE": "{:.4f}", "MAPE": "{:.2f}", "R2": "{:.4f}", "peak_value_error": "{:.4f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )
        st.markdown(
            """
            <div class="note-box">
                答辩时建议如实说明：历史流感序列本身解释力很强；引入气象与搜索指数后，
                对峰值幅度识别更有价值，但在当前样本规模下未必让所有误差指标同时最优。
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("未找到消融实验结果。")


def risk_thresholds(df: pd.DataFrame, target_col: str) -> tuple[float, float]:
    values = df[target_col].dropna()
    if values.empty:
        return 0.0, 0.0
    return float(values.quantile(0.70)), float(values.quantile(0.85))


def render_warning_review() -> None:
    df, target_col = get_target_df()
    preds = load_predictions()

    if df is None or preds.empty:
        st.warning("需要处理后数据和 `test_predictions.csv` 才能展示预警复盘。")
        return

    anchor_dates = sorted(preds["anchor_date"].dropna().unique())
    selected_anchor = st.select_slider(
        "选择一个历史时点，模拟当周做出的未来4周判断",
        options=[pd.Timestamp(d).strftime("%Y-%m-%d") for d in anchor_dates],
        value=pd.Timestamp(anchor_dates[-1]).strftime("%Y-%m-%d"),
    )
    anchor = pd.to_datetime(selected_anchor)
    review = preds[preds["anchor_date"] == anchor].sort_values("horizon").copy()

    watch, high = risk_thresholds(df, target_col)
    forecast_max = float(review["iTransformer"].max()) if "iTransformer" in review else float(review["actual"].max())
    actual_max = float(review["actual"].max())

    if forecast_max >= high:
        label = "高位预警"
        pill = "pill-red"
        advice = "预测值进入历史高位区间，答辩演示中可说明系统会提示加强监测与资源预案。"
    elif forecast_max >= watch:
        label = "关注上升"
        pill = "pill-amber"
        advice = "预测值处于历史偏高区间，适合展示提前关注和连续复核流程。"
    else:
        label = "总体平稳"
        pill = "pill-green"
        advice = "预测值未进入高位区间，适合说明系统并非只会报高风险，也能给出低风险判断。"

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("模拟预警时点", fmt_date(anchor), "仅使用该时点之前的历史窗口进行演示。", "#2563EB")
    with c2:
        kpi_card("未来4周预测峰值", f"{fmt_num(forecast_max, 2)}%", f"历史70/85分位阈值：{fmt_num(watch, 2)} / {fmt_num(high, 2)}。", "#D97706")
    with c3:
        kpi_card("事后真实峰值", f"{fmt_num(actual_max, 2)}%", "用于答辩现场说明回测验证方式。", "#16A34A")

    st.markdown(
        f"""
        <div class="verdict">
            <span class="status-pill {pill}">{label}</span>
            <p><strong>{advice}</strong></p>
            <p>阈值来自当前真实监测序列的历史分位数，避免使用病例量级阈值解释 ILI%。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history = df[(df["date"] <= anchor)].dropna(subset=[target_col]).tail(16)
    if history.empty:
        st.warning("该预警时点之前没有足够的历史观测值。")
        return

    start_date = history["date"].iloc[-1]
    start_value = float(history[target_col].iloc[-1])
    future_dates = [start_date] + review["target_date"].tolist()
    predicted_values = [start_value] + review["iTransformer"].tolist()
    actual_values = [start_value] + review["actual"].tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history[target_col],
            name="输入历史窗口",
            mode="lines+markers",
            line=dict(color="#111827", width=2.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=predicted_values,
            name="iTransformer 预测",
            mode="lines+markers",
            line=dict(color="#2563EB", width=2.8, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=actual_values,
            name="事后真实值",
            mode="lines+markers",
            line=dict(color="#16A34A", width=2.5),
        )
    )
    fig.add_vline(x=anchor, line_color="#667085", line_dash="dash")
    fig.add_hline(y=watch, line_color="#D97706", line_dash="dot", annotation_text="70分位")
    fig.add_hline(y=high, line_color="#DC2626", line_dash="dot", annotation_text="85分位")
    fig.update_layout(
        height=460,
        template="plotly_white",
        title="历史回测式预警复盘：预测曲线与事后真实值对照",
        margin=dict(l=10, r=10, t=45, b=10),
        yaxis_title=TARGET_LABEL,
        xaxis_title="",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("查看该时点未来4周明细"):
        show_cols = [c for c in ["horizon", "anchor_date", "target_date", "actual", "iTransformer", "ARIMA", "DLinear", "LSTM"] if c in review.columns]
        st.dataframe(review[show_cols], use_container_width=True, hide_index=True)


def render_appendix() -> None:
    summary = load_experiment_summary()
    audit = load_audit()

    st.markdown('<div class="section-title">实验配置摘要</div>', unsafe_allow_html=True)
    meta = summary.get("split_metadata", {})
    rows = pd.DataFrame(
        [
            ("随机种子", summary.get("random_seed", "n/a")),
            ("目标变量", summary.get("target_col", "ili_rate")),
            ("特征数量", summary.get("num_variables", meta.get("feature_count", "n/a"))),
            ("输入窗口", meta.get("lookback_window", "n/a")),
            ("预测步长", meta.get("forecast_horizon", "n/a")),
            ("训练集时间", f"{meta.get('train_date_range', {}).get('start', 'n/a')} 至 {meta.get('train_date_range', {}).get('end', 'n/a')}"),
            ("验证集时间", f"{meta.get('val_date_range', {}).get('start', 'n/a')} 至 {meta.get('val_date_range', {}).get('end', 'n/a')}"),
            ("测试集时间", f"{meta.get('test_date_range', {}).get('start', 'n/a')} 至 {meta.get('test_date_range', {}).get('end', 'n/a')}"),
        ],
        columns=["项目", "内容"],
    )
    rows["内容"] = rows["内容"].astype(str)
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">答辩可引用说明</div>', unsafe_allow_html=True)
    note = summary.get(
        "interpretation_note",
        "当前研究口径为中国国家流感中心北方省份真实周度 ILI% 监测序列，不应外推为单一城市病例预测结论。",
    )
    st.markdown(f'<div class="note-box">{note}</div>', unsafe_allow_html=True)

    if audit:
        st.markdown('<div class="section-title">数据审计 JSON</div>', unsafe_allow_html=True)
        st.json(audit)

    st.markdown('<div class="section-title">已生成图表文件</div>', unsafe_allow_html=True)
    figure_files = sorted([p.name for p in FIGURES_DIR.glob("*.png")]) if FIGURES_DIR.exists() else []
    if figure_files:
        st.dataframe(pd.DataFrame({"文件名": figure_files}), use_container_width=True, hide_index=True)
    else:
        st.info("暂无图表文件。")


def main() -> None:
    render_title()
    page = render_sidebar()

    if page == "答辩总览":
        render_overview()
    elif page == "数据与特征":
        render_data_story()
    elif page == "模型结果":
        render_model_results()
    elif page == "预警复盘":
        render_warning_review()
    elif page == "附录信息":
        render_appendix()


if __name__ == "__main__":
    main()
