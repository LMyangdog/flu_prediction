import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const {
  Presentation,
  PresentationFile,
} = await import("@oai/artifact-tool");

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const workspace = path.resolve(__dirname, "..");
const assetDir = path.join(workspace, "scratch", "assets", "flu-defense");
const previewDir = path.join(workspace, "scratch", "previews");
const layoutDir = path.join(workspace, "scratch", "layouts");
const outputDir = path.join(workspace, "output");
fs.mkdirSync(previewDir, { recursive: true });
fs.mkdirSync(layoutDir, { recursive: true });
fs.mkdirSync(outputDir, { recursive: true });

const W = 1920;
const H = 1080;
const C = {
  canvas: "#F8FAFC",
  ink: "#111827",
  muted: "#5B6472",
  faint: "#E5E7EB",
  line: "#CBD5E1",
  teal: "#0E7C7B",
  teal2: "#0A5D60",
  blue: "#2563EB",
  sky: "#38A3E8",
  coral: "#E85D4A",
  amber: "#F5B841",
  green: "#2E9E62",
  purple: "#6B46C1",
  white: "#FFFFFF",
  black: "#0B1220",
};

const font = "Microsoft YaHei";
const mono = "Consolas";

const assets = {
  dataOverview: path.join(assetDir, "data_overview.png"),
  iliCover: path.join(assetDir, "ili_trend_cover.png"),
  pipeline: path.join(assetDir, "thesis_data_pipeline.png"),
  architecture: path.join(assetDir, "thesis_system_architecture.png"),
  pred: path.join(assetDir, "iTransformer_predictions.png"),
  attention: path.join(assetDir, "attention_heatmap_layer0.png"),
  corr: path.join(assetDir, "correlation_matrix.png"),
};

const metrics = [
  { model: "iTransformer", rmse: 0.772, mae: 0.491, mape: 11.51, r2: 0.538 },
  { model: "ARIMA", rmse: 0.980, mae: 0.552, mape: 11.87, r2: 0.450 },
  { model: "DLinear", rmse: 1.031, mae: 0.694, mape: 15.73, r2: 0.176 },
  { model: "LSTM", rmse: 1.366, mae: 0.880, mape: 18.81, r2: -0.444 },
];

const horizonRmse = [0.525, 0.676, 0.828, 0.983];
const horizonMape = [8.38, 11.27, 12.48, 13.89];

const ablation = [
  { label: "仅流感历史", rmse: 0.761, r2: 0.552, features: 13 },
  { label: "流感+气象", rmse: 0.767, r2: 0.544, features: 20 },
  { label: "三源融合", rmse: 0.770, r2: 0.540, features: 29 },
  { label: "流感+搜索", rmse: 0.788, r2: 0.519, features: 22 },
];

function solid(color) {
  return { type: "solid", color };
}

function none() {
  return { type: "none" };
}

function addShape(slide, { x, y, w, h, fill = none(), line = { style: "none" }, geometry = "rect", name }) {
  return slide.shapes.add({
    name,
    geometry,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line,
  });
}

function addText(slide, value, { x, y, w, h, size = 28, color = C.ink, bold = false, align = "left", valign = "top", name, family = font, italic = false, fill = none(), line = { style: "none" } }) {
  const shape = addShape(slide, { x, y, w, h, fill, line, name });
  shape.text.style = {
    fontFamily: family,
    fontSize: size,
    color,
    bold,
    italic,
    alignment: align,
    verticalAlignment: valign,
  };
  shape.text = value;
  return shape;
}

function addLine(slide, x1, y1, x2, y2, color = C.line, width = 2) {
  const left = Math.min(x1, x2);
  const top = Math.min(y1, y2);
  const w = Math.abs(x2 - x1) || 1;
  const h = Math.abs(y2 - y1) || 1;
  return slide.shapes.add({
    geometry: "line",
    position: { left, top, width: w, height: h },
    line: { style: "solid", fill: color, width },
  });
}

function addHeader(slide, title, subtitle, page) {
  addText(slide, title, { x: 94, y: 56, w: 1320, h: 76, size: 44, bold: true, color: C.ink, name: `slide-${page}-title` });
  if (subtitle) {
    addText(slide, subtitle, { x: 96, y: 128, w: 1320, h: 42, size: 20, color: C.muted, name: `slide-${page}-subtitle` });
  }
  addText(slide, String(page).padStart(2, "0"), { x: 1712, y: 64, w: 86, h: 38, size: 22, bold: true, color: C.teal, align: "right", name: `slide-${page}-page` });
  addLine(slide, 96, 178, 1824, 178, C.faint, 1.2);
}

function addFooter(slide, source = "资料来源：项目 README、毕业论文修订核验版、开题报告与 results/reports 实验报告。") {
  addLine(slide, 96, 1012, 1824, 1012, C.faint, 1);
  addText(slide, source, { x: 96, y: 1024, w: 1500, h: 28, size: 12, color: "#7B8493", name: "source-rail" });
  addText(slide, "中国海洋大学 · 计算机科学与技术本科毕业设计", { x: 1420, y: 1024, w: 404, h: 28, size: 12, color: "#7B8493", align: "right" });
}

function addPill(slide, label, x, y, w, color = C.teal, name) {
  addShape(slide, {
    x, y, w, h: 44,
    geometry: "roundRect",
    fill: solid(color),
    line: { style: "none" },
    name,
  });
  addText(slide, label, { x, y: y + 6, w, h: 34, size: 17, bold: true, color: C.white, align: "center", valign: "middle" });
}

function addBullet(slide, title, body, x, y, accent = C.teal, width = 640) {
  addShape(slide, { x, y: y + 10, w: 12, h: 56, fill: solid(accent), line: { style: "none" } });
  addText(slide, title, { x: x + 28, y, w: width, h: 34, size: 28, bold: true, color: C.ink });
  addText(slide, body, { x: x + 28, y: y + 44, w: width, h: 64, size: 20, color: C.muted });
}

function addMetric(slide, label, value, x, y, color = C.teal, suffix = "") {
  addText(slide, value, { x, y, w: 230, h: 74, size: 52, bold: true, color, family: mono });
  addText(slide, suffix, { x: x + 228, y: y + 28, w: 70, h: 30, size: 22, bold: true, color });
  addText(slide, label, { x, y: y + 78, w: 300, h: 34, size: 18, color: C.muted });
}

function addTable(slide, rows, columns, { x, y, w, rowH = 54, colWidths, headerFill = "#EAF7F6", accent = C.teal, size = 18 }) {
  const widths = colWidths || Array(columns).fill(w / columns);
  let cx = x;
  addShape(slide, { x, y, w, h: rowH, fill: solid(headerFill), line: { style: "solid", fill: C.line, width: 1 } });
  for (let i = 0; i < columns; i += 1) {
    addText(slide, rows[0][i], { x: cx + 14, y: y + 12, w: widths[i] - 24, h: rowH - 18, size: size + 1, bold: true, color: accent, valign: "middle" });
    if (i > 0) addLine(slide, cx, y, cx, y + rowH * rows.length, "#D7E2E8", 1);
    cx += widths[i];
  }
  for (let r = 1; r < rows.length; r += 1) {
    const ry = y + rowH * r;
    addShape(slide, { x, y: ry, w, h: rowH, fill: solid(r % 2 ? C.white : "#F3F7FA"), line: { style: "solid", fill: C.line, width: 0.8 } });
    cx = x;
    for (let c = 0; c < columns; c += 1) {
      addText(slide, rows[r][c], { x: cx + 14, y: ry + 11, w: widths[c] - 24, h: rowH - 18, size, color: c === 0 ? C.ink : C.muted, bold: c === 0, valign: "middle" });
      cx += widths[c];
    }
  }
}

function addBarChart(slide, { x, y, w, h, title, categories, values, color = C.teal, yTitle = "", dataLabel = true }) {
  const chart = slide.charts.add("bar", {
    categories,
    series: [{
      name: title,
      values,
      fill: solid(color),
      line: { fill: color, style: "solid", width: 1 },
    }],
    hasLegend: false,
    chartFill: solid(C.white),
    plotAreaFill: solid(C.white),
    chartLine: { fill: C.white, style: "solid", width: 0 },
    plotAreaLine: { fill: C.white, style: "solid", width: 0 },
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 85 },
    dataLabels: dataLabel ? { showValue: true, position: "outEnd", textStyle: { fontSize: 11, fill: C.ink } } : { showValue: false },
    xAxis: { textStyle: { fontSize: 10, fill: C.ink }, line: { fill: C.line, style: "solid", width: 1 }, tickLabelPosition: "nextTo" },
    yAxis: {
      title: yTitle ? { text: yTitle, textStyle: { fontSize: 11, fill: C.muted } } : undefined,
      textStyle: { fontSize: 10, fill: C.muted },
      numberFormatCode: "0.0",
      numberFormatSourceLinked: false,
      majorGridlines: { fill: "#E6EDF3", style: "solid", width: 1 },
      line: { fill: C.line, style: "solid", width: 1 },
    },
  });
  for (const series of chart.series.items) {
    series.valuesFormatCode = "0.000";
  }
  chart.frame = { left: x, top: y, width: w, height: h };
  return chart;
}

function addLineChart(slide, { x, y, w, h, title, categories, series }) {
  const chart = slide.charts.add("line", {
    categories,
    series: series.map((s) => ({
      name: s.name,
      values: s.values,
      line: { fill: s.color, style: "solid", width: 2.5 },
      fill: solid(s.color),
    })),
    hasLegend: true,
    legend: { position: "bottom", textStyle: { fontSize: 11 } },
    chartFill: solid(C.white),
    plotAreaFill: solid(C.white),
    chartLine: { fill: C.white, style: "solid", width: 0 },
    plotAreaLine: { fill: C.white, style: "solid", width: 0 },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fontSize: 10, fill: C.ink } },
    xAxis: { textStyle: { fontSize: 11, fill: C.ink }, line: { fill: C.line, style: "solid", width: 1 } },
    yAxis: {
      textStyle: { fontSize: 10, fill: C.muted },
      numberFormatCode: "0.0",
      numberFormatSourceLinked: false,
      majorGridlines: { fill: "#E6EDF3", style: "solid", width: 1 },
      line: { fill: C.line, style: "solid", width: 1 },
    },
  });
  for (const item of chart.series.items) {
    item.valuesFormatCode = "0.000";
  }
  chart.title = title;
  chart.titleTextStyle.fontSize = 13;
  chart.titleTextStyle.bold = true;
  chart.frame = { left: x, top: y, width: w, height: h };
  return chart;
}

function addImage(slide, pathValue, { x, y, w, h, fit = "contain", alt = "" }) {
  return slide.images.add({
    path: pathValue,
    alt,
    position: { left: x, top: y, width: w, height: h },
    fit,
  });
}

function addCanvas(slide) {
  addShape(slide, { x: 0, y: 0, w: W, h: H, fill: solid(C.canvas), line: { style: "none" }, name: "slide-background" });
}

const presentation = Presentation.create({ slideSize: { width: W, height: H } });

// Slide 1: cover.
{
  const slide = presentation.slides.add();
  addCanvas(slide);
  addImage(slide, assets.iliCover, { x: 0, y: 560, w: W, h: 322, fit: "cover", alt: "北方省份 ILI% 趋势曲线" });
  addShape(slide, { x: 0, y: 868, w: W, h: 212, fill: solid(C.canvas), line: { style: "none" } });
  addText(slide, "基于深度学习和多元数据的", { x: 120, y: 128, w: 1120, h: 70, size: 42, color: C.teal2, bold: true });
  addText(slide, "流感爆发趋势预测", { x: 116, y: 202, w: 1120, h: 110, size: 76, color: C.ink, bold: true });
  addLine(slide, 122, 340, 526, 340, C.coral, 6);
  addText(slide, "本科毕业设计答辩", { x: 122, y: 382, w: 520, h: 44, size: 28, color: C.muted, bold: true });
  addText(slide, "国家流感中心北方省份周度 ILI% · 气象数据 · 百度指数 · iTransformer", { x: 122, y: 438, w: 980, h: 36, size: 20, color: C.muted });
  addText(slide, "学生：张宇鑫  |  学号：22090032057  |  指导教师：刘艳艳", { x: 122, y: 902, w: 860, h: 40, size: 20, color: C.ink, bold: true });
  addText(slide, "中国海洋大学 信息科学与工程学部\n计算机科学与技术 2022级 · 2026年5月", { x: 122, y: 948, w: 780, h: 62, size: 17, color: C.muted });
  addText(slide, "16周输入\n预测未来4周", { x: 1326, y: 160, w: 300, h: 120, size: 34, color: C.teal, bold: true, align: "center" });
  addShape(slide, { x: 1326, y: 304, w: 300, h: 8, fill: solid(C.teal), line: { style: "none" } });
  addText(slide, "真实公开数据\n严格时间切分\n可复现工程链路", { x: 1326, y: 344, w: 330, h: 132, size: 25, color: C.ink, bold: true, align: "center" });
  slide.speakerNotes.text = "开场强调三点：真实公开数据、多源融合、未来4周预测。";
}

// Slide 2: problem and goal.
{
  const slide = presentation.slides.add();
  addCanvas(slide);
  addHeader(slide, "研究背景与目标", "从被动监测走向提前 1-4 周的趋势预警", 2);
  addMetric(slide, "预测窗口", "4", 154, 254, C.coral, "周");
  addMetric(slide, "输入历史", "16", 486, 254, C.teal, "周");
  addMetric(slide, "建模特征", "29", 818, 254, C.blue, "个");
  addBullet(slide, "现实问题", "流感具有季节性与非线性高峰，单一监测序列往往存在滞后，难以及时支撑医疗资源调度。", 132, 472, C.coral, 720);
  addBullet(slide, "研究目标", "融合流感监测、气象环境和搜索行为数据，构建可复现的周度趋势预测系统。", 132, 620, C.teal, 720);
  addBullet(slide, "答辩重点", "说明数据可信、模型合理、实验可复现，以及结论边界不外推到单一城市病例数。", 132, 768, C.blue, 720);
  addShape(slide, { x: 1118, y: 254, w: 580, h: 520, geometry: "roundRect", fill: solid("#EAF7F6"), line: { style: "solid", fill: "#C6E3E1", width: 1 } });
  addText(slide, "核心问题", { x: 1180, y: 316, w: 260, h: 44, size: 30, bold: true, color: C.teal2 });
  addText(slide, "如何在真实周度公共卫生数据上，利用外部环境与行为信号，提前判断未来流感活动变化？", { x: 1180, y: 392, w: 448, h: 168, size: 34, bold: true, color: C.ink });
  addLine(slide, 1180, 610, 1536, 610, C.teal, 5);
  addText(slide, "输出形式：预测曲线、模型指标、消融解释、Web 展示原型。", { x: 1180, y: 652, w: 430, h: 80, size: 22, color: C.muted });
  addFooter(slide);
  slide.speakerNotes.text = "本页回答为什么做：公共卫生监测有滞后，项目目标是做周度趋势预警而非病例确诊预测。";
}

// Slide 3: data scope.
{
  const slide = presentation.slides.add();
  addCanvas(slide);
  addHeader(slide, "数据来源与研究口径", "正式实验使用国家流感中心北方省份真实周度 ILI% 序列", 3);
  addTable(slide, [
    ["数据类型", "来源 / 口径", "规模", "建模作用"],
    ["流感监测", "中国国家流感中心周报，北方省份 ILI%", "849周；2010-01-04 至 2026-04-13", "预测目标 ili_rate"],
    ["气象环境", "Open-Meteo Archive，8个北方代表城市", "5594天；2011-01-01 至 2026-04-25", "温度、湿度、风速、气压"],
    ["搜索行为", "百度指数，流感/感冒/发烧关键词", "5593天；2011-01-01 至 2026-04-24", "公众关注度外生变量"],
    ["周度融合", "按 ISO 周对齐并特征工程", "797周融合；793周进入切分", "训练、验证、测试"],
  ], 4, { x: 120, y: 236, w: 1320, rowH: 72, colWidths: [170, 430, 360, 360], size: 18 });
  addShape(slide, { x: 1490, y: 245, w: 274, h: 274, geometry: "ellipse", fill: solid("#EAF7F6"), line: { style: "solid", fill: "#C6E3E1", width: 2 } });
  addText(slide, "北方省份", { x: 1530, y: 310, w: 194, h: 42, size: 28, bold: true, color: C.teal2, align: "center" });
  addText(slide, "公开可追溯\n周度监测", { x: 1524, y: 372, w: 206, h: 92, size: 28, bold: true, color: C.ink, align: "center" });
  addPill(slide, "不再使用平谷代理", 1492, 586, 276, C.coral);
  addPill(slide, "不使用模拟目标序列", 1492, 648, 276, C.teal);
  addPill(slide, "不外推为病例数", 1492, 710, 276, C.blue);
  addFooter(slide, "资料来源：README、final_data_audit.md、毕业论文第2章。");
  slide.speakerNotes.text = "这一页强调最终口径：北方省份 ILI%，不是北京或平谷病例预测。";
}

// Slide 4: data process and quality.
{
  const slide = presentation.slides.add();
  addCanvas(slide);
  addHeader(slide, "数据处理与质量审计", "从原始采集留痕到严格时序切分，降低数据泄漏风险", 4);
  addImage(slide, assets.pipeline, { x: 110, y: 238, w: 1140, h: 482, fit: "contain", alt: "多源数据处理与实验流程" });
  addMetric(slide, "ili_rate 剩余缺失", "0", 1350, 246, C.teal, "周");
  addMetric(slide, "有来源补齐", "3", 1350, 398, C.coral, "周");
  addMetric(slide, "positive_rate 审计补齐", "108", 1350, 550, C.blue, "周");
  addBullet(slide, "质量策略", "source_manifest.json 记录来源，data_quality_report.json 复核行数、缺失、重复日期与时间边界。", 132, 772, C.teal, 760);
  addBullet(slide, "边界说明", "positive_rate 仅作为原始监测留痕字段，默认不进入训练特征，避免不稳定 PDF 解析影响模型。", 966, 772, C.coral, 760);
  addFooter(slide, "资料来源：final_data_audit.md、data_quality_report.json、论文第2章。");
  slide.speakerNotes.text = "说明数据处理流程和补齐透明度。重点不是补齐很多，而是目标变量已经无缺失且有来源记录。";
}

// Slide 5: task and features.
{
  const slide = presentation.slides.add();
  addCanvas(slide);
  addHeader(slide, "预测任务与特征设计", "用过去 16 周多源变量预测未来 4 周 ILI%", 5);
  addShape(slide, { x: 150, y: 290, w: 640, h: 150, geometry: "roundRect", fill: solid("#EAF2FF"), line: { style: "solid", fill: "#BCD4FF", width: 1 } });
  addText(slide, "输入窗口", { x: 196, y: 324, w: 180, h: 36, size: 26, bold: true, color: C.blue });
  addText(slide, "过去 16 周 × 29 个特征", { x: 196, y: 374, w: 456, h: 44, size: 32, bold: true, color: C.ink });
  addLine(slide, 790, 365, 1070, 365, C.teal, 7);
  addShape(slide, { x: 1020, y: 338, w: 52, h: 52, geometry: "triangle", fill: solid(C.teal), line: { style: "none" } });
  addShape(slide, { x: 1110, y: 290, w: 640, h: 150, geometry: "roundRect", fill: solid("#FFF3E8"), line: { style: "solid", fill: "#FFD4B6", width: 1 } });
  addText(slide, "输出目标", { x: 1156, y: 324, w: 180, h: 36, size: 26, bold: true, color: C.coral });
  addText(slide, "未来 1-4 周 ILI%", { x: 1156, y: 374, w: 440, h: 44, size: 32, bold: true, color: C.ink });
  addTable(slide, [
    ["特征组", "代表变量", "设计目的"],
    ["流感历史", "ili_rate、1/2/4周滞后、4/8周滚动统计", "捕捉自相关与季节高峰"],
    ["气象环境", "温度、湿度、风速、气压、体感/交互项", "引入环境变化背景"],
    ["搜索行为", "流感、感冒、发烧搜索指数及变化率", "反映公众关注度变化"],
    ["时间特征", "week/month 周期编码、流感季标记", "表达周期性和季节性"],
  ], 3, { x: 150, y: 530, w: 1600, rowH: 66, colWidths: [220, 690, 690], headerFill: "#F1F5F9", accent: C.blue, size: 17 });
  addFooter(slide, "资料来源：experiment_summary.json、config/config.yaml、论文第2.3节。");
  slide.speakerNotes.text = "强调这是多步预测：不是只预测下一周，而是同时输出未来四周。";
}

// Slide 6: model and system.
{
  const slide = presentation.slides.add();
  addCanvas(slide);
  addHeader(slide, "模型与系统设计", "iTransformer 建模跨变量关系，工程链路覆盖采集、训练、评估和展示", 6);
  addText(slide, "iTransformer 的关键变化", { x: 130, y: 230, w: 620, h: 42, size: 30, bold: true, color: C.teal2 });
  addBullet(slide, "变量作为 token", "将每个变量的历史窗口序列作为一个 token，在变量维度计算注意力。", 132, 300, C.teal, 680);
  addBullet(slide, "目标 token 输出", "学习流感历史、气象与搜索行为之间的关联后，映射为未来 4 周预测。", 132, 446, C.blue, 680);
  addBullet(slide, "基准模型对照", "设置 LSTM、DLinear、ARIMA，分别代表循环网络、线性分解与传统统计方法。", 132, 592, C.coral, 680);
  addImage(slide, assets.architecture, { x: 840, y: 248, w: 910, h: 382, fit: "contain", alt: "系统总体架构" });
  addTable(slide, [
    ["层次", "项目实现"],
    ["数据层", "CSV / source_manifest.json"],
    ["处理层", "清洗、周聚合、特征工程、切分"],
    ["模型层", "iTransformer + 基准模型"],
    ["展示层", "Streamlit Web 原型"],
  ], 2, { x: 900, y: 690, w: 720, rowH: 56, colWidths: [170, 550], headerFill: "#EAF7F6", accent: C.teal, size: 18 });
  addFooter(slide, "资料来源：src/models/itransformer.py、scripts/train.py、web/app.py、论文第3章与第5章。");
  slide.speakerNotes.text = "用本页快速说明为什么用 iTransformer，以及项目不是单个模型脚本，而是一条完整工程链路。";
}

// Slide 7: experiment setup.
{
  const slide = presentation.slides.add();
  addCanvas(slide);
  addHeader(slide, "实验设计", "按时间顺序切分，避免未来信息进入训练", 7);
  const x0 = 154;
  const y0 = 330;
  const total = 1320;
  const trainW = 920;
  const valW = 198;
  const testW = 202;
  addShape(slide, { x: x0, y: y0, w: trainW, h: 86, fill: solid(C.teal), line: { style: "none" } });
  addShape(slide, { x: x0 + trainW, y: y0, w: valW, h: 86, fill: solid(C.blue), line: { style: "none" } });
  addShape(slide, { x: x0 + trainW + valW, y: y0, w: testW, h: 86, fill: solid(C.coral), line: { style: "none" } });
  addText(slide, "训练集 555行\n2011-01-31 至 2021-09-20", { x: x0 + 28, y: y0 + 16, w: trainW - 56, h: 60, size: 22, bold: true, color: C.white, align: "center", valign: "middle" });
  addText(slide, "验证集 119行\n2021-09-27 至 2024-01-01", { x: x0 + trainW + 8, y: y0 + 16, w: valW - 16, h: 60, size: 17, bold: true, color: C.white, align: "center", valign: "middle" });
  addText(slide, "测试集 119行\n2024-01-08 至 2026-04-13", { x: x0 + trainW + valW + 8, y: y0 + 16, w: testW - 16, h: 60, size: 17, bold: true, color: C.white, align: "center", valign: "middle" });
  addLine(slide, x0, y0 + 128, x0 + total, y0 + 128, C.line, 2);
  addText(slide, "2011", { x: x0 - 4, y: y0 + 148, w: 80, h: 28, size: 16, color: C.muted });
  addText(slide, "2021", { x: x0 + trainW - 28, y: y0 + 148, w: 80, h: 28, size: 16, color: C.muted });
  addText(slide, "2024", { x: x0 + trainW + valW - 28, y: y0 + 148, w: 80, h: 28, size: 16, color: C.muted });
  addText(slide, "2026", { x: x0 + total - 48, y: y0 + 148, w: 80, h: 28, size: 16, color: C.muted, align: "right" });
  addTable(slide, [
    ["训练配置", "取值"],
    ["随机种子", "42"],
    ["最大轮数 / 早停", "300 / patience=30"],
    ["优化器", "Adam，学习率 0.001，权重衰减 0.01"],
    ["评价指标", "RMSE、MAE、MAPE、R2、峰值指标"],
  ], 2, { x: 154, y: 590, w: 760, rowH: 58, colWidths: [210, 550], headerFill: "#F1F5F9", accent: C.blue, size: 18 });
  addBullet(slide, "实验边界", "指标用于当前北方地区真实周度监测口径下的模型比较，不作为单一城市病例数预测结论。", 1010, 620, C.coral, 610);
  addFooter(slide, "资料来源：experiment_summary.json、config/config.yaml、论文第4.1节。");
  slide.speakerNotes.text = "本页说明数据切分非常关键：按时间顺序，不随机打乱，避免未来信息泄漏。";
}

// Slide 8: model comparison.
{
  const slide = presentation.slides.add();
  addCanvas(slide);
  addHeader(slide, "模型对比结果", "iTransformer 在整体误差和解释方差上表现最好", 8);
  addMetric(slide, "iTransformer RMSE", "0.772", 128, 242, C.teal);
  addMetric(slide, "MAE", "0.491", 426, 242, C.blue);
  addMetric(slide, "MAPE", "11.51", 724, 242, C.coral, "%");
  addMetric(slide, "R²", "0.538", 1060, 242, C.green);
  addBarChart(slide, {
    x: 125,
    y: 466,
    w: 770,
    h: 430,
    title: "RMSE（越低越好）",
    categories: metrics.map((m) => m.model),
    values: metrics.map((m) => m.rmse),
    color: C.teal,
    yTitle: "RMSE",
  });
  addBarChart(slide, {
    x: 986,
    y: 466,
    w: 770,
    h: 430,
    title: "R2（越高越好）",
    categories: metrics.map((m) => m.model),
    values: metrics.map((m) => m.r2),
    color: C.green,
    yTitle: "R2",
  });
  addText(slide, "结论：当前数据版本下，iTransformer 的整体 RMSE 最低；ARIMA 在短期与峰值指标上仍有竞争力。", { x: 122, y: 912, w: 1390, h: 46, size: 24, bold: true, color: C.ink });
  addFooter(slide, "资料来源：all_metrics.json、experiment_brief.md、论文第4.2节。");
  slide.speakerNotes.text = "突出结果，同时保持谨慎：模型排序依赖当前数据版本、聚合方法和时间切分。";
}

// Slide 9: forecast and horizons.
{
  const slide = presentation.slides.add();
  addCanvas(slide);
  addHeader(slide, "预测效果与多步误差", "越远的预测步长不确定性越高，符合多步时间序列规律", 9);
  addImage(slide, assets.pred, { x: 94, y: 222, w: 1086, h: 630, fit: "cover", alt: "iTransformer 预测结果与误差分布" });
  addShape(slide, { x: 1230, y: 240, w: 508, h: 272, geometry: "roundRect", fill: solid("#FFFFFF"), line: { style: "solid", fill: C.line, width: 1 } });
  addText(slide, "分 Horizon 误差", { x: 1270, y: 274, w: 360, h: 36, size: 26, bold: true, color: C.teal2 });
  addLineChart(slide, {
    x: 1260,
    y: 328,
    w: 430,
    h: 220,
    title: "",
    categories: ["H1", "H2", "H3", "H4"],
    series: [
      { name: "RMSE", values: horizonRmse, color: C.teal },
    ],
  });
  addMetric(slide, "H1 RMSE", "0.525", 1260, 610, C.teal);
  addMetric(slide, "H4 RMSE", "0.983", 1510, 610, C.coral);
  addText(slide, "解释：第1周误差最低，预测距离增加后 RMSE 和 MAE 上升；高峰阶段仍是主要误差来源。", { x: 1260, y: 760, w: 430, h: 104, size: 22, bold: true, color: C.ink });
  addFooter(slide, "资料来源：iTransformer_predictions.png、horizon_metrics.json、论文第4.3节。");
  slide.speakerNotes.text = "这页讲模型可用性和局限：趋势有捕捉能力，但峰值时段误差仍更明显。";
}

// Slide 10: ablation.
{
  const slide = presentation.slides.add();
  addCanvas(slide);
  addHeader(slide, "消融实验与解释", "外生变量的贡献需要结合聚合噪声和流行季背景解释", 10);
  addBarChart(slide, {
    x: 128,
    y: 260,
    w: 840,
    h: 520,
    title: "不同特征组合 RMSE",
    categories: ablation.map((a) => a.label),
    values: ablation.map((a) => a.rmse),
    color: C.blue,
    yTitle: "RMSE",
  });
  addTable(slide, [
    ["特征组合", "特征数", "RMSE", "R²"],
    ...ablation.map((a) => [a.label, String(a.features), a.rmse.toFixed(3), a.r2.toFixed(3)]),
  ], 4, { x: 1050, y: 260, w: 640, rowH: 62, colWidths: [240, 120, 140, 140], headerFill: "#EAF7F6", accent: C.teal, size: 18 });
  addBullet(slide, "实验发现", "仅流感历史 RMSE 最低，流感+气象与三源融合非常接近。", 1050, 650, C.teal, 650);
  addBullet(slide, "答辩表述", "不能简单说“搜索指数一定提升性能”；更稳妥的结论是外生变量提供辅助解释，但受代表城市聚合口径影响。", 1050, 790, C.coral, 650);
  addFooter(slide, "资料来源：ablation_report.md、ablation_metrics.csv、论文第4.4节。");
  slide.speakerNotes.text = "答辩时主动说明消融结果，避免过度宣传多源数据一定提升。";
}

// Slide 11: web and reproducibility.
{
  const slide = presentation.slides.add();
  addCanvas(slide);
  addHeader(slide, "Web 展示与工程复现", "答辩演示可从结果文件直接读取，不依赖临时手工整理", 11);
  addImage(slide, assets.architecture, { x: 118, y: 226, w: 970, h: 410, fit: "contain", alt: "系统总体架构" });
  addText(slide, "Streamlit 原型功能", { x: 1160, y: 250, w: 430, h: 44, size: 30, bold: true, color: C.teal2 });
  addBullet(slide, "数据总览", "多源时序、样本数量、时间范围、相关性矩阵。", 1160, 326, C.teal, 550);
  addBullet(slide, "预测分析", "训练曲线、预测曲线、误差分布、注意力热力图。", 1160, 454, C.blue, 550);
  addBullet(slide, "对比实验", "多模型指标、消融实验与预警调度演示。", 1160, 582, C.coral, 550);
  addShape(slide, { x: 142, y: 704, w: 910, h: 150, geometry: "roundRect", fill: solid("#0B1220"), line: { style: "none" } });
  addText(slide, "python scripts/train.py --skip-collect\npython scripts/run_ablation.py --skip-collect\nstreamlit run web/app.py", { x: 188, y: 732, w: 830, h: 104, size: 22, color: "#DFF7EF", family: mono });
  addFooter(slide, "资料来源：README、web/app.py、scripts/train.py、scripts/run_ablation.py。");
  slide.speakerNotes.text = "本页说明系统可以复现，结果不是一次性的图。现场如果时间允许，可打开 web/app.py 对应的 Streamlit 页面。";
}

// Slide 12: conclusion.
{
  const slide = presentation.slides.add();
  addCanvas(slide);
  addHeader(slide, "结论与展望", "完成了真实数据、多源融合、深度模型和可视化系统的闭环", 12);
  addText(slide, "主要结论", { x: 126, y: 242, w: 420, h: 46, size: 34, bold: true, color: C.teal2 });
  addBullet(slide, "数据可信", "正式实验切换为国家流感中心北方省份周度 ILI%，并保留缺失补齐和来源审计。", 132, 320, C.teal, 760);
  addBullet(slide, "模型有效", "iTransformer 在测试集上 RMSE=0.772、MAE=0.491、MAPE=11.51%、R²=0.538。", 132, 466, C.blue, 760);
  addBullet(slide, "工程完整", "项目覆盖采集、预处理、特征工程、训练评估、消融实验和 Web 展示。", 132, 612, C.coral, 760);
  addText(slide, "不足与展望", { x: 1036, y: 242, w: 420, h: 46, size: 34, bold: true, color: C.coral });
  addBullet(slide, "区域聚合仍可细化", "百度指数和气象采用代表城市简单聚合，后续可引入人口权重或省级分层。", 1036, 320, C.coral, 650);
  addBullet(slide, "峰值预测仍需增强", "高发期误差更明显，可尝试高峰样本加权、分季节训练和概率预测。", 1036, 466, C.blue, 650);
  addBullet(slide, "解释性不等于因果", "注意力热力图只能辅助理解信息流，不能直接作为流感传播因果结论。", 1036, 612, C.teal, 650);
  addText(slide, "谢谢各位老师，请批评指正", { x: 360, y: 874, w: 1200, h: 70, size: 48, bold: true, color: C.ink, align: "center" });
  addFooter(slide, "资料来源：毕业论文第6-7章、experiment_brief.md、ablation_report.md。");
  slide.speakerNotes.text = "结尾回到闭环和边界：数据真实、模型表现、工程可展示，同时不夸大因果和泛化。";
}

const pendingImages = presentation.getPendingImageHydrationRequests();
if (pendingImages.length) {
  presentation.hydrateImageAssets(
    pendingImages.map((request) => ({
      assetId: request.assetId,
      contentType: request.contentType,
      data: new Uint8Array(fs.readFileSync(request.uri)),
    })),
  );
}

const pptxBlob = await PresentationFile.exportPptx(presentation);
const pptxPath = path.join(outputDir, "output.pptx");
await pptxBlob.save(pptxPath);

for (const [index, slide] of presentation.slides.items.entries()) {
  const pngBlob = await slide.export({ format: "png" });
  fs.writeFileSync(
    path.join(previewDir, `slide-${String(index + 1).padStart(2, "0")}.png`),
    Buffer.from(await pngBlob.arrayBuffer()),
  );
  const layout = await slide.export({ format: "layout" });
  fs.writeFileSync(path.join(layoutDir, `slide-${String(index + 1).padStart(2, "0")}.layout.json`), JSON.stringify(layout, null, 2), "utf8");
}

console.log(JSON.stringify({
  pptxPath,
  slideCount: presentation.slides.count,
  previewDir,
  layoutDir,
}, null, 2));
