"""
Fetch Baidu Index data and aggregate it for northern China representative regions.

Usage:
    $env:BAIDU_INDEX_COOKIE='your cookie'
    python scripts/fetch_baidu.py

Optional environment variables:
    BAIDU_INDEX_START=2011-01-01
    BAIDU_INDEX_END=2026-04-25
    BAIDU_INDEX_SCOPE=north      # north / beijing
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd


KEYWORDS = [["流感"], ["感冒"], ["发烧"]]
RENAME_MAP = {
    "流感": "flu_search_index",
    "感冒": "cold_search_index",
    "发烧": "fever_search_index",
}

# Municipalities use Baidu province-level codes; other entries use city codes.
NORTH_REPRESENTATIVE_AREAS = {
    "北京": 911,
    "天津": 923,
    "济南": 1,
    "石家庄": 141,
    "沈阳": 150,
    "哈尔滨": 152,
    "西安": 165,
    "太原": 231,
}

BEIJING_ONLY = {"北京": 911}


def fetch_region_keyword(get_search_index, cookie: str, start_date: str, end_date: str, region: str, area: int, keyword: list[str]) -> list[dict]:
    rows = []
    index_data = get_search_index(
        keywords_list=[keyword],
        start_date=start_date,
        end_date=end_date,
        cookies=cookie,
        area=area,
    )
    for item in index_data:
        if item.get("type") != "all":
            continue
        rows.append(
            {
                "date": item["date"],
                "region": region,
                "keyword": item["keyword"][0],
                "index": pd.to_numeric(item["index"], errors="coerce"),
            }
        )
    return rows


def aggregate_daily(rows: Iterable[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("未获取到任何百度指数数据。通常是 Cookie 失效、账户权限不足、关键词无权限或接口被限流。")

    df = df.dropna(subset=["index"]).drop_duplicates(subset=["date", "region", "keyword"])
    df["date"] = pd.to_datetime(df["date"])

    aggregated = (
        df.groupby(["date", "keyword"], as_index=False)
        .agg(index=("index", "mean"), contributing_regions=("region", "nunique"))
        .sort_values(["date", "keyword"])
    )
    pivot = aggregated.pivot(index="date", columns="keyword", values="index").reset_index()
    pivot.rename(columns=RENAME_MAP, inplace=True)

    region_counts = aggregated.pivot(index="date", columns="keyword", values="contributing_regions").reset_index()
    region_counts.rename(
        columns={keyword: f"{RENAME_MAP[keyword]}_region_count" for keyword in RENAME_MAP},
        inplace=True,
    )

    result = pivot.merge(region_counts, on="date", how="left")
    result = result.sort_values("date").reset_index(drop=True)
    return result


def fetch_data() -> None:
    try:
        from qdata.baidu_index import get_search_index
    except ImportError as exc:
        raise RuntimeError(
            "未安装 qdata，请先执行 `pip install qdata -i https://pypi.tuna.tsinghua.edu.cn/simple`"
        ) from exc

    cookie = os.environ.get("BAIDU_INDEX_COOKIE", "").strip()
    if not cookie:
        raise RuntimeError(
            "缺少 BAIDU_INDEX_COOKIE。请先登录百度指数，在浏览器开发者工具中复制 Cookie，"
            "然后设置环境变量后再运行。"
        )

    start_date = os.environ.get("BAIDU_INDEX_START", "2011-01-01")
    end_date = os.environ.get("BAIDU_INDEX_END", "2026-04-25")
    scope = os.environ.get("BAIDU_INDEX_SCOPE", "north").strip().lower()

    if scope == "beijing":
        areas = BEIJING_ONLY
        save_path = Path("data/raw/search/beijing_baidu_index.csv")
        scope_label = "北京"
    else:
        areas = NORTH_REPRESENTATIVE_AREAS
        save_path = Path("data/raw/search/north_baidu_index.csv")
        scope_label = "北方代表城市/省份"

    all_rows = []
    failures = []
    print(f"开始爬取百度指数（{scope_label}聚合）: {start_date} -> {end_date}")
    for region, area in areas.items():
        for keyword in KEYWORDS:
            name = keyword[0]
            print(f"正在获取: {region} / {name}")
            try:
                all_rows.extend(fetch_region_keyword(get_search_index, cookie, start_date, end_date, region, area, keyword))
            except Exception as exc:
                failures.append({"region": region, "keyword": name, "error": str(exc)})
                print(f"  获取失败: {region} / {name}: {exc}")

    df_pivot = aggregate_daily(all_rows)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df_pivot.to_csv(save_path, index=False, encoding="utf-8-sig")

    if failures:
        failure_path = save_path.with_name(save_path.stem + "_failures.json")
        failure_path.write_text(pd.Series(failures).to_json(force_ascii=False, indent=2), encoding="utf-8")
        print(f"部分区域/关键词获取失败，详情已保存: {failure_path}")

    print(f"成功保存百度指数数据: {save_path}")
    print(f"数据行数: {len(df_pivot)}, 时间范围: {df_pivot['date'].min().date()} -> {df_pivot['date'].max().date()}")


if __name__ == "__main__":
    fetch_data()
