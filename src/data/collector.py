"""
数据采集与多源对齐模块。

设计原则：
    1. 毕设默认启用严格真实数据模式，不再回退到模拟/占位数据。
    2. 所有数据源都需要在 source_manifest.json 中登记来源、区域、粒度与文件路径。
    3. 合并后自动输出数据质量报告，便于中期检查、答辩和论文附录留档。
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests

from src.data.quality import DataQualityAuditor, assert_required_columns, assert_source_manifest


class SourceManifest:
    """读取并校验原始数据来源清单。"""

    REQUIRED_KEYS = {"path", "source_name", "region", "granularity", "collection_method"}

    def __init__(self, manifest_path: str, strict: bool = True):
        self.manifest_path = manifest_path
        self.strict = strict
        self.data = assert_source_manifest(manifest_path, strict=strict)

    def get(self, dataset_name: str) -> Dict[str, str]:
        if dataset_name not in self.data:
            raise KeyError(
                f"source_manifest.json 中缺少 `{dataset_name}` 配置，请先登记该数据源的路径与来源信息。"
            )

        entry = self.data[dataset_name]
        missing = sorted(self.REQUIRED_KEYS - set(entry.keys()))
        if missing:
            raise KeyError(f"source_manifest.json 的 `{dataset_name}` 缺少字段: {missing}")
        return entry


class FluDataCollector:
    """流感监测数据读取器。当前默认读取国家流感中心北方省份周度序列。"""

    BASE_REQUIRED_COLS = ["date", "year", "week"]

    def __init__(self, save_dir: str = "data/raw/flu", required_cols=None):
        self.save_dir = save_dir
        self.required_cols = required_cols or self.BASE_REQUIRED_COLS
        os.makedirs(save_dir, exist_ok=True)

    def fetch(self, manifest_entry: Dict[str, str], start_year: int, end_year: int) -> pd.DataFrame:
        source_path = manifest_entry["path"]
        region = manifest_entry.get("region", "未标注区域")
        if not os.path.exists(source_path):
            raise FileNotFoundError(
                f"未找到流感真实数据文件: {source_path}（区域：{region}）。"
                "请重新爬取或整理官方周报后再运行。"
            )

        df = pd.read_csv(source_path)
        assert_required_columns(df, self.required_cols, "flu")

        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()
        df = df.sort_values("date").reset_index(drop=True)

        standardized_path = os.path.join(self.save_dir, "flu_data.csv")
        df.to_csv(standardized_path, index=False, encoding="utf-8-sig")
        print(f"[FluDataCollector] 已加载流感真实数据: {source_path}（区域：{region}）")
        return df


class WeatherDataCollector:
    """历史气象数据读取器；如 manifest 指定 API，可自动拉取并缓存。"""

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    REQUIRED_COLS = ["date", "temperature", "humidity", "wind_speed", "pressure"]
    DEFAULT_LOCATION = {"name": "北京", "latitude": 39.9042, "longitude": 116.4074}

    def __init__(self, save_dir: str = "data/raw/weather", locations: Optional[List[Dict[str, float]]] = None):
        self.save_dir = save_dir
        self.locations = locations or [self.DEFAULT_LOCATION]
        os.makedirs(save_dir, exist_ok=True)

    def fetch(self, manifest_entry: Dict[str, str], start_date: str, end_date: str) -> pd.DataFrame:
        source_path = manifest_entry["path"]
        collection_method = manifest_entry.get("collection_method", "").lower()

        if os.path.exists(source_path):
            df = pd.read_csv(source_path)
            print(f"[WeatherDataCollector] 已加载历史气象数据: {source_path}")
        elif collection_method == "open-meteo-api":
            df = self._fetch_from_api(start_date, end_date)
            os.makedirs(os.path.dirname(source_path), exist_ok=True)
            df.to_csv(source_path, index=False, encoding="utf-8-sig")
            print(f"[WeatherDataCollector] 已通过 Open-Meteo 拉取并缓存到: {source_path}")
        else:
            raise FileNotFoundError(
                f"未找到气象真实数据文件: {source_path}，且 manifest 未声明可自动拉取。"
            )

        assert_required_columns(df, self.REQUIRED_COLS, "weather")
        df["date"] = pd.to_datetime(df["date"])
        standardized_path = os.path.join(self.save_dir, "weather_data.csv")
        df.to_csv(standardized_path, index=False, encoding="utf-8-sig")
        return df.sort_values("date").reset_index(drop=True)

    def _fetch_from_api(self, start_date: str, end_date: str) -> pd.DataFrame:
        location_frames = []
        for location in self.locations:
            location_frames.append(self._fetch_location_from_api(location, start_date, end_date))

        if not location_frames:
            raise RuntimeError("未配置可用气象站点，无法构建真实气象数据。")

        df = pd.concat(location_frames, ignore_index=True)
        if "location" not in df.columns or df["location"].nunique() == 1:
            return df.drop(columns=["location"], errors="ignore")

        value_cols = ["temperature", "humidity", "wind_speed", "pressure"]
        aggregated = df.groupby("date", as_index=False)[value_cols].mean()
        return aggregated.sort_values("date").reset_index(drop=True)

    def _fetch_location_from_api(self, location: Dict[str, float], start_date: str, end_date: str) -> pd.DataFrame:
        name = str(location.get("name", "未命名站点"))
        latitude = float(location["latitude"])
        longitude = float(location["longitude"])
        all_data = []
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        current_start = start

        while current_start <= end:
            current_end = min(current_start + timedelta(days=365), end)
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": current_start.strftime("%Y-%m-%d"),
                "end_date": current_end.strftime("%Y-%m-%d"),
                "daily": ",".join(
                    [
                        "temperature_2m_mean",
                        "relative_humidity_2m_mean",
                        "wind_speed_10m_max",
                        "surface_pressure_mean",
                    ]
                ),
                "timezone": "Asia/Shanghai",
            }
            resp = None
            success = False
            last_exc = None
            for attempt in range(4):
                try:
                    resp = requests.get(self.BASE_URL, params=params, timeout=45)
                    resp.raise_for_status()
                    success = True
                    break
                except requests.RequestException as exc:
                    last_exc = exc
                    if attempt < 3:
                        wait_seconds = 2 * (attempt + 1)
                        print(f"[WeatherDataCollector] {name} {params['start_date']}~{params['end_date']} 拉取失败，{wait_seconds}s 后重试: {exc}")
                        time.sleep(wait_seconds)
            if not success or resp is None:
                raise RuntimeError(
                    f"Open-Meteo 拉取失败: {name} {params['start_date']}~{params['end_date']}"
                ) from last_exc

            data = resp.json().get("daily", {})
            if data:
                chunk = pd.DataFrame(
                    {
                        "date": data["time"],
                        "location": name,
                        "temperature": data.get("temperature_2m_mean"),
                        "humidity": data.get("relative_humidity_2m_mean"),
                        "wind_speed": data.get("wind_speed_10m_max"),
                        "pressure": data.get("surface_pressure_mean"),
                    }
                )
                all_data.append(chunk)

            current_start = current_end + timedelta(days=1)

        if not all_data:
            raise RuntimeError(f"Open-Meteo 返回为空，无法构建 {name} 的真实气象数据。")

        df = pd.concat(all_data, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        return df


class SearchIndexCollector:
    """搜索指数读取器。"""

    REQUIRED_COLS = ["date", "flu_search_index", "cold_search_index", "fever_search_index"]

    def __init__(self, save_dir: str = "data/raw/search"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def fetch(self, manifest_entry: Dict[str, str]) -> pd.DataFrame:
        source_path = manifest_entry["path"]
        if not os.path.exists(source_path):
            raise FileNotFoundError(
                f"未找到搜索指数真实数据文件: {source_path}。"
                "请优先使用 scripts/fetch_baidu.py 重新抓取或人工导出后再运行训练。"
            )

        df = pd.read_csv(source_path)
        assert_required_columns(df, self.REQUIRED_COLS, "search")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        standardized_path = os.path.join(self.save_dir, "search_data.csv")
        df.to_csv(standardized_path, index=False, encoding="utf-8-sig")
        print(f"[SearchIndexCollector] 已加载真实搜索指数数据: {source_path}")
        return df


class MultiSourceDataCollector:
    """多源数据统一读取、周粒度对齐与质量审计。"""

    def __init__(self, config: dict):
        self.config = config
        data_cfg = config.get("data", {})

        raw_dir = data_cfg.get("raw_dir", "data/raw")
        reports_dir = config.get("reporting", {}).get("reports_dir", "results/reports")
        strict_real_data = data_cfg.get("strict_real_data", True)
        manifest_path = data_cfg.get("manifest_path", os.path.join(raw_dir, "source_manifest.json"))

        self.manifest = SourceManifest(manifest_path=manifest_path, strict=strict_real_data)
        self.auditor = DataQualityAuditor(reports_dir=reports_dir)

        flu_cols = config.get("features", {}).get("flu_cols", [])
        self.flu_required_cols = ["date", "year", "week"] + flu_cols
        self.flu_collector = FluDataCollector(os.path.join(raw_dir, "flu"), self.flu_required_cols)
        self.weather_collector = WeatherDataCollector(
            os.path.join(raw_dir, "weather"),
            locations=data_cfg.get("weather_locations"),
        )
        self.search_collector = SearchIndexCollector(os.path.join(raw_dir, "search"))

    def collect_all(self) -> pd.DataFrame:
        data_cfg = self.config.get("data", {})
        start_year = data_cfg.get("start_year", 2018)
        end_year = data_cfg.get("end_year", 2025)
        start_date = f"{start_year}-01-01"
        configured_end = datetime.strptime(f"{end_year}-12-31", "%Y-%m-%d")
        end_date = min(configured_end, datetime.today()).strftime("%Y-%m-%d")

        flu_meta = self.manifest.get("flu")
        weather_meta = self.manifest.get("weather")
        search_meta = self.manifest.get("search")

        flu_df = self.flu_collector.fetch(flu_meta, start_year, end_year)
        weather_df = self.weather_collector.fetch(weather_meta, start_date, end_date)
        search_df = self.search_collector.fetch(search_meta)

        weather_weekly = self._aggregate_to_weekly(
            weather_df,
            agg_funcs={
                "temperature": "mean",
                "humidity": "mean",
                "wind_speed": "mean",
                "pressure": "mean",
            },
        )
        search_weekly = self._aggregate_to_weekly(
            search_df,
            agg_funcs={
                "flu_search_index": "mean",
                "cold_search_index": "mean",
                "fever_search_index": "mean",
            },
        )

        merged = self._merge_datasets(flu_df, weather_weekly, search_weekly)

        processed_dir = data_cfg.get("processed_dir", "data/processed")
        os.makedirs(processed_dir, exist_ok=True)
        merged_path = os.path.join(processed_dir, "merged_dataset.csv")
        merged.to_csv(merged_path, index=False, encoding="utf-8-sig")

        quality_report = {
            "manifest": {
                "path": self.manifest.manifest_path,
                "flu": flu_meta,
                "weather": weather_meta,
                "search": search_meta,
            },
            "datasets": {
                "flu": self.auditor.audit_dataset(flu_df, "flu", self.flu_required_cols, "weekly"),
                "weather": self.auditor.audit_dataset(
                    weather_df, "weather", WeatherDataCollector.REQUIRED_COLS, "daily"
                ),
                "search": self.auditor.audit_dataset(
                    search_df, "search", SearchIndexCollector.REQUIRED_COLS, "daily"
                ),
                "merged": self.auditor.audit_dataset(
                    merged,
                    "merged",
                    [
                        "date",
                        "year",
                        "week",
                        *self.config.get("features", {}).get("flu_cols", []),
                        "temperature",
                        "humidity",
                        "wind_speed",
                        "pressure",
                        "flu_search_index",
                        "cold_search_index",
                        "fever_search_index",
                    ],
                    "weekly",
                ),
            },
            "relationship_checks": self.auditor.audit_relationships(
                merged,
                target_col=self.config.get("features", {}).get("target_col", "ili_cases"),
            ),
        }
        self.auditor.save_report(quality_report, filename="data_quality_report.json")

        print("\n[MultiSourceDataCollector] 数据采集与对齐完成")
        print(f"  - 合并样本数: {len(merged)}")
        print(f"  - 时间范围: {merged['date'].min()} ~ {merged['date'].max()}")
        print(f"  - 数据质量报告: {os.path.join(self.auditor.reports_dir, 'data_quality_report.json')}")

        return merged

    def _aggregate_to_weekly(self, df: pd.DataFrame, agg_funcs: Dict[str, str]) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        iso = df["date"].dt.isocalendar()
        df["year"] = iso.year.astype(int)
        df["week"] = iso.week.astype(int)

        weekly = df.groupby(["year", "week"], as_index=False).agg(agg_funcs)
        weekly["date"] = weekly.apply(
            lambda row: datetime.strptime(
                f"{int(row['year'])}-W{int(row['week']):02d}-1", "%G-W%V-%u"
            ),
            axis=1,
        )
        return weekly.sort_values("date").reset_index(drop=True)

    def _merge_datasets(self, flu_df: pd.DataFrame, weather_df: pd.DataFrame, search_df: pd.DataFrame) -> pd.DataFrame:
        flu_cols = self.config.get("features", {}).get("flu_cols", [])
        metadata_cols = [
            col
            for col in ["flu_season", "digitized_source", "source_type"]
            if col in flu_df.columns and col not in flu_cols
        ]
        merged = flu_df[["date", "year", "week"] + flu_cols + metadata_cols].copy()

        weather_cols = [c for c in weather_df.columns if c not in {"date", "year", "week"}]
        search_cols = [c for c in search_df.columns if c not in {"date", "year", "week"}]

        merged = merged.merge(weather_df[["year", "week"] + weather_cols], on=["year", "week"], how="left")
        merged = merged.merge(search_df[["year", "week"] + search_cols], on=["year", "week"], how="left")

        merged["date"] = pd.to_datetime(merged["date"])
        return merged.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    import yaml

    with open("config/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    collector = MultiSourceDataCollector(cfg)
    data = collector.collect_all()
    print(data.head())
