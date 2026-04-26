"""
Fetch weekly influenza surveillance data from the Chinese National Influenza Center.

The script builds a real weekly target series for northern China from official CNIC
weekly reports. It first parses report HTML pages and falls back to attached PDFs
when recent pages only expose summary text.

Default output:
    data/raw/flu/cnic_north_weekly_flu.csv

Recommended run:
    python scripts/fetch_flu_cn_weekly.py --start-year 2010 --end-year 2026
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

REPORT_LISTS = [
    "https://ivdc.chinacdc.cn/cnic/zyzx/lgzb/",
    "https://ivdc.chinacdc.cn/lgzx/zyzx/lgzb/",
]


@dataclass(frozen=True)
class ReportLink:
    year: int
    week: int
    url: str
    list_url: str


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("（", "(").replace("）", ")")
    return re.sub(r"\s+", " ", text).strip()


def iso_week_monday(year: int, week: int) -> str:
    return datetime.strptime(f"{year}-W{week:02d}-1", "%G-W%V-%u").date().isoformat()


def safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    return float(value.replace(",", "").replace(" ", ""))


def cache_name(url: str, suffix: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"{digest}{suffix}"


def fetch_bytes(url: str, cache_path: Path, refresh: bool = False, timeout: int = 60) -> bytes:
    if cache_path.exists() and not refresh:
        return cache_path.read_bytes()

    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(resp.content)
    return resp.content


def fetch_html(url: str, cache_path: Path, refresh: bool = False) -> str:
    raw = fetch_bytes(url, cache_path, refresh=refresh, timeout=30)
    for encoding in ("utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def collect_report_links(max_pages: int = 80) -> list[ReportLink]:
    dedup: Dict[tuple[int, int], ReportLink] = {}

    for base_url in REPORT_LISTS:
        for page_idx in range(max_pages):
            page_url = base_url + ("index.htm" if page_idx == 0 else f"index_{page_idx}.htm")
            try:
                resp = requests.get(page_url, headers=HEADERS, timeout=30)
            except requests.RequestException:
                break
            if resp.status_code != 200 or len(resp.text) < 1000:
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            page_count = 0
            for link in soup.find_all("a"):
                text = normalize_text(link.get_text(" ", strip=True))
                match = re.match(r"^(20\d{2})\s*第\s*(\d+)\s*周$", text)
                href = link.get("href")
                if not match or not href:
                    continue

                year = int(match.group(1))
                week = int(match.group(2))
                key = (year, week)
                dedup.setdefault(
                    key,
                    ReportLink(
                        year=year,
                        week=week,
                        url=urljoin(page_url, href),
                        list_url=base_url,
                    ),
                )
                page_count += 1

            if page_count == 0:
                break

    return sorted(dedup.values(), key=lambda item: (item.year, item.week))


def html_to_text_and_pdf_url(html: str, page_url: str) -> tuple[str, Optional[str]]:
    soup = BeautifulSoup(html, "html.parser")
    text = normalize_text(soup.get_text(" ", strip=True))

    pdf_url = None
    for link in soup.find_all("a"):
        href = link.get("href", "")
        if href.lower().endswith(".pdf"):
            pdf_url = urljoin(page_url, href)
            break
    return text, pdf_url


def extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("缺少 PyMuPDF，请先执行 `pip install PyMuPDF`。") from exc

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return normalize_text("\n".join(page.get_text() for page in doc))


def extract_ili_rate(text: str, region: str) -> Optional[float]:
    region_key = f"{region}省份"
    number = r"(\d+(?:\s*\.\s*\d+)?)"

    pair_match = re.search(
        rf"南、北方省份[^。；;]*?ILI%[^。；;]*?分别为\s*{number}\s*%\s*[、,，]\s*{number}\s*%",
        text,
    )
    if pair_match:
        return safe_float(pair_match.group(2) if region == "北方" else pair_match.group(1))

    for sentence in re.split(r"[。；;]", text):
        sentence = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", sentence)
        if region_key not in sentence or "ILI%" not in sentence:
            continue
        match = re.search(rf"ILI%\s*\)?\s*(?:为|\()?[\s:：]*{number}\s*%", sentence)
        if match:
            return safe_float(match.group(1))
    return None


def extract_positive_table(text: str) -> Dict[str, Optional[float]]:
    match = re.search(
        r"检测数\s+(\d+)\s+(\d+)\s+(\d+)\s+"
        r"阳性数\s*\(\s*%\s*\)\s+"
        r"(\d+)\s*\((\d+(?:\.\d+)?)%\)\s+"
        r"(\d+)\s*\((\d+(?:\.\d+)?)%\)\s+"
        r"(\d+)\s*\((\d+(?:\.\d+)?)%\)",
        text,
    )
    if not match:
        return {}

    return {
        "southern_sample_count": int(match.group(1)),
        "northern_sample_count": int(match.group(2)),
        "total_sample_count": int(match.group(3)),
        "southern_positive_count": int(match.group(4)),
        "southern_positive_rate": safe_float(match.group(5)),
        "positive_count": int(match.group(6)),
        "positive_rate": safe_float(match.group(7)),
        "total_positive_count": int(match.group(8)),
        "total_positive_rate": safe_float(match.group(9)),
    }


def extract_positive_rate_from_sentences(text: str, region: str) -> Optional[float]:
    region_key = f"{region}省份"

    pair_match = re.search(
        r"南、北方省份(?:流感病毒|流感|标本)?检测阳性率分别(?:为|是)?\s*"
        r"(\d+(?:\.\d+)?)\s*%\s*[、,，]\s*(\d+(?:\.\d+)?)\s*%",
        text,
    )
    if pair_match:
        return safe_float(pair_match.group(2) if region == "北方" else pair_match.group(1))

    for sentence in re.split(r"[。；;]", text):
        if region_key not in sentence:
            continue
        if "未检测到流感" in sentence and "阳性" in sentence:
            return 0.0
        if "阳性率" not in sentence and "检测阳性率" not in sentence:
            continue
        match = re.search(
            r"(?:流感病毒|流感|标本)?检测?阳性率"
            r".{0,20}?(?:为|升至|上升至|下降至|达到|达)\s*(\d+(?:\.\d+)?)\s*%",
            sentence,
        )
        if match:
            return safe_float(match.group(1))
    return None


def parse_report(
    report: ReportLink,
    cache_dir: Path,
    refresh: bool = False,
    allow_pdf: bool = True,
) -> dict:
    html_cache = cache_dir / "html" / f"{report.year}_{report.week:02d}_{cache_name(report.url, '.html')}"
    row = {
        "date": iso_week_monday(report.year, report.week),
        "year": report.year,
        "week": report.week,
        "ili_rate": None,
        "positive_rate": None,
        "positive_count": None,
        "sample_count": None,
        "southern_ili_rate": None,
        "southern_positive_rate": None,
        "southern_positive_count": None,
        "southern_sample_count": None,
        "source_type": "official_cnic_report",
        "source_url": report.url,
        "pdf_url": None,
        "parse_method": "html",
        "parse_status": "ok",
        "parse_note": "",
    }

    try:
        html = fetch_html(report.url, html_cache, refresh=refresh)
        text, pdf_url = html_to_text_and_pdf_url(html, report.url)
        row["pdf_url"] = pdf_url
    except Exception as exc:
        row["parse_status"] = "failed"
        row["parse_note"] = f"HTML fetch failed: {exc}"
        return row

    def fill_from_text(candidate_text: str) -> None:
        row["ili_rate"] = row["ili_rate"] if row["ili_rate"] is not None else extract_ili_rate(candidate_text, "北方")
        row["southern_ili_rate"] = (
            row["southern_ili_rate"]
            if row["southern_ili_rate"] is not None
            else extract_ili_rate(candidate_text, "南方")
        )

        table_values = extract_positive_table(candidate_text)
        if table_values:
            row["positive_rate"] = row["positive_rate"] if row["positive_rate"] is not None else table_values.get("positive_rate")
            row["positive_count"] = row["positive_count"] if row["positive_count"] is not None else table_values.get("positive_count")
            row["sample_count"] = row["sample_count"] if row["sample_count"] is not None else table_values.get("northern_sample_count")
            row["southern_positive_rate"] = (
                row["southern_positive_rate"]
                if row["southern_positive_rate"] is not None
                else table_values.get("southern_positive_rate")
            )
            row["southern_positive_count"] = (
                row["southern_positive_count"]
                if row["southern_positive_count"] is not None
                else table_values.get("southern_positive_count")
            )
            row["southern_sample_count"] = (
                row["southern_sample_count"]
                if row["southern_sample_count"] is not None
                else table_values.get("southern_sample_count")
            )

        row["positive_rate"] = (
            row["positive_rate"]
            if row["positive_rate"] is not None
            else extract_positive_rate_from_sentences(candidate_text, "北方")
        )
        row["southern_positive_rate"] = (
            row["southern_positive_rate"]
            if row["southern_positive_rate"] is not None
            else extract_positive_rate_from_sentences(candidate_text, "南方")
        )

    fill_from_text(text)

    needs_pdf = row["ili_rate"] is None or row["positive_rate"] is None
    if allow_pdf and needs_pdf and row["pdf_url"]:
        try:
            pdf_cache = cache_dir / "pdf" / f"{report.year}_{report.week:02d}_{cache_name(row['pdf_url'], '.pdf')}"
            pdf_bytes = fetch_bytes(str(row["pdf_url"]), pdf_cache, refresh=refresh, timeout=120)
            fill_from_text(extract_pdf_text(pdf_bytes))
            row["parse_method"] = "html+pdf"
        except Exception as exc:
            row["parse_note"] = f"PDF fallback failed: {exc}"

    missing = [name for name in ("ili_rate", "positive_rate") if row[name] is None]
    if missing:
        row["parse_status"] = "partial"
        note = f"missing {','.join(missing)}"
        row["parse_note"] = f"{row['parse_note']}; {note}".strip("; ")

    if report.year == 2010 and report.week <= 13 and row["ili_rate"] is not None:
        row["parse_note"] = (
            f"{row['parse_note']}; early 2010 report may use internal-medicine ILI% before the wording stabilized"
        ).strip("; ")

    return row


def write_outputs(rows: Iterable[dict], output_path: Path, report_path: Path) -> None:
    df = pd.DataFrame(rows).sort_values(["year", "week"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = {
        "rows": int(len(df)),
        "date_range": {
            "start": None if df.empty else str(df["date"].min()),
            "end": None if df.empty else str(df["date"].max()),
        },
        "parse_status_counts": df["parse_status"].value_counts(dropna=False).to_dict() if not df.empty else {},
        "missing_counts": df.isna().sum().astype(int).to_dict() if not df.empty else {},
        "output_path": str(output_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def write_checkpoint(rows: Iterable[dict], output_path: Path, report_path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    partial_output = output_path.with_suffix(".partial.csv")
    partial_report = report_path.with_suffix(".partial.json")
    write_outputs(rows, partial_output, partial_report)


def main() -> int:
    parser = argparse.ArgumentParser(description="采集中国国家流感中心北方省份周度流感监测数据")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=datetime.now().year)
    parser.add_argument("--output", default="data/raw/flu/cnic_north_weekly_flu.csv")
    parser.add_argument("--cache-dir", default="data/raw/flu/cnic_cache")
    parser.add_argument("--report", default="results/reports/cnic_weekly_parse_report.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 份报告，用于调试")
    parser.add_argument("--no-pdf", action="store_true", help="只解析 HTML，不下载 PDF")
    parser.add_argument("--refresh", action="store_true", help="忽略本地缓存，重新下载 HTML/PDF")
    args = parser.parse_args()

    links = [
        item
        for item in collect_report_links()
        if args.start_year <= item.year <= args.end_year
    ]
    if args.limit:
        links = links[: args.limit]

    if not links:
        print("未在国家流感中心周报列表中找到匹配年份的数据。", file=sys.stderr)
        return 1

    print(f"发现周报 {len(links)} 份，年份范围 {links[0].year}-{links[-1].year}", flush=True)
    cache_dir = Path(args.cache_dir)

    rows = []
    output_path = Path(args.output)
    report_path = Path(args.report)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(
                parse_report,
                report,
                cache_dir,
                args.refresh,
                not args.no_pdf,
            )
            for report in links
        ]
        for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            if idx % 10 == 0 or idx == len(futures):
                write_checkpoint(rows, output_path, report_path)
            if idx % 25 == 0 or idx == len(futures):
                print(f"已处理 {idx}/{len(futures)}", flush=True)

    write_outputs(rows, output_path, report_path)

    df = pd.DataFrame(rows)
    status_counts = df["parse_status"].value_counts(dropna=False).to_dict()
    print(f"已保存: {output_path}", flush=True)
    print(f"解析报告: {report_path}", flush=True)
    print(f"解析状态: {status_counts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
