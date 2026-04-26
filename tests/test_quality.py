import json

import pandas as pd
import pytest

from src.data.quality import DataQualityAuditor, assert_required_columns, assert_source_manifest


def test_assert_source_manifest_requires_file_in_strict_mode(tmp_path):
    missing_path = tmp_path / "source_manifest.json"

    with pytest.raises(FileNotFoundError):
        assert_source_manifest(str(missing_path), strict=True)


def test_assert_source_manifest_reads_json(tmp_path):
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(json.dumps({"flu": {"path": "flu.csv"}}), encoding="utf-8")

    assert assert_source_manifest(str(manifest_path), strict=True)["flu"]["path"] == "flu.csv"


def test_assert_required_columns_reports_missing_columns():
    df = pd.DataFrame({"date": ["2026-01-05"], "ili_rate": [2.3]})

    with pytest.raises(ValueError, match="week"):
        assert_required_columns(df, ["date", "week", "ili_rate"], "flu")


def test_data_quality_auditor_flags_duplicate_dates_and_constant_columns(tmp_path):
    df = pd.DataFrame(
        {
            "date": ["2026-01-05", "2026-01-05", "2026-01-19"],
            "ili_rate": [2.0, 2.5, 3.0],
            "region_count": [8, 8, 8],
        }
    )

    auditor = DataQualityAuditor(reports_dir=str(tmp_path))
    report = auditor.audit_dataset(df, "flu", ["date", "ili_rate"], "weekly")

    assert report["duplicate_dates"] == 1
    assert report["is_chronological"] is True
    assert "region_count" in report["constant_numeric_columns"]
