import numpy as np
import pandas as pd

from src.data.preprocessor import DataPreprocessor


def make_config(tmp_path):
    return {
        "data": {
            "lookback_window": 3,
            "forecast_horizon": 2,
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "splits_dir": str(tmp_path / "splits"),
        },
        "features": {
            "target_col": "ili_rate",
            "use_engineered_features": True,
            "exclude_from_training": ["year", "week"],
        },
    }


def make_weekly_frame(rows=30):
    dates = pd.date_range("2025-01-06", periods=rows, freq="W-MON")
    return pd.DataFrame(
        {
            "date": dates,
            "year": dates.year,
            "week": dates.isocalendar().week.astype(int),
            "ili_rate": np.linspace(1.0, 4.0, rows),
            "temperature": np.linspace(-5.0, 25.0, rows),
            "humidity": np.linspace(40.0, 80.0, rows),
            "flu_search_index": np.linspace(100.0, 200.0, rows),
        }
    )


def test_preprocessor_uses_chronological_splits_without_edge_overlap(tmp_path):
    preprocessor = DataPreprocessor(make_config(tmp_path))

    X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.process(make_weekly_frame())

    assert X_train.shape == (14, 4, 3)
    assert y_train.shape == (14, 2)
    assert X_val.shape == (2, 4, 3)
    assert y_val.shape == (2, 2)
    assert X_test.shape == (2, 4, 3)
    assert y_test.shape == (2, 2)
    assert preprocessor.feature_cols[0] == "ili_rate"
    assert preprocessor.split_metadata["train_date_range"]["end"] == "2025-05-05"
    assert preprocessor.split_metadata["val_date_range"]["start"] == "2025-05-12"
    assert preprocessor.split_metadata["test_date_range"]["start"] == "2025-06-23"


def test_preprocessor_interpolates_missing_target_before_sequence_creation(tmp_path):
    df = make_weekly_frame()
    df.loc[5, "ili_rate"] = np.nan
    preprocessor = DataPreprocessor(make_config(tmp_path))

    X_train, y_train, *_ = preprocessor.process(df)

    assert not np.isnan(X_train).any()
    assert not np.isnan(y_train).any()
