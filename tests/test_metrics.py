import numpy as np

from src.utils.metrics import compute_all_metrics, compute_horizon_metrics, mape, r2_score, rmse


def test_basic_metrics_use_original_scale_values():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])

    assert rmse(y_true, y_pred) == np.sqrt(4 / 3)
    assert r2_score(y_true, y_pred) == -1.0
    assert round(mape(y_true, y_pred), 6) == round((0 + 0 + 2 / 3) / 3 * 100, 6)


def test_compute_all_metrics_includes_peak_metrics():
    metrics = compute_all_metrics(np.array([1.0, 5.0, 2.0]), np.array([1.0, 4.5, 3.0]))

    assert {"RMSE", "MAE", "MAPE", "R2", "peak_hit_rate", "peak_time_offset", "peak_value_error"} <= set(metrics)


def test_compute_horizon_metrics_returns_one_group_per_step():
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[1.0, 1.0], [5.0, 4.0]])

    metrics = compute_horizon_metrics(y_true, y_pred)

    assert set(metrics) == {"H1_RMSE", "H1_MAE", "H1_MAPE", "H2_RMSE", "H2_MAE", "H2_MAPE"}
    assert metrics["H1_RMSE"] == np.sqrt(2.0)
    assert metrics["H2_MAE"] == 0.5
