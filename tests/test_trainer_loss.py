import torch

from src.training.trainer import Trainer


def make_trainer(tmp_path, loss="mse", **training_overrides):
    config = {
        "training": {
            "device": "cpu",
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "log_dir": str(tmp_path / "logs"),
            "loss": loss,
            **training_overrides,
        }
    }
    return Trainer(torch.nn.Linear(2, 2), config, model_name="loss_test")


def test_peak_weighted_loss_emphasizes_high_targets(tmp_path):
    trainer = make_trainer(
        tmp_path,
        loss="peak_weighted_mse",
        peak_threshold=0.7,
        peak_weight=2.0,
    )
    predictions = torch.tensor([[0.0, 0.0]])
    targets = torch.tensor([[0.5, 0.8]])

    loss = trainer._compute_loss(predictions, targets)
    plain_mse = torch.mean((predictions - targets) ** 2)

    assert loss.item() > plain_mse.item()


def test_peak_trend_loss_adds_direction_penalty(tmp_path):
    trainer = make_trainer(
        tmp_path,
        loss="peak_trend_mse",
        peak_threshold=0.7,
        peak_weight=0.0,
        trend_weight=0.5,
    )
    predictions = torch.tensor([[0.7, 0.5, 0.3]])
    targets = torch.tensor([[0.3, 0.5, 0.7]])

    peak_trend_loss = trainer._compute_loss(predictions, targets)
    mse_only = torch.mean((predictions - targets) ** 2)

    assert peak_trend_loss.item() > mse_only.item()
