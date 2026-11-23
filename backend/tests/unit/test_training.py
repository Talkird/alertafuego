"""Unit tests for model/training: segmentation and fire-recall metrics, threshold
choice, BCE+Tversky loss, per-band normalization, the dataset's crop/shift helpers,
hour balancing and sun position. No training is run."""

import csv
from datetime import datetime, timezone

import numpy as np
import pytest
import torch

from model.data_pipeline.config import PATCH_FILL_VALUE
from model.training.config import pick_threshold
from model.training.dataset import FirePatchDataset, center_crop, shift
from model.training.losses import BCETverskyLoss, tversky_loss
from model.training.metrics import confusion_counts, f1, f_beta, fire_counts, iou, precision, recall, tolerant_counts
from model.training.normalization import (
    BandStats,
    compute_band_stats,
    load_band_stats,
    normalize_patch,
    save_band_stats,
)
from model.training.strata import hour_balance_weights, is_day, solar_zenith_deg

BIG = 20.0  # sigmoid(20) ~ 1, sigmoid(-20) ~ 0


def _logits(pred: list[list[int]]) -> torch.Tensor:
    return (torch.tensor(pred, dtype=torch.float32) * 2 - 1).mul(BIG)[None, None]


def _targets(mask: list[list[int]]) -> torch.Tensor:
    return torch.tensor(mask, dtype=torch.float32)[None, None]


def test_confusion_counts():
    logits = _logits([[1, 1], [0, 0]])
    targets = _targets([[1, 0], [1, 0]])
    tp, fp, fn, tn = (float(x) for x in confusion_counts(logits, targets, threshold=0.5))
    assert (tp, fp, fn, tn) == (1, 1, 1, 1)


def test_tolerant_counts_accept_a_prediction_one_pixel_off():
    logits = _logits([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    targets = _targets([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    strict_tp, _, _, _ = confusion_counts(logits, targets, 0.5)
    hit_preds, preds, hit_targets, positives = (float(x) for x in tolerant_counts(logits, targets, 0.5, radius=1))
    assert float(strict_tp) == 0
    assert (hit_preds, preds, hit_targets, positives) == (1, 1, 1, 1)


def test_tolerant_counts_reject_a_prediction_two_pixels_off():
    logits = _logits([[1, 0, 0]])
    targets = _targets([[0, 0, 1]])
    hit_preds, preds, hit_targets, positives = (float(x) for x in tolerant_counts(logits, targets, 0.5, radius=1))
    assert (hit_preds, hit_targets) == (0, 0)
    assert (preds, positives) == (1, 1)


def test_scalar_metrics():
    assert precision(3, 1) == pytest.approx(0.75)
    assert recall(3, 3) == pytest.approx(0.5)
    assert f1(0.75, 0.5) == pytest.approx(0.6)
    assert iou(3, 1, 2) == pytest.approx(0.5)


@pytest.mark.parametrize("metric,args", [(precision, (0, 0)), (recall, (0, 0)), (f1, (0, 0)), (iou, (0, 0, 0))])
def test_scalar_metrics_are_zero_without_division_by_zero(metric, args):
    assert metric(*args) == 0.0


def test_fire_counts_per_sample_recall_and_false_alarms():
    """One sample = one VIIRS fire: detected if any prediction is within 1 pixel."""
    targets = torch.zeros((3, 1, 3, 3))
    targets[0, 0, 1, 1] = 1  # fire, predicted one pixel off: detected
    targets[1, 0, 1, 1] = 1  # fire, no prediction: missed
    logits = torch.full((3, 1, 3, 3), -BIG)
    logits[0, 0, 1, 2] = BIG
    logits[2, 0, 0, 0] = BIG  # no fire, a prediction: false alarm
    detected, fires, alarms, free = (float(x) for x in fire_counts(logits, targets, threshold=0.5))
    assert (detected, fires, alarms, free) == (1, 2, 1, 1)


def test_f_beta_weights_recall():
    assert f_beta(0.5, 0.5, 2) == pytest.approx(0.5)
    assert f_beta(1.0, 0.5, 2) < f_beta(0.5, 1.0, 2)  # recall counts more
    assert f_beta(0.0, 0.0, 2) == 0.0


def test_pick_threshold_highest_reaching_target_else_best_recall():
    sweep = [
        {"threshold": 0.1, "fire_recall": 0.97},
        {"threshold": 0.3, "fire_recall": 0.95},
        {"threshold": 0.5, "fire_recall": 0.90},
    ]
    assert pick_threshold(sweep, 0.95)["threshold"] == 0.3
    assert pick_threshold(sweep, 0.99)["threshold"] == 0.1


def test_tversky_at_half_is_dice_and_perfect_on_all_background():
    """Half the dataset is all-background: predicting none of it must not be a 0/0."""
    logits = torch.full((2, 1, 4, 4), -BIG)
    targets = torch.zeros((2, 1, 4, 4))
    assert float(tversky_loss(logits, targets, fn_weight=0.5)) == pytest.approx(0.0, abs=1e-3)


def test_tversky_penalizes_a_miss_more_than_a_false_alarm():
    targets = torch.zeros((1, 1, 4, 4))
    targets[0, 0, 1, 1] = 1
    miss = torch.full((1, 1, 4, 4), -BIG)  # the fire pixel predicted as background
    false_alarm = torch.full((1, 1, 4, 4), -BIG)
    false_alarm[0, 0, 1, 1] = BIG
    false_alarm[0, 0, 2, 2] = BIG  # the fire found, plus one extra pixel
    assert float(tversky_loss(miss, targets, fn_weight=0.7)) > float(tversky_loss(false_alarm, targets, fn_weight=0.7))


def test_bce_tversky_loss_mixes_both_terms():
    logits = torch.randn(2, 1, 8, 8, generator=torch.Generator().manual_seed(0))
    targets = (torch.rand(2, 1, 8, 8, generator=torch.Generator().manual_seed(1)) > 0.8).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    tversky = tversky_loss(logits, targets, fn_weight=0.7)
    loss = BCETverskyLoss(tversky_weight=0.25, fn_weight=0.7)(logits, targets)
    assert float(loss) == pytest.approx(float(0.25 * tversky + 0.75 * bce), rel=1e-5)


def _save_patch(path, patch):
    np.savez(path, patch=patch.astype(np.float32), mask=np.zeros(patch.shape[1:], dtype=np.uint8))
    return path


def test_compute_band_stats_ignores_fill_pixels(tmp_path):
    a = np.array([[[1.0, 3.0], [PATCH_FILL_VALUE, PATCH_FILL_VALUE]], [[10.0, 10.0], [10.0, 10.0]]])
    b = np.array([[[5.0, PATCH_FILL_VALUE], [PATCH_FILL_VALUE, PATCH_FILL_VALUE]], [[20.0, 20.0], [20.0, 20.0]]])
    stats = compute_band_stats([_save_patch(tmp_path / "a.npz", a), _save_patch(tmp_path / "b.npz", b)])
    np.testing.assert_allclose(stats.mean, [3.0, 15.0])
    np.testing.assert_allclose(stats.std, [np.std([1, 3, 5]), 5.0], rtol=1e-5)


def test_compute_band_stats_constant_band_has_positive_std(tmp_path):
    patch = np.ones((1, 2, 2))
    stats = compute_band_stats([_save_patch(tmp_path / "a.npz", patch)])
    assert stats.std[0] > 0


def test_band_stats_round_trip(tmp_path):
    stats = BandStats(mean=np.array([1.5, 2.5], dtype=np.float32), std=np.array([0.5, 4.0], dtype=np.float32))
    path = tmp_path / "nested" / "norm_stats.json"
    save_band_stats(stats, path)
    loaded = load_band_stats(path)
    np.testing.assert_array_equal(loaded.mean, stats.mean)
    np.testing.assert_array_equal(loaded.std, stats.std)


def test_normalize_patch_zscores_and_zeroes_fill():
    stats = BandStats(mean=np.array([2.0], dtype=np.float32), std=np.array([2.0], dtype=np.float32))
    patch = np.array([[[4.0, PATCH_FILL_VALUE]]], dtype=np.float32)
    normalized = normalize_patch(patch, stats)
    np.testing.assert_array_equal(normalized, [[[1.0, 0.0]]])
    assert normalized.dtype == np.float32


def test_center_crop_2d_and_3d():
    array = np.arange(36).reshape(6, 6)
    np.testing.assert_array_equal(center_crop(array, 2), array[2:4, 2:4])
    stacked = np.stack([array, array])
    assert center_crop(stacked, (4, 2)).shape == (2, 4, 2)


def test_shift_moves_patch_and_mask_together():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1, 1] = 1
    patch = mask[None].astype(np.float32) * 7
    shifted_patch, shifted_mask = shift(patch, mask, dy=1, dx=2)
    assert shifted_mask[2, 3] == 1 and shifted_mask.sum() == 1
    assert shifted_patch[0, 2, 3] == 7 and shifted_patch.sum() == 7


def test_shift_fills_uncovered_border_with_zero():
    patch = np.ones((1, 3, 3), dtype=np.float32)
    mask = np.ones((3, 3), dtype=np.uint8)
    shifted_patch, shifted_mask = shift(patch, mask, dy=-1, dx=0)
    assert (shifted_patch[0, 2] == 0).all() and (shifted_mask[2] == 0).all()
    assert (shifted_patch[0, :2] == 1).all()


def _write_dataset(tmp_path, rows):
    """A manifest and one 33x33 sample per row: (filename, goes_time, has_fire, split)."""
    with (tmp_path / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "lat", "lon", "goes_time", "has_fire", "split"])
        writer.writeheader()
        for i, (name, goes_time, has_fire, split) in enumerate(rows):
            patch = np.arange(16 * 33 * 33, dtype=np.float32).reshape(16, 33, 33) + i
            np.savez(tmp_path / name, patch=patch, mask=np.ones((33, 33), dtype=np.uint8))
            writer.writerow({"filename": name, "lat": -31, "lon": -64, "goes_time": goes_time, "has_fire": has_fire, "split": split})
    return tmp_path / "manifest.csv"


def test_random_crop_adds_no_filled_border_and_is_fixed_for_val(tmp_path):
    manifest = _write_dataset(tmp_path, [("a.npz", "2025-08-01T17:00:00", "True", "val")])
    stats = BandStats(mean=np.zeros(16, dtype=np.float32), std=np.ones(16, dtype=np.float32))
    dataset = FirePatchDataset(manifest, tmp_path, "val", stats, crop_size=24, augment=False, random_crop=True)
    patch, mask = dataset[0]
    assert patch.shape == (16, 24, 24) and mask.shape == (24, 24)
    assert bool((mask == 1).all())  # every pixel is a real one: nothing filled in
    again, _ = dataset[0]
    assert torch.equal(patch, again)


def test_hour_balance_weights_even_out_fire_and_fire_free_per_hour(tmp_path):
    rows = [(f"f{i}.npz", "2025-08-01T17:00:00", "True", "train") for i in range(3)]
    rows += [("n0.npz", "2025-08-01T17:00:00", "False", "train"), ("n1.npz", "2025-08-01T12:00:00", "False", "train")]
    manifest = _write_dataset(tmp_path, rows)
    files = [tmp_path / name for name, *_ in rows]
    weights = hour_balance_weights(manifest, tmp_path, files)
    # 17 UTC: 3 fires and 1 fire-free sample get the same total weight (2 each).
    assert weights[:3] == pytest.approx([2 / 3] * 3) and weights[3] == pytest.approx(2.0)
    assert weights[4] == 1.0  # 12 UTC has fire-free samples only: nothing to balance
    capped = hour_balance_weights(manifest, tmp_path, files, max_weight=1.5)
    assert capped[3] == 1.5


def test_solar_zenith_and_day_night():
    # Córdoba at local noon (~16:15 UTC) near the June solstice: the sun ~55 degrees up.
    noon = datetime(2025, 6, 21, 16, 15, tzinfo=timezone.utc)
    assert solar_zenith_deg(-31.4, -64.2, noon) == pytest.approx(54.8, abs=1.0)
    assert bool(is_day(-31.4, -64.2, noon))
    assert not bool(is_day(-31.4, -64.2, datetime(2025, 6, 21, 4, 15, tzinfo=timezone.utc)))
    lat = np.array([-31.4, -31.4])
    assert is_day(lat, np.array([-64.2, 115.8]), noon).tolist() == [True, False]  # other side of the planet
