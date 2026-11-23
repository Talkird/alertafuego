"""Unit tests for model/inference: raster assembly, pixel geolocation, the Argentina
border filter, whole-raster prediction, the ensemble's detection score, grouping and
a detection run with two models. No Earth Engine: fetching and the projection are
stubbed."""

from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest
import torch
from shapely.geometry import box

from model.data_pipeline.config import PATCH_FILL_VALUE, BBox
from model.inference import predictor, service, tiling
from model.inference.clusters import clusters
from model.inference.config import default_config
from model.inference.ensemble import SCORE_THRESHOLD, detection_score
from model.inference.land_filter import filter_to_polygon
from model.training.config import ModelConfig
from model.training.model import TinyUNet
from model.training.normalization import BandStats


@pytest.fixture
def one_degree_per_pixel(monkeypatch):
    """1 degree per pixel, so bbox arithmetic reads as pixel arithmetic."""
    monkeypatch.setattr(tiling, "target_projection", lambda: None)
    monkeypatch.setattr(tiling, "degrees_per_pixel", lambda projection: (1.0, 1.0))


def test_pixel_to_latlon_returns_pixel_centers():
    bbox = BBox(west=-70.0, south=-40.0, east=-60.0, north=-30.0)
    assert tiling.pixel_to_latlon(0, 0, bbox, (10, 10)) == pytest.approx((-30.5, -69.5))
    assert tiling.pixel_to_latlon(9, 9, bbox, (10, 10)) == pytest.approx((-39.5, -60.5))


def test_assemble_raster_places_chunks_by_position_north_up(one_degree_per_pixel):
    bbox = BBox(west=0.0, south=0.0, east=4.0, north=2.0)
    north_west = (BBox(0.0, 1.0, 2.0, 2.0), np.full((1, 1, 2), 1.0, dtype=np.float32))
    south_east = (BBox(2.0, 0.0, 4.0, 1.0), np.full((1, 1, 2), 2.0, dtype=np.float32))

    canvas = tiling.assemble_raster([south_east, north_west], bbox)

    expected = np.array([[[1, 1, PATCH_FILL_VALUE, PATCH_FILL_VALUE], [PATCH_FILL_VALUE, PATCH_FILL_VALUE, 2, 2]]])
    np.testing.assert_array_equal(canvas, expected)


def test_assemble_raster_clips_chunks_overflowing_the_bbox(one_degree_per_pixel):
    bbox = BBox(west=0.0, south=0.0, east=2.0, north=2.0)
    # Chunks fetched from Earth Engine can come back one pixel larger than asked.
    oversized = (BBox(0.0, 0.0, 2.0, 2.0), np.ones((1, 3, 3), dtype=np.float32))
    canvas = tiling.assemble_raster([oversized], bbox)
    assert canvas.shape == (1, 2, 2)
    assert (canvas == 1).all()


def test_filter_to_polygon_keeps_points_inside_and_passes_extra_fields():
    polygon = box(-70, -40, -60, -30)  # lon/lat
    inside = (-35.0, -65.0, 0.9, b"png")
    outside = (-25.0, -57.0, 0.8, b"png")  # Paraguay-ish
    assert filter_to_polygon([inside, outside], polygon) == [inside]


def test_filter_to_polygon_reads_lat_lon_order():
    polygon = box(-70, -40, -60, -30)
    # (lat, lon): swapped coordinates would land outside.
    assert filter_to_polygon([(-35.0, -65.0)], polygon) == [(-35.0, -65.0)]
    assert filter_to_polygon([(-65.0, -35.0)], polygon) == []


STATS = BandStats(mean=np.zeros(16, dtype=np.float32), std=np.ones(16, dtype=np.float32))


def _model() -> TinyUNet:
    torch.manual_seed(0)
    return TinyUNet(in_channels=16, base_channels=4, depth=2).eval()


def test_predict_raster_in_blocks_matches_one_pass(monkeypatch):
    """Production splits the country into blocks with a context margin (memory); the
    result must not depend on it."""
    raster = np.random.default_rng(0).normal(size=(16, 70, 90)).astype(np.float32)
    model = _model()
    whole = predictor.predict_raster(model, STATS, raster, torch.device("cpu"))
    monkeypatch.setattr(predictor, "BLOCK_SIZE", 32)
    monkeypatch.setattr(predictor, "BLOCK_MARGIN", 16)
    blocks = predictor.predict_raster(model, STATS, raster, torch.device("cpu"))
    assert whole.shape == (70, 90)
    np.testing.assert_allclose(blocks, whole, atol=1e-5)


def test_predict_raster_tta_is_orientation_independent():
    raster = np.random.default_rng(1).normal(size=(16, 32, 32)).astype(np.float32)
    model = _model()
    plain = predictor.predict_raster(model, STATS, raster, torch.device("cpu"), tta=True)
    flipped = predictor.predict_raster(model, STATS, raster[:, :, ::-1].copy(), torch.device("cpu"), tta=True)
    np.testing.assert_allclose(flipped[:, ::-1], plain, atol=1e-5)


def test_detection_score_is_half_at_each_models_threshold():
    """b-shift detects at 0.005 and d-noleak at 0.995: each scores 0.5 exactly there."""
    score = detection_score([np.array([0.005, 0.001]), np.array([0.5, 0.995])], [0.005, 0.995])
    np.testing.assert_allclose(score, [0.5, 0.5], atol=1e-6)


def test_detection_score_takes_the_surest_model_and_keeps_one_models_order():
    probabilities = [np.array([0.05, 0.001, 0.001]), np.array([0.9, 0.9, 0.999])]
    score = detection_score(probabilities, [0.005, 0.995])
    assert score[0] > SCORE_THRESHOLD  # b-shift well above its threshold, d-noleak below
    assert score[1] < SCORE_THRESHOLD  # neither reaches its threshold
    assert score[2] > SCORE_THRESHOLD  # d-noleak above its threshold
    single = detection_score([np.array([0.1, 0.3, 0.2])], [0.08])
    assert list(np.argsort(single)) == [0, 2, 1]


def test_threshold_map_uses_the_night_threshold_where_night():
    night = np.array([[True, False]])
    np.testing.assert_allclose(service.threshold_map(ModelConfig(16, 3, 0.005, 0.08), night), [[0.08, 0.005]])
    assert service.threshold_map(ModelConfig(16, 3, 0.995), night) == 0.995


def test_clusters_group_8_connected_pixels():
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 0] = mask[1, 1] = True  # diagonal neighbours: one fire
    mask[3, 3] = mask[3, 4] = True
    mask[0, 4] = True
    groups = sorted(sorted(map(tuple, g)) for g in clusters(mask))
    assert groups == [[(0, 0), (1, 1)], [(0, 4)], [(3, 3), (3, 4)]]


def test_checkpoint_dirs_main_model_then_ensemble(tmp_path):
    (tmp_path / "ensemble" / "d-noleak").mkdir(parents=True)
    (tmp_path / "ensemble" / "d-noleak" / "model_best.pt").touch()
    (tmp_path / "ensemble" / "empty").mkdir()  # no weights: ignored
    assert service.checkpoint_dirs(tmp_path) == [tmp_path, tmp_path / "ensemble" / "d-noleak"]
    assert service.checkpoint_dirs(tmp_path / "ensemble" / "d-noleak") == [tmp_path / "ensemble" / "d-noleak"]


@pytest.fixture
def two_model_run(monkeypatch):
    """run_detection over a 4x4 bbox with two fake models and no Earth Engine."""
    bbox = BBox(west=-65.0, south=-33.0, east=-63.0, north=-31.0)
    raster = np.zeros((16, 4, 4), dtype=np.float32)
    a = np.full((4, 4), 0.001, dtype=np.float32)
    a[0, 0], a[0, 1] = 0.5, 0.05  # model A (threshold 0.005): one fire, two pixels
    b = np.full((4, 4), 0.5, dtype=np.float32)
    b[3, 3] = 0.999  # model B (threshold 0.995): another fire
    probabilities = {"a": a, "b": b}
    models = [
        service.LoadedModel("a", "a", STATS, ModelConfig(16, 3, 0.005)),
        service.LoadedModel("b", "b", STATS, ModelConfig(16, 3, 0.995)),
    ]
    monkeypatch.setattr(service, "fetch_calibrated_chunks", lambda image, bbox, size: [None])
    monkeypatch.setattr(service, "assemble_raster", lambda chunks, bbox: raster)
    monkeypatch.setattr(service, "image_capture_time", lambda image: datetime(2025, 8, 28, 18, 0, tzinfo=timezone.utc))
    monkeypatch.setattr(service, "predict_raster", lambda model, stats, raster, device, tta=False: probabilities[model])
    monkeypatch.setattr(service, "render_detection_thumbnail", lambda raster, row, col: b"png")

    def run(group_fires: bool):
        cfg = replace(default_config(), group_fires=group_fires)
        ctx = service.InferenceContext(models, torch.device("cpu"), cfg, box(-70, -40, -60, -30))
        return service.run_detection(ctx, bbox, goes_image=object())

    return run


def test_run_detection_reports_what_either_model_finds(two_model_run):
    result = two_model_run(group_fires=False)
    assert result.threshold == SCORE_THRESHOLD
    assert len(result.detections) == 3
    assert all(d[2] >= SCORE_THRESHOLD and d[3] == SCORE_THRESHOLD and d[4] == b"png" for d in result.detections)


def test_run_detection_groups_a_fire_into_its_surest_pixel(two_model_run):
    result = two_model_run(group_fires=True)
    lat_lon = sorted((round(lat, 2), round(lon, 2)) for lat, lon, *_ in result.detections)
    # Pixel (0, 0) for model A's two-pixel fire, and (3, 3) for model B's.
    assert lat_lon == [(-32.75, -63.25), (-31.25, -64.75)]
