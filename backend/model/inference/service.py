"""Orchestrates a single detection run: fetch latest image, predict, threshold.

The model in cfg.checkpoint_dir always runs. Each folder in its ensemble/
subfolder holding a model_best.pt is one more model, run on the same raster; a
pixel is a detection if any model finds it (ensemble.detection_score)."""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import ee
import numpy as np
import torch
from shapely.geometry.base import BaseGeometry

from model.data_pipeline.config import BBox
from model.data_pipeline.ee_client import init_earth_engine
from model.inference.clusters import clusters
from model.inference.config import InferenceConfig, default_config
from model.inference.ensemble import SCORE_THRESHOLD, detection_score
from model.inference.land_filter import filter_to_polygon, load_argentina_polygon
from model.inference.predictor import load_model, predict_raster
from model.inference.raster_fetch import fetch_calibrated_chunks, get_latest_goes_image, image_capture_time
from model.inference.tiling import assemble_raster, pixel_to_latlon
from model.inference.visualize import render_detection_thumbnail, save_debug_image, save_true_color_image
from model.training.config import ModelConfig, load_model_config
from model.training.normalization import BandStats, load_band_stats
from model.training.strata import is_day

logger = logging.getLogger(__name__)


#: Subfolder of InferenceConfig.checkpoint_dir with the ensemble's other models.
ENSEMBLE_DIRNAME = "ensemble"


@dataclass(frozen=True)
class LoadedModel:
    name: str
    model: torch.nn.Module
    band_stats: BandStats
    #: Its day threshold, and night threshold if any, chosen on val scenes.
    config: ModelConfig


@dataclass(frozen=True)
class InferenceContext:
    models: list[LoadedModel]
    device: torch.device
    cfg: InferenceConfig
    argentina_polygon: BaseGeometry


@dataclass(frozen=True)
class DetectionResult:
    image_time: datetime
    bbox: BBox
    #: The score a detection must reach (ensemble.SCORE_THRESHOLD).
    threshold: float
    chunk_count: int
    #: (lat, lon, score, threshold, thumbnail_png). The score is ensemble.detection_score,
    #: not a raw model probability: SCORE_THRESHOLD at the threshold of the model surest
    #: of the pixel, rising as it is surer.
    detections: list[tuple[float, float, float, float, bytes]]


def checkpoint_dirs(checkpoint_dir: Path) -> list[Path]:
    """checkpoint_dir, then each model folder in its ensemble/ subfolder, by name."""
    ensemble_dir = checkpoint_dir / ENSEMBLE_DIRNAME
    extra = sorted(d for d in ensemble_dir.iterdir() if (d / "model_best.pt").exists()) if ensemble_dir.is_dir() else []
    return [checkpoint_dir, *extra]


def load_models(checkpoint_dir: Path, device: torch.device) -> list[LoadedModel]:
    return [
        LoadedModel(
            name=d.name,
            model=load_model(d / "model_best.pt", device),
            band_stats=load_band_stats(d / "norm_stats.json"),
            config=load_model_config(d),
        )
        for d in checkpoint_dirs(checkpoint_dir)
    ]


def threshold_map(config: ModelConfig, night: np.ndarray | None) -> np.ndarray | float:
    """config's threshold per pixel: its night threshold where night is True, if it has one."""
    if night is None or config.night_threshold is None:
        return config.threshold
    return np.where(night, config.night_threshold, config.threshold).astype(np.float32)


def load_context(cfg: InferenceConfig | None = None) -> InferenceContext:
    """One-time startup routine: Earth Engine auth, normalization stats, model weights."""
    cfg = cfg if cfg is not None else default_config()
    init_earth_engine()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(cfg.checkpoint_dir, device)
    argentina_polygon = load_argentina_polygon(cfg.argentina_bbox)
    logger.info(
        "Inference context loaded on device %s: %s",
        device,
        ", ".join(f"{m.name} (day {m.config.threshold}, night {m.config.night_threshold})" for m in models),
    )
    return InferenceContext(models=models, device=device, cfg=cfg, argentina_polygon=argentina_polygon)


def run_detection(
    ctx: InferenceContext,
    bbox: BBox,
    debug_image_dir: Path | None = None,
    goes_image: ee.Image | None = None,
) -> DetectionResult:
    """Detects fires in goes_image (default: the latest GOES-19 capture) over bbox
    with every model in ctx, each at its own thresholds (its night threshold where
    the sun is below the horizon)."""
    if goes_image is None:
        goes_image = get_latest_goes_image()
    image_time = image_capture_time(goes_image)

    chunks = fetch_calibrated_chunks(goes_image, bbox, ctx.cfg.chunk_size_px)
    raster = assemble_raster(chunks, bbox)
    _, height, width = raster.shape
    night = None
    if any(m.config.night_threshold is not None for m in ctx.models):
        lat = bbox.north - (np.arange(height) + 0.5) / height * (bbox.north - bbox.south)
        lon = bbox.west + (np.arange(width) + 0.5) / width * (bbox.east - bbox.west)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        night = ~is_day(lat_grid, lon_grid, image_time)
    scores = detection_score(
        [predict_raster(m.model, m.band_stats, raster, ctx.device, tta=ctx.cfg.tta) for m in ctx.models],
        [threshold_map(m.config, night) for m in ctx.models],
    )

    fire = scores >= SCORE_THRESHOLD
    if ctx.cfg.group_fires:
        pixels = sorted(max(group, key=lambda p: scores[p]) for group in clusters(fire))
    else:
        pixels = list(zip(*fire.nonzero()))

    detections: list[tuple[float, float, float, int, int]] = []  # lat, lon, score, row, col
    for row, col in pixels:
        row, col = int(row), int(col)
        lat, lon = pixel_to_latlon(row, col, bbox, (height, width))
        detections.append((lat, lon, float(scores[row, col]), row, col))

    if debug_image_dir is not None:
        pixel_detections = [(row, col, score) for _, _, score, row, col in detections]
        save_debug_image(raster, pixel_detections, image_time, SCORE_THRESHOLD, debug_image_dir)
        save_true_color_image(raster, image_time, debug_image_dir)

    detections = filter_to_polygon(detections, ctx.argentina_polygon)
    detections_with_thumbnails = [
        (lat, lon, score, SCORE_THRESHOLD, render_detection_thumbnail(raster, row, col))
        for lat, lon, score, row, col in detections
    ]

    logger.info("Detection run: %d chunks, %d detections", len(chunks), len(detections))
    return DetectionResult(
        image_time=image_time,
        bbox=bbox,
        threshold=SCORE_THRESHOLD,
        chunk_count=len(chunks),
        detections=detections_with_thumbnails,
    )
