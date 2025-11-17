"""
Utility helpers for loading typed application configuration from YAML.
위 YAML을 읽어서 AppConfig라는 dataclass로 변환합니다. 다른 모듈은 이걸 통해 안심하고 설정 값을 사용합니다
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ModelPaths:
    detector: str
    embedder: str


@dataclass
class ProcessingConfig:
    mosaic_kernel: int
    detection_min_size: int


@dataclass
class LoggingConfig:
    level: str
    save_frames: bool


@dataclass
class AppConfig:
    mosaic_threshold: float
    reference_image: str
    test_image: str
    input_dir: str
    output_dir: str
    webcam_id: int
    models: ModelPaths
    processing: ProcessingConfig
    logging: LoggingConfig


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config(path: Path) -> AppConfig:
    """Parse config.yaml and build a structured AppConfig instance."""
    raw = _load_yaml(path)
    return AppConfig(
        mosaic_threshold=float(raw["mosaic_threshold"]),
        reference_image=str(raw["reference_image"]),
        test_image=str(raw["test_image"]),
        input_dir=str(raw["input_dir"]),
        output_dir=str(raw["output_dir"]),
        webcam_id=int(raw["webcam_id"]),
        models=ModelPaths(**raw["models"]),
        processing=ProcessingConfig(**raw["processing"]),
        logging=LoggingConfig(**raw["logging"]),
    )
