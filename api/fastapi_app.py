"""
FastAPI application exposing HTTP endpoints for mosaic processing.
"""
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, UploadFile

from components.core.pipeline import MosaicPipeline
from config.loader import load_config

app = FastAPI(title="Selective Mosaic API", version="0.1.0")
CONFIG_PATH = Path("config/pipeline.yaml")
CONFIG = load_config(CONFIG_PATH)
PIPELINE = MosaicPipeline(CONFIG)


@app.post("/process-image")
async def process_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Accept an image upload, run the mosaic pipeline, and respond with metadata.
    The output image is saved to the configured `output_dir`.
    """
    contents = await file.read()
    input_dir = Path(CONFIG.input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / file.filename
    input_path.write_bytes(contents)

    PIPELINE.process_image(
        input_image=input_path,
        reference_image=Path(CONFIG.reference_image),
        output_dir=Path(CONFIG.output_dir),
    )
    return {
        "input_image": str(input_path),
        "output_dir": CONFIG.output_dir,
        "reference_image": CONFIG.reference_image,
    }
