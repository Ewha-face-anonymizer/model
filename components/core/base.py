"""
Common abstractions and helper mixins for pipeline components.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class ImageProcessor(Protocol):
    """Protocol for processors that mutate numpy image buffers."""

    def apply(self, frame, bboxes):
        ...


@dataclass(frozen=True)
class ModelResource:
    """Describes a model that needs to be loaded from disk."""

    name: str
    path: Path
