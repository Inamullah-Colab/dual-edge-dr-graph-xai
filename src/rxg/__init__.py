"""Retina XAI Graph DR package."""

__version__ = "0.1.0"

from .lesions import LESION_CLASSES, LesionClass
from .config import PipelineConfig

__all__ = ["LESION_CLASSES", "LesionClass", "PipelineConfig"]
