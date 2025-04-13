# Deepfake Detector Module
# This file marks the directory as a Python package

from .spatial import SpatialAnalyzer
from .temporal import TemporalAnalyzer
from .audio_visual import AudioVisualAnalyzer

__all__ = [
    'SpatialAnalyzer',
    'TemporalAnalyzer',
    'AudioVisualAnalyzer'
]