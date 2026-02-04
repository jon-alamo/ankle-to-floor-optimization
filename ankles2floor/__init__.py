"""
ankles2floor - Ankle-to-floor classification for padel players.

This package provides tools to classify ankle movements from COCO17 pose detection
coordinates to determine when a player's foot is grounded.
"""

from ankles2floor.core import get_ankles_in_floor
from ankles2floor.config import best_parameters

__version__ = "1.0.0"
__all__ = ["get_ankles_in_floor", "best_parameters"]
