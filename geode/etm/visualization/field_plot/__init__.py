"""
Field Plot - Cartopy-based velocity field visualization

This module provides components for plotting geodetic velocity/displacement fields
on maps with interactive field selection, zoom preservation, and station click handling.
"""

from .data_classes import ViewState, FieldPlotData, FieldPlotConfig
from .map_renderer import CartopyRenderer
from .widget_manager import WidgetManager
from .interaction_handler import InteractionHandler
from .field_visualizer import FieldVisualizer

__all__ = [
    'ViewState',
    'FieldPlotData',
    'FieldPlotConfig',
    'CartopyRenderer',
    'WidgetManager',
    'InteractionHandler',
    'FieldVisualizer',
]
