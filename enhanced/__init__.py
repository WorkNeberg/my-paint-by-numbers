# Enhanced package for advanced paint-by-numbers features
from .enhanced_processor import EnhancedProcessor
from .feature_preserver import FeaturePreserver
from .image_type_detector import ImageTypeDetector
from .integration import enhance_paint_by_numbers_generator
from .settings_manager import SettingsManager

__all__ = [
    'EnhancedProcessor',
    'FeaturePreserver',
    'ImageTypeDetector',
    'enhance_paint_by_numbers_generator',
    'SettingsManager'
]