# Enhanced package for advanced paint-by-numbers features

from .image_type_detector import ImageTypeDetector
from .settings_manager import SettingsManager
from .image_preprocessor import ImagePreprocessor
from .feature_detector import FeatureDetector
from .color_processor import ColorProcessor
from .number_placer import NumberPlacer
from .processor_pipeline import ProcessorPipeline
from .template_generator import TemplateGenerator
from .region_optimizer import RegionOptimizer
from .edge_enhancer import EdgeEnhancer
from .template_styler import TemplateStyler
from .integration import EnhancedIntegration, enhanced_integration

__all__ = [
    'ImageTypeDetector',
    'SettingsManager',
    'ImagePreprocessor',
    'FeatureDetector',
    'ColorProcessor',
    'NumberPlacer',
    'ProcessorPipeline',
    'TemplateGenerator',
    'RegionOptimizer',
    'EdgeEnhancer',
    'TemplateStyler',
    'EnhancedIntegration',
    'enhanced_integration'
]