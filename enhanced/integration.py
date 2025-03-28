import os
import time
import logging
import cv2
import numpy as np
from enhanced.processor_pipeline import ProcessorPipeline
from enhanced.settings_manager import SettingsManager
from enhanced.image_type_detector import ImageTypeDetector

logger = logging.getLogger('pbn-app.integration')

class EnhancedIntegration:
    """
    Integration layer that connects the UI with the enhanced processing pipeline.
    Provides methods for processing images, generating previews, and managing settings.
    """
    
    def __init__(self):
        """Initialize the integration layer"""
        self.settings_manager = SettingsManager()
        self.processor = ProcessorPipeline(self.settings_manager)
        self.image_type_detector = ImageTypeDetector()
        
    def process_image(self, image_path, custom_settings=None, force_type=None):
        """
        Process an image using the enhanced pipeline
        
        Args:
            image_path: Path to the image file
            custom_settings: Optional custom settings to apply
            force_type: Optional image type to force
            
        Returns:
            Dictionary with processing results
        """
        # Detect image type if not forced
        image_type = force_type
        if not image_type:
            image_type = self.image_type_detector.detect_image_type(image_path)
            
        # Get settings (default or custom)
        settings = None
        if custom_settings:
            settings = self.settings_manager.get_settings(image_type, custom_params=custom_settings)
        
        # Process the image
        start_time = time.time()
        results = self.processor.process_image(image_path, settings, image_type)
        
        # Add processing time and image type
        results['processing_time'] = time.time() - start_time
        results['image_type'] = image_type
        
        return results
        
    def generate_preview(self, image_path, settings, preview_type='template'):
        """
        Generate a preview based on current settings with caching to avoid repeated processing
        
        Args:
            image_path: Path to the image file
            settings: Settings to use for preview
            preview_type: Type of preview to generate ('template', 'segments', etc.)
            
        Returns:
            Preview image and metadata
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
            
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process based on preview type
        if preview_type == 'original':
            return {'preview': image, 'type': 'original'}
        
        # IMPORTANT: Respect user's explicit image type choice
        user_image_type = settings.get('image-type', 'auto')
        
        # Only auto-detect if user has chosen 'auto'
        if user_image_type == 'auto':
            image_type = self.image_type_detector.detect_image_type(image)
            logger.info(f"User selected 'auto', detected type: {image_type}")
        else:
            logger.info(f"Using user-selected image type: {user_image_type}")
            image_type = user_image_type
        
        # Get appropriate settings with the correct image type
        processed_settings = self.settings_manager.get_settings(image_type, custom_params=settings)
        
        # For preprocessed preview, we don't need full processing
        if preview_type == 'preprocessed':
            preprocessed = self.processor.preprocessor.preprocess(image, processed_settings)
            return {'preview': preprocessed, 'type': 'preprocessed'}
        
        # Create cache key that captures important settings including the ACTUAL image type used
        cache_key = f"{image_path}_{image_type}_{settings.get('colors', '0')}_{settings.get('edge_style', 'default')}"
        
        # Check if we should refresh cache
        force_refresh = settings.get('refresh_cache', False)
        
        # Initialize cache if needed
        if not hasattr(self, '_preview_cache'):
            self._preview_cache = {}
        
        # Use cached result if available, otherwise process
        if not force_refresh and cache_key in self._preview_cache:
            logger.debug(f"Using cached result for {cache_key}")
            result = self._preview_cache[cache_key]
        else:
            logger.debug(f"Processing new result for {cache_key}")
            # Process the image once and cache the results
            # FIXED: Removed the problematic force_type parameter
            result = self.processor.process_image(image_path, processed_settings)
            self._preview_cache[cache_key] = result
            
            # Limit cache size
            if len(self._preview_cache) > 5:
                oldest = next(iter(self._preview_cache))
                del self._preview_cache[oldest]
        
        # Return appropriate preview based on request with proper fallbacks
        if preview_type == 'segments':
            if 'segments' in result:
                return {'preview': result['segments'], 'type': 'segments'}
            elif 'colored_segments' in result:
                return {'preview': result['colored_segments'], 'type': 'segments'}
            else:
                logger.warning(f"No segments found in result! Available keys: {list(result.keys())}")
                return {'preview': image, 'type': 'segments'}
                
        elif preview_type == 'edges':
            if 'edges' in result:
                return {'preview': result['edges'], 'type': 'edges'}
            elif 'enhanced_edges' in result:
                return {'preview': result['enhanced_edges'], 'type': 'edges'}
            else:
                logger.warning(f"No edges found in result! Available keys: {list(result.keys())}")
                return {'preview': image, 'type': 'edges'}
                
        else:  # Default to template
            if 'template' in result:
                return {'preview': result['template'], 'type': 'template'}
            else:
                logger.warning(f"No template found in result! Available keys: {list(result.keys())}")
                return {'preview': image, 'type': 'template'}
    
    def get_parameter_metadata(self):
        """Get metadata for all parameters"""
        # Make sure this returns a valid metadata object
        return {
            'categories': self.settings_manager.parameter_metadata.get('categories', {}),
            'parameters': self.settings_manager.parameter_metadata.get('parameters', {}),
            'validation': self.settings_manager.parameter_metadata.get('validation', {})
        }
        
    def get_image_type_settings(self, image_type):
        """Get settings for a specific image type"""
        return self.settings_manager.get_settings(image_type)
        
    def get_available_presets(self):
        """Get list of available presets"""
        return self.settings_manager.get_available_presets()
        
    def load_preset(self, preset_name):
        """Load a preset by name"""
        presets = self.settings_manager.get_available_presets()
        for preset in presets:
            if preset.get('name') == preset_name:
                return preset.get('settings', {})
        return {}
        
    def save_preset(self, name, settings, description=""):
        """Save current settings as a preset"""
        return self.settings_manager.save_preset(name, settings, description)
        
    def analyze_settings(self, settings):
        """Analyze the current settings"""
        return self.settings_manager.analyze_settings_complexity(settings)

# Create the integration object for export
enhanced_integration = EnhancedIntegration()

# Legacy function for compatibility
def enhance_paint_by_numbers_generator(base_generator):
    """
    Legacy function to enhance the base generator
    
    Args:
        base_generator: The base PBN generator to enhance
        
    Returns:
        Enhanced generator
    """
    return enhanced_integration