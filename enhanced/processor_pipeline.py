import cv2
import numpy as np
import logging
import os
import time
from enhanced.image_preprocessor import ImagePreprocessor
from enhanced.feature_detector import FeatureDetector
from enhanced.color_processor import ColorProcessor
from enhanced.number_placer import NumberPlacer
from enhanced.settings_manager import SettingsManager
from enhanced.template_generator import TemplateGenerator
from enhanced.region_optimizer import RegionOptimizer
from enhanced.edge_enhancer import EdgeEnhancer
from enhanced.template_styler import TemplateStyler

logger = logging.getLogger('pbn-app.processor-pipeline')

class ProcessorPipeline:
    """
    Complete processing pipeline for paint-by-numbers generation,
    integrating all specialized modules
    """
    
    def __init__(self, settings_manager=None):
        """Initialize the processing pipeline"""
        logger.info("Initializing processing pipeline")
        
        # Initialize components
        self.preprocessor = ImagePreprocessor()
        self.feature_detector = FeatureDetector()
        self.color_processor = ColorProcessor()
        self.number_placer = NumberPlacer()
        
        # Initialize new template generation components
        self.template_generator = TemplateGenerator()
        self.region_optimizer = RegionOptimizer()
        self.edge_enhancer = EdgeEnhancer()
        self.template_styler = TemplateStyler()
        
        # Use provided settings manager or create a new one
        self.settings_manager = settings_manager if settings_manager else SettingsManager()
        
    def process_image(self, image_path, settings=None, image_type=None):
        """
        Process an image through the complete pipeline
        
        Args:
            image_path: Path to input image
            settings: Optional dictionary of settings (will use defaults if None)
            image_type: Optional image type (will auto-detect if None)
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        logger.info(f"Starting processing pipeline for {os.path.basename(image_path)}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return {"error": "Failed to load image"}
            
        # Convert to RGB (OpenCV loads as BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Step 1: Determine image type if not specified
        if image_type is None:
            try:
                # Assuming feature_detector has a detect_type method
                image_type = self.feature_detector.detect_type(image)
                logger.info(f"Auto-detected image type: {image_type}")
            except:
                # Fallback to generic type
                image_type = 'generic'
                logger.info("Using generic image type (auto-detection failed)")
        
        # Step 2: Load appropriate settings
        if settings is None:
            settings = self.settings_manager.get_settings(image_type)
            logger.info(f"Using default settings for {image_type}")
        
        # Step 3: Apply preprocessing
        try:
            processed_image = self.preprocessor.preprocess(image, settings)
            logger.info(f"Preprocessing completed in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}", exc_info=True)
            processed_image = image  # Use original as fallback
        
        # Step 4: Detect features if enabled
        feature_map = None
        feature_regions = []
        if settings.get('feature_detection', {}).get('feature_detection_enabled', True):
            try:
                feature_results = self.feature_detector.detect_features(processed_image, settings)
                feature_map = feature_results.get('importance_map')
                feature_regions = feature_results.get('features', [])
                logger.info(f"Feature detection completed in {time.time() - start_time:.2f}s")
            except Exception as e:
                logger.error(f"Error during feature detection: {e}", exc_info=True)
        
        # Steps 5-8: Generate template using our new components
        try:
            # Generate template using the template generator
            template_results = self.template_generator.generate_template(
                processed_image, settings, feature_map
            )
            
            segments = template_results['segments']
            edges = template_results['edges']
            colored_segments = template_results['colored_segments']
            palette = template_results['palette']
            segment_stats = template_results.get('segment_stats', {})
            
            # Optimize regions if needed
            optimized_segments = self.region_optimizer.optimize_regions(
                segments, settings, feature_map
            )
            
            # Enhance edges
            enhanced_edges = self.edge_enhancer.enhance_edges(
                optimized_segments, settings, feature_map
            )
            
            # Apply template styling
            template = self.template_styler.apply_style(
                colored_segments, enhanced_edges, optimized_segments, palette, settings
            )
            
            # Process colors
            color_results = self.color_processor.process_colors(
                processed_image, optimized_segments, settings, feature_map
            )
            colored_segments = color_results.get('colored_segments', colored_segments)
            palette = color_results.get('palette', palette)
            
            # Place numbers
            numbers_image, number_data = self.number_placer.place_numbers(
                optimized_segments, settings, feature_map, palette
            )
            
            logger.info(f"Template generation completed in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"Error during template generation: {e}", exc_info=True)
            # Fallback to a basic template
            template = np.zeros((processed_image.shape[0], processed_image.shape[1], 3), dtype=np.uint8)
            segments = np.zeros((processed_image.shape[0], processed_image.shape[1]), dtype=np.int32)
            optimized_segments = segments.copy()
            colored_segments = np.zeros_like(processed_image)
            palette = None
            edges = None
            enhanced_edges = None
            numbers_image = np.zeros((processed_image.shape[0], processed_image.shape[1], 4), dtype=np.uint8)
            number_data = []
            segment_stats = {}
        
        # Create final template with numbers
        try:
            # Convert template to RGBA
            template_rgba = cv2.cvtColor(template, cv2.COLOR_RGB2RGBA)
            
            # Add numbers using alpha compositing
            alpha = numbers_image[:, :, 3:] / 255.0
            template_with_numbers = template_rgba * (1 - alpha) + numbers_image * alpha
            
            logger.info(f"Final template creation completed in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"Error during final template creation: {e}", exc_info=True)
            template_with_numbers = template
        
        # Calculate total processing time
        elapsed = time.time() - start_time
        logger.info(f"Total processing time: {elapsed:.2f}s")
        
        # Prepare results
        results = {
            'processed_image': processed_image,
            'segments': segments,
            'optimized_segments': optimized_segments,
            'colored_segments': colored_segments,
            'feature_map': feature_map,
            'feature_regions': feature_regions,
            'edges': edges,
            'enhanced_edges': enhanced_edges,
            'numbers_image': numbers_image,
            'number_data': number_data,
            'template': template,
            'template_with_numbers': template_with_numbers,
            'palette': palette,
            'segment_stats': segment_stats,
            'processing_time': elapsed,
            'image_type': image_type,
        }
        
        return results