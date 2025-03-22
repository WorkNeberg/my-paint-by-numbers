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
        
        # Step 5: Segment image into regions
        try:
            # This would typically be where you'd apply your segmentation logic
            # For now, we'll assume it's part of the color processing
            segments = np.zeros(processed_image.shape[:2], dtype=np.int32)
            # In a real implementation, you'd apply your segmentation algorithm here
            
            # For now, we'll create dummy segments for testing
            h, w = processed_image.shape[:2]
            for y in range(h):
                for x in range(w):
                    # Create some simple regions for testing
                    segments[y, x] = (x // (w // 5)) + (y // (h // 5)) * 5 + 1
            
            logger.info(f"Segmentation completed in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"Error during segmentation: {e}", exc_info=True)
            return {"error": "Segmentation failed"}
        
        # Step 6: Process colors
        try:
            color_results = self.color_processor.process_colors(
                processed_image, segments, settings, feature_map
            )
            colored_segments = color_results.get('colored_segments')
            palette = color_results.get('palette')
            logger.info(f"Color processing completed in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"Error during color processing: {e}", exc_info=True)
            colored_segments = segments
            palette = None
        
        # Step 7: Place numbers
        try:
            numbers_image, number_data = self.number_placer.place_numbers(
                segments, settings, feature_map, palette
            )
            logger.info(f"Number placement completed in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"Error during number placement: {e}", exc_info=True)
            numbers_image = np.zeros((processed_image.shape[0], processed_image.shape[1], 4), dtype=np.uint8)
            number_data = []
        
        # Step 8: Create final template
        try:
            # Convert segments to RGB with borders
            template_rgb = np.zeros((processed_image.shape[0], processed_image.shape[1], 3), dtype=np.uint8)
            
            # Fill regions with colors from palette
            if palette is not None:
                unique_segments = np.unique(segments)
                for segment_id in unique_segments:
                    if segment_id == 0:  # Skip background
                        continue
                    mask = segments == segment_id
                    color_idx = segment_id - 1
                    if color_idx < len(palette):
                        template_rgb[mask] = palette[color_idx]
            
            # Add segment borders
            borders = self._create_segment_borders(segments)
            template_rgb[borders] = [0, 0, 0]  # Black borders
            
            # Add numbers
            template_rgba = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2RGBA)
            # Use alpha compositing to add numbers
            alpha = numbers_image[:, :, 3:] / 255.0
            template_rgba = template_rgba * (1 - alpha) + numbers_image * alpha
            
            logger.info(f"Template creation completed in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"Error during template creation: {e}", exc_info=True)
            template_rgba = cv2.cvtColor(processed_image, cv2.COLOR_RGB2RGBA)
        
        # Calculate total processing time
        elapsed = time.time() - start_time
        logger.info(f"Total processing time: {elapsed:.2f}s")
        
        # Prepare results
        results = {
            'processed_image': processed_image,
            'segments': segments,
            'colored_segments': colored_segments,
            'feature_map': feature_map,
            'feature_regions': feature_regions,
            'numbers_image': numbers_image,
            'number_data': number_data,
            'template': template_rgba,
            'palette': palette,
            'processing_time': elapsed,
            'image_type': image_type,
        }
        
        return results
        
    def _create_segment_borders(self, segments):
        """Create borders between segments"""
        # Create structuring element for dilation
        kernel = np.ones((3, 3), np.uint8)
        
        # Dilate each region to find borders
        borders = np.zeros_like(segments, dtype=bool)
        
        unique_segments = np.unique(segments)
        for segment_id in unique_segments:
            if segment_id == 0:  # Skip background
                continue
                
            # Create mask for this segment
            mask = segments == segment_id
            
            # Dilate mask
            dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
            
            # Border is dilated area minus original region
            border = dilated.astype(bool) & ~mask
            
            # Add to overall borders
            borders = borders | border
            
        return borders