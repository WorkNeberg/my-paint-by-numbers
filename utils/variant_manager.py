import os
import time
import cv2
import numpy as np
import logging
from utils.parameter_manager import ParameterManager

logger = logging.getLogger('pbn-app.variant_manager')

class VariantManager:
    """Manages the creation and comparison of processing variants"""
    
    def __init__(self, enhanced_processor, parameter_manager=None):
        """Initialize the variant manager"""
        self.enhanced_processor = enhanced_processor
        self.parameter_manager = parameter_manager or ParameterManager()
        
    def generate_variants(self, image_path, image_type, output_dir=None, variants=None):
        """
        Generate multiple processing variants of the same image.
        
        Args:
            image_path: Path to the input image
            image_type: Type of image (pet, portrait, landscape)
            output_dir: Directory to save the variants
            variants: List of variant names to generate (or None for all)
            
        Returns:
            Dictionary with paths to generated variants and metadata
        """
        logger.info(f"Generating variants for {image_path} ({image_type})")
        
        # Create output directory if needed
        if output_dir is None:
            timestamp = int(time.time())
            base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
            output_dir = os.path.join(base_dir, 'static', 'variants', f'comparison_{timestamp}')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to RGB for processing
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get available variants for this image type
        available_variants = self.parameter_manager.get_available_variants(image_type)
        
        # If no specific variants requested, use all available
        if variants is None:
            variants = list(available_variants.keys())
        
        # Process each variant
        results = []
        
        # Always include standard processing as a baseline
        if "standard" not in variants:
            variants.insert(0, "standard")
        
        for variant_name in variants:
            if variant_name not in available_variants and variant_name != "standard":
                logger.warning(f"Variant '{variant_name}' not found for {image_type}")
                continue
                
            logger.info(f"Processing variant: {variant_name}")
            start_time = time.time()
            
            # Get settings for this variant
            settings = self.parameter_manager.get_settings(image_type, variant_name)
            
            # Define output path
            output_path = os.path.join(output_dir, f"{variant_name}.png")
            
            try:
                # Process the image
                result = self.enhanced_processor.process_image(image.copy(), settings)
                
                # Get the output image
                output_image = None
                if 'segmented_image' in result:
                    output_image = result['segmented_image']
                elif 'processed_image' in result:
                    output_image = result['processed_image']
                elif 'output_image' in result:
                    output_image = result['output_image']
                
                # Save the output image
                if output_image is not None:
                    cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
                else:
                    logger.warning(f"No output image found for variant: {variant_name}")
                    # Create a placeholder
                    placeholder = np.ones((400, 400, 3), dtype=np.uint8) * 200
                    cv2.putText(placeholder, f"No output for {variant_name}", (50, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    cv2.imwrite(output_path, placeholder)
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Get metadata
                description = available_variants.get(variant_name, "Standard processing")
                
                # Store result info
                results.append({
                    "name": variant_name,
                    "description": description,
                    "path": output_path.replace('\\', '/'),  # Web-friendly path
                    "processing_time": f"{processing_time:.2f}s",
                    "settings": settings,
                    "metrics": self._calculate_metrics(result)
                })
                
            except Exception as e:
                logger.error(f"Error processing variant '{variant_name}': {e}", exc_info=True)
                
        # Save original image for comparison
        original_path = os.path.join(output_dir, "original.png")
        cv2.imwrite(original_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        return {
            "variants": results,
            "original": original_path.replace('\\', '/'),
            "image_type": image_type
        }
    
    def _calculate_metrics(self, result):
        """Calculate metrics for a processed image"""
        metrics = {}
        
        # Number of regions
        if 'region_count' in result:
            metrics['region_count'] = result['region_count']
        elif 'regions' in result:
            metrics['region_count'] = len(result['regions'])
        
        # Number of colors
        if 'colors' in result:
            metrics['color_count'] = len(result['colors'])
        
        # Paintability score if available
        if 'paintability' in result:
            metrics['paintability'] = result['paintability']
            
        return metrics