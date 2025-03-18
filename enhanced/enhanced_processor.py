import cv2
import numpy as np
from image_processor import ImageProcessor
from feature_preserver import FeaturePreserver
from image_type_detector import ImageTypeDetector
from settings_manager import SettingsManager
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class EnhancedProcessor:
    def __init__(self):
        self.base_processor = ImageProcessor()
        self.feature_preserver = FeaturePreserver()
        self.type_detector = ImageTypeDetector()
        self.settings_manager = SettingsManager()
        
    def process_with_feature_preservation(self, image, settings, image_type):
        """
        Process image with special handling for important features
        
        Parameters:
        - image: Input RGB image
        - settings: Dictionary of processing settings
        - image_type: Type of image for specialized processing
        
        Returns:
        - result: Dictionary with processing results
        """
        h, w = image.shape[:2]
        
        # Step 1: Create feature mask if feature preservation is enabled
        feature_mask = None
        feature_regions = None
        if settings.get('preserve_features', False):
            feature_mask, feature_regions = self.feature_preserver.create_feature_mask(image, image_type)
            
            # Debug: Save feature mask
            if feature_mask is not None:
                plt.imsave('debug_feature_mask.png', feature_mask, cmap='gray')
        
        # Step 2: Apply preprocessing and enhancements
        processed_image = self.preprocess_image(image, settings, image_type)
        
        # Step 3: Process with or without feature preservation
        if feature_mask is not None and np.any(feature_mask > 0):
            return self.process_with_mask(processed_image, feature_mask, feature_regions, settings)
        else:
            # Use regular processing if no features to preserve or feature preservation disabled
            return self.process_regular(processed_image, settings)
    
    def preprocess_image(self, image, settings, image_type):
        """Apply specialized preprocessing based on image type"""
        # Clone image to avoid modifying original
        processed = image.copy()
        
        # Apply dark area enhancement if enabled
        if settings.get('enhance_dark_areas', False):
            processed = self.base_processor._enhance_dark_areas(
                processed, 
                settings.get('dark_threshold', 50)
            )
            
        # Apply specialized preprocessing based on image type
        if image_type == 'portrait':
            # For portraits, slightly enhance skin tones
            # Convert to YCrCb
            ycrcb = cv2.cvtColor(processed, cv2.COLOR_RGB2YCrCb)
            
            # Create a mask for potential skin
            lower = np.array([0, 133, 77], dtype=np.uint8)
            upper = np.array([255, 173, 127], dtype=np.uint8)
            skin_mask = cv2.inRange(ycrcb, lower, upper)
            
            # Apply slight smoothing to skin areas
            if np.any(skin_mask):
                skin_area = cv2.bitwise_and(processed, processed, mask=skin_mask)
                smoothed_skin = cv2.bilateralFilter(skin_area, 9, 75, 75)
                
                # Blend back with original
                skin_mask_3ch = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2RGB) / 255
                processed = processed * (1 - skin_mask_3ch) + smoothed_skin * skin_mask_3ch
                
        elif image_type == 'pet':
            # For pets, use bilateral filtering to smooth fur while preserving edges
            processed = cv2.bilateralFilter(processed, 9, 100, 100)
            
        return processed
    
    def process_with_mask(self, image, feature_mask, feature_regions, settings):
        """
        Process image with different parameters for feature and non-feature regions
        
        This enables higher detail in important areas like eyes, and more
        merging/simplification in background areas.
        """
        h, w = image.shape[:2]
        
        # Create enhanced settings for feature regions
        feature_settings = settings.copy()
        feature_boost = settings.get('feature_detail_boost', 1.5)
        
        # Adjust feature region settings for more detail
        feature_settings['num_colors'] = int(settings.get('num_colors', 15) * feature_boost)
        feature_settings['simplification_level'] = 'low'  # Always use high detail for features
        feature_settings['edge_strength'] = settings.get('edge_strength', 1.0) * 0.8
        feature_settings['merge_regions_level'] = 'low'  # Less merging in features
        feature_settings['min_region_percent'] = settings.get('min_region_percent', 0.5) / 2
        
        # Create background settings for less detail
        bg_settings = settings.copy()
        
        # Define feature and background regions
        feature_mask_binary = feature_mask > 128
        bg_mask = ~feature_mask_binary
        
        # Extract regions
        feature_img = image.copy()
        bg_img = image.copy()
        
        # Process feature regions with higher detail
        print("Processing feature regions with enhanced detail...")
        vectorized_features, label_features, edges_features, centers_features, region_data_features, _ = (
            self.base_processor.process_image(
                feature_img,
                feature_settings['num_colors'],
                feature_settings.get('simplification_level', 'low'),
                feature_settings.get('edge_strength', 0.8),
                feature_settings.get('edge_width', 1),
                feature_settings.get('enhance_dark_areas', False),
                feature_settings.get('dark_threshold', 50),
                feature_settings.get('edge_style', 'soft'),
                feature_settings.get('merge_regions_level', 'low'),
                'feature'
            )
        )
        
        # Process background regions with normal settings
        print("Processing background regions...")
        vectorized_bg, label_bg, edges_bg, centers_bg, region_data_bg, paintability = (
            self.base_processor.process_image(
                bg_img,
                bg_settings.get('num_colors', 15),
                bg_settings.get('simplification_level', 'medium'),
                bg_settings.get('edge_strength', 1.0),
                bg_settings.get('edge_width', 1),
                bg_settings.get('enhance_dark_areas', False),
                bg_settings.get('dark_threshold', 50),
                bg_settings.get('edge_style', 'normal'),
                bg_settings.get('merge_regions_level', 'normal'),
                'background'
            )
        )
        
        # Combine results
        combined_vectorized = np.zeros_like(image)
        combined_label = np.zeros((h, w), dtype=np.int32)
        combined_edges = np.zeros((h, w), dtype=np.uint8)
        
        # Create binary masks
        feature_mask_3d = np.stack([feature_mask_binary] * 3, axis=2)
        
        # Use feature results for feature regions, background results for bg regions
        combined_vectorized = np.where(feature_mask_3d, vectorized_features, vectorized_bg)
        combined_edges = np.maximum(edges_features * feature_mask_binary, edges_bg * bg_mask)
        
        # Combine label images (needs offset for feature labels to avoid duplication)
        label_offset = np.max(label_bg) + 1
        combined_label = label_bg.copy()
        combined_label[feature_mask_binary] = label_features[feature_mask_binary] + label_offset
        
        # Combine region data
        # Adjust feature region IDs to account for offset
        for region in region_data_features:
            region['id'] += label_offset
            
        combined_region_data = region_data_bg + region_data_features
        
        # Combine color centers
        combined_centers = np.vstack([centers_bg, centers_features])
        
        return {
            'vectorized': combined_vectorized,
            'label_image': combined_label,
            'edges': combined_edges,
            'centers': combined_centers,
            'region_data': combined_region_data,
            'paintability': paintability,
            'feature_regions': feature_regions
        }
        
    def process_regular(self, image, settings):
        """Process the entire image with uniform settings"""
        vectorized, label_image, edges, centers, region_data, paintability = (
            self.base_processor.process_image(
                image,
                settings.get('num_colors', 15),
                settings.get('simplification_level', 'medium'),
                settings.get('edge_strength', 1.0),
                settings.get('edge_width', 1),
                settings.get('enhance_dark_areas', False),
                settings.get('dark_threshold', 50),
                settings.get('edge_style', 'normal'),
                settings.get('merge_regions_level', 'normal'),
                'general'
            )
        )
        
        return {
            'vectorized': vectorized,
            'label_image': label_image,
            'edges': edges,
            'centers': centers,
            'region_data': region_data,
            'paintability': paintability,
            'feature_regions': None
        }
        
    def create_comparison_variants(self, image, output_dir='variants'):
        """
        Create multiple variants of the processed image for comparison
        
        Parameters:
        - image: Input RGB image
        - output_dir: Directory to save variant images
        
        Returns:
        - variants: Dictionary of variant results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Detect image type
        image_type, confidence = self.type_detector.detect_type_simple(image)
        print(f"Detected image type: {image_type} (confidence: {confidence:.2f})")
        
        variants = {}
        
        # Get all presets for this image type
        settings_dict = self.settings_manager.presets.get(image_type, self.settings_manager.presets['general'])
        
        # Process with each preset
        for preset_name, settings in settings_dict.items():
            print(f"Processing with {preset_name} preset for {image_type}...")
            
            # Create a suitable display name
            style_name = settings.get('style_name', preset_name.capitalize())
            variant_name = f"{image_type}_{style_name}"
            
            # Process the image
            result = self.process_with_feature_preservation(image, settings, image_type)
            
            # Save the processed image
            output_path = os.path.join(output_dir, f"{variant_name}.png")
            plt.imsave(output_path, result['vectorized'])
            
            # Also save template
            template = self.create_template(result, style_name)
            template_path = os.path.join(output_dir, f"{variant_name}_template.png")
            plt.imsave(template_path, template)
            
            # Store result
            variants[variant_name] = {
                'settings': settings,
                'result': result,
                'output_path': output_path,
                'template_path': template_path
            }
            
        # Create comparison sheet
        self.create_comparison_sheet(variants, output_dir)
        
        return variants
        
    def create_template(self, result, style_name):
        """Create template with region numbers"""
        # You can reuse your template creation code here
        # or integrate with your existing NumberPlacer class
        
        # For now, just create a simple template
        vectorized = result['vectorized']
        edges = result['edges']
        
        # Create white background
        template = np.ones_like(vectorized) * 255
        
        # Apply edges
        for i in range(3):
            template[:,:,i] = np.where(edges > 0, 0, template[:,:,i])
        
        return template
        
    def create_comparison_sheet(self, variants, output_dir):
        """Create a comparison sheet of all variants"""
        # Determine grid size
        n_variants = len(variants)
        cols = min(3, n_variants)
        rows = (n_variants + cols - 1) // cols
        
        # Get image size from first variant
        first_variant = list(variants.values())[0]
        img_height, img_width = first_variant['result']['vectorized'].shape[:2]
        
        # Create comparison sheet
        grid_height = rows * img_height
        grid_width = cols * img_width
        
        comparison = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
        
        # Add each variant
        for i, (name, variant) in enumerate(variants.items()):
            row = i // cols
            col = i % cols
            
            # Add the image
            y = row * img_height
            x = col * img_width
            img = variant['result']['vectorized']
            comparison[y:y+img_height, x:x+img_width] = img
            
            # Add text label
            y_text = y + 20
            x_text = x + 10
            cv2.putText(comparison, name, (x_text, y_text), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
        # Save comparison sheet
        comparison_path = os.path.join(output_dir, "comparison_sheet.png")
        plt.imsave(comparison_path, comparison)
        
        # Also create a comparison of templates
        template_comparison = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
        
        for i, (name, variant) in enumerate(variants.items()):
            row = i // cols
            col = i % cols
            
            # Add the template image
            y = row * img_height
            x = col * img_width
            
            # Load the template image
            template = plt.imread(variant['template_path'])
            template_comparison[y:y+img_height, x:x+img_width] = (template * 255).astype(np.uint8)
            
            # Add text label
            y_text = y + 20
            x_text = x + 10
            cv2.putText(template_comparison, name, (x_text, y_text), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Save template comparison sheet
        template_comparison_path = os.path.join(output_dir, "template_comparison_sheet.png")
        plt.imsave(template_comparison_path, template_comparison)