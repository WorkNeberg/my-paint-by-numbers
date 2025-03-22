import cv2
import numpy as np
from core.image_processor import ImageProcessor
from .feature_preserver import FeaturePreserver
from .image_type_detector import ImageTypeDetector
from .settings_manager import SettingsManager
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time
from enhanced.pet_pattern_preserver import PetPatternPreserver
import logging

# Add these imports at the very top of the file
import os
import cv2
import numpy as np
import logging
from enhanced.pet_pattern_preserver import PetPatternPreserver
from enhanced.image_type_detector import ImageTypeDetector
from typing import Dict, Any, Union, Optional, List, Tuple

# Setup logging
logger = logging.getLogger('pbn-app')

class EnhancedProcessor:
    def __init__(self):
        """Initialize processor with pet pattern preservation"""
        try:
            # Import PetPatternPreserver
            from enhanced.pet_pattern_preserver import PetPatternPreserver
            self.pet_pattern_preserver = PetPatternPreserver()
            
            # Import other required components
            from enhanced.feature_preserver import FeaturePreserver
            self.feature_preserver = FeaturePreserver()
            
            # Other initializations...
            self.base_processor = ImageProcessor()
            self.type_detector = ImageTypeDetector()
            self.settings_manager = SettingsManager()
            self.image_processor = None  # This would be your existing image processor
            
            # Log successful initialization
            logging.getLogger('pbn-app').info("Initialized EnhancedProcessor with pet pattern preservation")
        except Exception as e:
            logging.getLogger('pbn-app').error(f"Error initializing EnhancedProcessor: {e}")
        
    def process_with_feature_preservation(self, image, settings, image_type='general', feature_mask=None):
        """Process image while preserving important features"""
        logger = logging.getLogger('pbn-app')
        
        # Create feature mask if not provided
        if feature_mask is None:
            try:
                feature_mask, _ = self.feature_preserver.create_feature_mask(image, image_type)
            except Exception as e:
                logger.error(f"Error creating feature mask: {e}")
                feature_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Check if image_processor exists and is not None
        if hasattr(self, 'image_processor') and self.image_processor is not None:
            try:
                return self.image_processor.process_image(
                    image, 
                    settings.get('colors', 15),
                    settings.get('complexity', 'medium'),
                    feature_mask
                )
            except Exception as e:
                logger.error(f"Error using image_processor: {e}")
                # Fall back to basic processing
        
        # Always use the fallback if image_processor is None or fails
        logger.info("Using fallback basic processing")
        return self._create_basic_result(image, settings)
    
    def preprocess_image(self, image, settings=None, image_type='auto'):
        """
        Enhanced preprocessing with adaptive settings based on image type and content
        
        Parameters:
        - image: Input RGB image
        - settings: Dictionary of settings, or None
        - image_type: 'auto', 'portrait', 'pet', 'landscape', 'general', etc.
        
        Returns:
        - Preprocessed image ready for PBN generation
        """
        # Parse settings if provided
        if settings is None:
            settings = {}
        
        enhance_dark_areas = settings.get('enhance_dark_areas', False)
        dark_threshold = settings.get('dark_threshold', 50)
        enhance_colors = settings.get('enhance_colors', True)
        sharpen_level = settings.get('sharpen_level', 1.0)
        
        # Start with a copy of the image
        processed = image.copy()
        
        # Step 1: Determine image type if set to auto
        if image_type == 'auto':
            image_type = self.type_detector.detect_type_simple(processed)[0]
            print(f"Auto-detected image type: {image_type}")
        
        # Step 2: Apply dark area enhancement if requested
        if enhance_dark_areas:
            # Adaptive dark threshold based on image type
            adaptive_threshold = dark_threshold
            if image_type == 'pet':
                # Pets often have dark areas that need enhancement
                adaptive_threshold = max(40, dark_threshold - 10)
            elif image_type == 'portrait':
                # Portraits may have shadows that need enhancement
                adaptive_threshold = max(45, dark_threshold - 5)
            
            # Apply enhanced dark area processing
            processed = self._enhance_dark_areas(processed, adaptive_threshold, image_type)
        
        # Step 3: Apply type-specific processing
        if image_type == 'portrait':
            # For portraits, preserve skin tones and facial features
            # Apply bilateral filter to smooth skin while preserving edges
            processed = cv2.bilateralFilter(processed, 9, 75, 75)
            
            # Try to detect and enhance face features
            try:
                # Convert to grayscale for detection
                gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
                
                # Detect face
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    # Get face region
                    face_roi = processed[y:y+h, x:x+w]
                    
                    # Apply subtle contrast enhancement to face
                    face_lab = cv2.cvtColor(face_roi, cv2.COLOR_RGB2LAB)
                    l, a, b = cv2.split(face_lab)
                    
                    # Apply CLAHE to L channel
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
                    l_enhanced = clahe.apply(l)
                    
                    # Merge channels back and convert to RGB
                    face_enhanced = cv2.merge([l_enhanced, a, b])
                    face_enhanced = cv2.cvtColor(face_enhanced, cv2.COLOR_LAB2RGB)
                    
                    # Blend enhanced face back to original (70% enhanced, 30% original)
                    processed[y:y+h, x:x+w] = cv2.addWeighted(face_enhanced, 0.7, face_roi, 0.3, 0)
                    
                    # Try to detect eyes for special enhancement
                    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                    eyes = eye_cascade.detectMultiScale(gray[y:y+h, x:x+w])
                    
                    for (ex, ey, ew, eh) in eyes:
                        # Enhance contrast and sharpness for eyes
                        eye_roi = processed[y+ey:y+ey+eh, x+ex:x+ex+ew]
                        
                        # Apply stronger contrast enhancement
                        eye_lab = cv2.cvtColor(eye_roi, cv2.COLOR_RGB2LAB)
                        l_eye, a_eye, b_eye = cv2.split(eye_lab)
                        
                        # Stronger CLAHE for eyes
                        clahe_eye = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3, 3))
                        l_eye_enhanced = clahe_eye.apply(l_eye)
                        
                        # Merge and convert back
                        eye_enhanced = cv2.merge([l_eye_enhanced, a_eye, b_eye])
                        eye_enhanced = cv2.cvtColor(eye_enhanced, cv2.COLOR_LAB2RGB)
                        
                        # Apply sharpening to eyes
                        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                        eye_enhanced = cv2.filter2D(eye_enhanced, -1, kernel)
                        
                        # Replace in the image
                        processed[y+ey:y+ey+eh, x+ex:x+ex+ew] = eye_enhanced
            
            except Exception as e:
                print(f"Warning: Error in face/eye enhancement: {e}")
                # Continue with standard processing
        
        elif image_type == 'pet':
            # For pets, use specialized filtering to preserve details in fur and eyes
            processed = cv2.bilateralFilter(processed, 9, 100, 100)
            
            # Special handling for pet eyes and dark features
            try:
                # Convert to HSV for better feature detection
                hsv = cv2.cvtColor(processed, cv2.COLOR_RGB2HSV)
                v_channel = hsv[:,:,2]
                
                # Detect potential eye regions (small dark circles)
                dark_regions = v_channel < 60
                
                # Use morphological operations to identify potential eye regions
                if np.any(dark_regions):
                    dark_mask = dark_regions.astype(np.uint8) * 255
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
                    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
                    
                    # Find contours in dark regions
                    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Process potential eye regions
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > 20 and area < 500:  # Size filter for eyes
                            x, y, w, h = cv2.boundingRect(contour)
                            aspect_ratio = float(w) / h
                            
                            # Eye-like regions typically have aspect ratio close to 1
                            if 0.5 < aspect_ratio < 2.0:
                                # Get this eye region
                                eye_region = processed[y:y+h, x:x+w]
                                
                                # Enhance contrast
                                eye_lab = cv2.cvtColor(eye_region, cv2.COLOR_RGB2LAB)
                                l, a, b = cv2.split(eye_lab)
                                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3,3))
                                l = clahe.apply(l)
                                enhanced_eye = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
                                
                                # Apply sharpening
                                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                                enhanced_eye = cv2.filter2D(enhanced_eye, -1, kernel)
                                
                                # Replace the eye region with enhanced version
                                processed[y:y+h, x:x+w] = enhanced_eye
                
                # Enhance fur texture
                gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
                
                # Create fur texture enhancement mask
                fur_mask = edges.astype(np.float32) / 255.0
                fur_mask = cv2.GaussianBlur(fur_mask, (3, 3), 0)
                
                # Apply mild sharpening to fur texture areas
                sharpened = cv2.filter2D(processed, -1, np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]]))
                
                # Blend sharpened and original
                for c in range(3):
                    processed[:,:,c] = processed[:,:,c] * (1 - fur_mask) + sharpened[:,:,c] * fur_mask
            
            except Exception as e:
                print(f"Warning: Error in pet feature enhancement: {e}")
                # Continue with standard processing
        
        elif image_type == 'landscape':
            # For landscapes, enhance natural colors and preserve details
            hsv = cv2.cvtColor(processed, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            
            # Slightly boost saturation for more vibrant colors
            s = cv2.multiply(s, 1.2)
            s = np.clip(s, 0, 255).astype(np.uint8)
            
            # Merge channels back
            hsv_enhanced = cv2.merge([h, s, v])
            processed = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)
            
            # Apply adaptive bilateral filter for edge preservation
            processed = cv2.bilateralFilter(processed, 9, 75, 75)
        
        else:  # General processing for other image types
            # Apply basic bilateral filter for noise reduction while preserving edges
            processed = cv2.bilateralFilter(processed, 7, 50, 50)
        
        # Step 4: Apply color enhancement if requested
        if enhance_colors:
            # Enhance colors based on image type
            if image_type == 'portrait':
                # For portraits, enhance skin tones
                processed = self._enhance_skin_tones(processed)
            elif image_type == 'pet':
                # For pets, enhance fur colors
                processed = self._enhance_fur_colors(processed)
            else:
                # For other types, apply general color enhancement
                processed = self._enhance_general_colors(processed)
        
        # Step 5: Apply adaptive sharpening based on sharpen_level
        if sharpen_level > 0:
            # Create sharpening kernel based on level
            alpha = 0.5 + (sharpen_level * 0.5)  # Map 0-1 to 0.5-1.0
            kernel = np.array([[-alpha, -alpha, -alpha], 
                            [-alpha, 1 + 8*alpha, -alpha], 
                            [-alpha, -alpha, -alpha]])
            
            # Apply sharpening
            sharpened = cv2.filter2D(processed, -1, kernel)
            
            # Blend with original based on sharpen_level
            processed = cv2.addWeighted(processed, 1 - sharpen_level*0.3, sharpened, sharpen_level*0.3, 0)
        
        return processed

    def _enhance_dark_areas(self, image, dark_threshold, image_type):
        """
        Enhanced version of dark area processing with adaptive parameters
        
        Parameters:
        - image: Input RGB image
        - dark_threshold: Threshold for dark area detection
        - image_type: Type of image for specialized handling
        
        Returns:
        - Enhanced image with better dark area details
        """
        print(f"Applying enhanced dark area processing for {image_type} image (threshold: {dark_threshold})")
        
        # Set a maximum computation time limit
        max_processing_time = 5  # seconds
        start_time = time.time()
        
        try:
            # Create a copy to preserve original
            image_enhanced = image.copy()
            
            # Convert to HSV for better dark area detection
            hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            v_channel = hsv_img[:,:,2]
            
            # Analyze histogram to find better threshold adaptively
            hist = cv2.calcHist([v_channel], [0], None, [256], [0, 256])
            hist_cumsum = np.cumsum(hist) / np.sum(hist)
            
            # Find threshold that covers the darkest 15% of pixels (adaptive)
            for i in range(256):
                if hist_cumsum[i] > 0.15:
                    adaptive_threshold = max(dark_threshold - 10, i + 5)
                    break
            else:
                adaptive_threshold = dark_threshold
            
            # Create an adaptive dark mask
            dark_mask = v_channel < adaptive_threshold
            
            # Calculate dark area percentage
            dark_pixel_percentage = np.sum(dark_mask) / dark_mask.size * 100
            print(f"Dark area detected: {dark_pixel_percentage:.1f}% (adaptive threshold: {adaptive_threshold})")
            
            # Check processing time
            if time.time() - start_time > max_processing_time:
                print(f"Dark area enhancement taking too long, using simplified processing")
                # Skip to a simpler enhancement
                gradient_mask = dark_mask.astype(np.float32)
                gradient_mask_3ch = np.stack([gradient_mask] * 3, axis=2)
                
                # Simple enhancement - just brighten dark areas
                enhanced_rgb = np.minimum(image_enhanced * 1.3, 255).astype(np.uint8)
                image_enhanced = image_enhanced * (1 - gradient_mask_3ch) + enhanced_rgb * gradient_mask_3ch
                return image_enhanced
            
            if np.sum(dark_mask) > 0:
                # Create gradient mask for smooth transition
                gradient_mask = dark_mask.astype(np.float32)
                gradient_mask = cv2.GaussianBlur(gradient_mask, (15, 15), 0)
                
                # Check processing time again
                if time.time() - start_time > max_processing_time:
                    print(f"Dark area enhancement taking too long, using simplified processing")
                    gradient_mask_3ch = np.stack([gradient_mask] * 3, axis=2)
                    enhanced_rgb = np.minimum(image_enhanced * 1.3, 255).astype(np.uint8)
                    image_enhanced = image_enhanced * (1 - gradient_mask_3ch) + enhanced_rgb * gradient_mask_3ch
                    return image_enhanced
                
                # Customize CLAHE parameters based on image type and darkness
                if image_type == 'pet' and dark_pixel_percentage > 30:
                    # Stronger enhancement for dark pets (like black cats)
                    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4,4))
                elif image_type == 'portrait' and dark_pixel_percentage > 40:
                    # Strong but not too aggressive for dark portraits
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6,6))
                else:
                    # Standard enhancement for other cases
                    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
                
                # Convert to LAB color space (better for enhancement)
                lab = cv2.cvtColor(image_enhanced, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # Check processing time again
                if time.time() - start_time > max_processing_time:
                    print(f"Dark area enhancement taking too long, using simplified processing")
                    # Return partially enhanced image
                    return image_enhanced
                
                # Apply CLAHE to the L channel
                l_enhanced = clahe.apply(l)
                
                # Merge channels back
                lab_enhanced = cv2.merge((l_enhanced, a, b))
                enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
                
                # Prepare for blending
                gradient_mask_3ch = np.stack([gradient_mask] * 3, axis=2)
                
                # Blend enhanced and original using gradient mask
                image_enhanced = image_enhanced * (1 - gradient_mask_3ch) + enhanced_rgb * gradient_mask_3ch
                
                # Check processing time again
                if time.time() - start_time > max_processing_time:
                    print(f"Dark area enhancement taking too long, skipping gamma correction")
                    return image_enhanced
                
                # Apply adaptive gamma correction for dark areas
                if dark_pixel_percentage > 20:
                    # Adaptive gamma based on darkness and image type
                    if image_type == 'pet' and dark_pixel_percentage > 40:
                        gamma = 0.65  # Stronger correction for very dark pets
                    elif dark_pixel_percentage > 50:
                        gamma = 0.7   # Strong correction for very dark images
                    else:
                        gamma = 0.8   # Moderate correction
                        
                    # Create lookup table for gamma correction
                    lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(0, 256)]).astype("uint8")
                    
                    # Apply gamma correction only to dark areas
                    for i in range(3):
                        temp = image_enhanced[:,:,i].copy()
                        temp = cv2.LUT(temp, lookup_table)
                        image_enhanced[:,:,i] = image_enhanced[:,:,i] * (1 - gradient_mask_3ch[:,:,i]) + \
                                            temp * gradient_mask_3ch[:,:,i]
                
                # Skip the pet-specific processing if we're running out of time
                if time.time() - start_time > max_processing_time:
                    return image_enhanced
                
                # Special handling for pets - detect and enhance eyes in dark regions
                if image_type == 'pet':
                    # Simple eye enhancement for pets - don't do the full processing
                    pass  # Removed complex eye detection to prevent timeouts
            
            return image_enhanced
            
        except Exception as e:
            # If anything fails, return the original image
            print(f"Warning: Error in dark area enhancement: {e}")
            return image
    
    def _enhance_skin_tones(self, image):
        """
        Enhance skin tones for better visual quality in portraits
        
        Parameters:
        - image: Input RGB image
        
        Returns:
        - Image with enhanced skin tones
        """
        # Convert to YCrCb color space which is good for skin detection
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        
        # Skin tone range in YCrCb
        skin_mask = np.zeros(y.shape, dtype=np.uint8)
        
        # Common skin tone range in YCrCb
        skin_mask = np.logical_and(cr >= 135, cr <= 180)
        skin_mask = np.logical_and(skin_mask, cb >= 85)
        skin_mask = np.logical_and(skin_mask, cb <= 135)
        skin_mask = skin_mask.astype(np.uint8) * 255
        
        # Smooth the mask
        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
        skin_mask = (skin_mask / 255.0).astype(np.float32)
        
        # Create 3-channel mask
        skin_mask_3ch = np.stack([skin_mask] * 3, axis=2)
        
        # Convert to LAB for better enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Ensure all channels are 8-bit unsigned integers
        l = l.astype(np.uint8)
        a = a.astype(np.uint8)
        b = b.astype(np.uint8)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Make sure enhanced L channel is uint8
        l_enhanced = l_enhanced.astype(np.uint8)
        
        # For skin areas, warm up the color slightly (increase a channel)
        a_skin = np.clip(a + 3 * skin_mask, 0, 255).astype(np.uint8)
        
        # Merge channels and convert back
        lab_enhanced = cv2.merge([l_enhanced, a_skin, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        # Blend with original using skin mask
        result = image * (1 - skin_mask_3ch * 0.7) + enhanced * (skin_mask_3ch * 0.7)
        
        return result.astype(np.uint8)

    def _enhance_fur_colors(self, image):
        """
        Enhance fur colors and textures for pet images
        
        Parameters:
        - image: Input RGB image
        
        Returns:
        - Image with enhanced fur details
        """
        # Use bilateral filter to preserve edges while smoothing
        filtered = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Enhance local contrast
        gray = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        
        # Create detail mask using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        detail_mask = np.abs(laplacian)
        detail_mask = cv2.GaussianBlur(detail_mask, (5, 5), 0)
        detail_mask = detail_mask / np.max(detail_mask) if np.max(detail_mask) > 0 else detail_mask
        
        # Create 3-channel detail mask
        detail_mask_3ch = np.stack([detail_mask] * 3, axis=2)
        
        # Enhance contrast using LAB color space
        lab = cv2.cvtColor(filtered, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Ensure all channels are 8-bit unsigned integers
        l = l.astype(np.uint8)
        a = a.astype(np.uint8)
        b = b.astype(np.uint8)
        
        l_enhanced = clahe.apply(l)
        
        # Make sure enhanced L channel is uint8
        l_enhanced = l_enhanced.astype(np.uint8)
        
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        # Sharpen details
        kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Blend based on detail mask
        result = filtered * (1 - detail_mask_3ch * 0.7) + sharpened * (detail_mask_3ch * 0.7)
        
        return result.astype(np.uint8)

    def _enhance_general_colors(self, image):
        """
        General color enhancement for better visual quality
        
        Parameters:
        - image: Input RGB image
        
        Returns:
        - Image with enhanced colors
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Ensure all channels are 8-bit unsigned integers
        l = l.astype(np.uint8)
        a = a.astype(np.uint8)
        b = b.astype(np.uint8)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Make sure enhanced L channel remains the same type as a and b
        l_enhanced = l_enhanced.astype(np.uint8)
        
        # Merge channels and convert back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        # Apply subtle saturation boost
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Ensure all HSV channels are 8-bit unsigned integers
        h = h.astype(np.uint8)
        s = s.astype(np.uint8)
        v = v.astype(np.uint8)
        
        # Increase saturation slightly
        s = np.clip(s * 1.1, 0, 255).astype(np.uint8)
        
        # Merge channels and convert back
        hsv_enhanced = cv2.merge([h, s, v])
        result = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)
        
        # Add the return statement
        return result.astype(np.uint8)
        
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
        
        # Always pass enhance_dark_areas=False here to prevent double enhancement
        vectorized, label_image, edges, centers, region_data, paintability = (
            self.base_processor.process_image(
                image,
                settings.get('num_colors', 15),
                settings.get('simplification_level', 'medium'),
                settings.get('edge_strength', 1.0),
                settings.get('edge_width', 1),
                False,  # IMPORTANT: Always False to prevent double enhancement
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

    def _enhance_fur_texture(self, image, fur_type='generic'):
        """
        Enhanced fur texture processing based on fur type
        
        Parameters:
        - image: Input RGB image
        - fur_type: String indicating fur type ('short', 'long', 'curly', etc.)
        
        Returns:
        - Image with enhanced fur texture
        """
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect edges for fur texture map
        edges = cv2.Canny(gray, 30, 150)
        
        # Different kernel sizes based on fur type
        if fur_type == 'long':
            kernel_size = 5
            detail_boost = 1.4
        elif fur_type == 'curly':
            kernel_size = 3
            detail_boost = 1.6
        else:  # short or generic
            kernel_size = 7
            detail_boost = 1.2
        
        # Create fur mask with appropriate dilation
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        fur_mask = cv2.dilate(edges, kernel)
        fur_mask = cv2.GaussianBlur(fur_mask, (5, 5), 0)
        fur_mask = fur_mask / 255.0
        
        # Enhance details in fur areas
        kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,4+detail_boost,-0.5], [-0.5,-0.5,-0.5]])
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Apply fur-specific color enhancement
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Slightly boost saturation for more vibrant fur
        s = np.clip(s * 1.1, 0, 255).astype(np.uint8)
        
        # Blend enhanced image
        enhanced_hsv = cv2.merge([h, s, v])
        enhanced_color = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)
        
        # Create 3-channel mask
        fur_mask_3ch = np.stack([fur_mask] * 3, axis=2)
        
        # Blend sharpened and color-enhanced versions
        result = image * (1 - fur_mask_3ch) + (sharpened * 0.6 + enhanced_color * 0.4) * fur_mask_3ch
        
        return result.astype(np.uint8)

    def _detect_pet_features(self, image):
        """
        Enhanced detection of pet features (eyes, nose, ears)
        
        Parameters:
        - image: Input RGB image
        
        Returns:
        - Dictionary of detected feature regions
        """
        features = {
            'eyes': [],
            'nose': [],
            'ears': []
        }
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Try cat face detection first
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 3)
        
        if len(faces) > 0:
            # Use cat face detection results
            for (x, y, w, h) in faces:
                # Estimate eye positions (typically in upper half of face)
                eye_y = y + int(h * 0.35)
                left_eye_x = x + int(w * 0.3)
                right_eye_x = x + int(w * 0.7)
                eye_size = max(int(h * 0.1), 5)
                
                features['eyes'].append((left_eye_x, eye_y, eye_size*2, eye_size))
                features['eyes'].append((right_eye_x, eye_y, eye_size*2, eye_size))
                
                # Estimate nose position (typically in center-bottom of face)
                nose_x = x + int(w * 0.5)
                nose_y = y + int(h * 0.65)
                nose_size = max(int(h * 0.15), 8)
                
                features['nose'].append((nose_x, nose_y, nose_size, nose_size))
        else:
            # Fallback to dark region detection for eyes
            v_channel = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:,:,2]
            dark_regions = v_channel < 60
            
            if np.any(dark_regions):
                dark_mask = dark_regions.astype(np.uint8) * 255
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
                dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
                
                contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 20 < area < 500:  # Size filter for possible eyes
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = float(w) / h
                        
                        # Eyes typically have aspect ratio close to 1
                        if 0.5 < aspect_ratio < 2.0:
                            features['eyes'].append((x, y, w, h))
        
        return features

    def _enhance_pet_by_color(self, image):
        """
        Apply specialized enhancement based on pet's dominant color
        
        Parameters:
        - image: Input RGB image
        
        Returns:
        - Enhanced image specific to pet color
        """
        # Analyze color histogram to determine dominant colors
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Get average brightness and saturation
        avg_v = np.mean(v)
        avg_s = np.mean(s)
        
        # Initialize enhanced image
        enhanced = image.copy()
        
        # Special handling for dark/black pets (low brightness)
        if avg_v < 70:
            print("Detected dark/black pet - applying specialized enhancement")
            
            # Create dark mask
            dark_mask = (v < 60).astype(np.float32)
            dark_mask = cv2.GaussianBlur(dark_mask, (9, 9), 0)
            dark_mask_3ch = np.stack([dark_mask] * 3, axis=2)
            
            # Apply stronger detail enhancement in dark areas
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Stronger CLAHE for dark pets
            clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
            l_enhanced = clahe.apply(l)
            
            # Merge and convert back
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            dark_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
            
            # Apply sharpening to bring out details
            kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,6,-0.5], [-0.5,-0.5,-0.5]])
            sharpened = cv2.filter2D(dark_enhanced, -1, kernel)
            
            # Blend original with enhanced version
            enhanced = image * (1 - dark_mask_3ch * 0.8) + sharpened * (dark_mask_3ch * 0.8)
            
        # Special handling for white/light pets (high brightness, low saturation)
        elif avg_v > 180 and avg_s < 50:
            print("Detected white/light pet - applying specialized enhancement")
            
            # Enhance subtle details in light areas
            # Increase contrast slightly to bring out subtle features
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Custom CLAHE for light pets - lower clip limit to avoid over-processing
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            # Merge and convert back
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced.astype(np.uint8)

    def process_pet_image(self, image, settings):
        """
        Process pet image with pattern preservation
        """
        logger = logging.getLogger('pbn-app')
        logger.info("Processing pet image with pattern preservation")
        
        # Make a copy of the image for processing
        processed = image.copy()
        h, w = processed.shape[:2]
        
        # Step 1: Detect and enhance pet patterns
        feature_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Use pet pattern preserver if available
        if hasattr(self, 'pet_pattern_preserver'):
            try:
                logger.info("Calling pet pattern preserver")
                pattern_mask, enhanced_image = self.pet_pattern_preserver.detect_and_preserve(processed)
                
                # Use enhanced image for further processing
                processed = enhanced_image
                
                # Add pattern mask to feature mask
                feature_mask = cv2.bitwise_or(feature_mask, pattern_mask)
                
                logger.info("Pet pattern preservation applied successfully")
            except Exception as e:
                logger.error(f"Error in pet pattern preservation: {e}", exc_info=True)
        
        # Step 2: Apply pre-processing
        complexity = settings.get('complexity', 'medium')
        
        # Set blur intensity based on complexity
        blur_size = 7 if complexity == 'low' else (3 if complexity == 'high' else 5)
        
        # Apply bilateral filter for smoothing while preserving edges
        processed = cv2.bilateralFilter(processed, blur_size, 75, 75)
        
        # Step 3: Color quantization - use number of colors from settings
        num_colors = settings.get('colors', 15)
        
        # Prepare image for k-means
        pixels = processed.reshape(-1, 3).astype(np.float32)
        
        # Run k-means for color segmentation
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to 8-bit image
        centers = np.uint8(centers)
        segmented_flat = centers[labels.flatten()]
        segmented = segmented_flat.reshape(processed.shape)
        
        # Step 4: Apply edge enhancement to make regions distinct
        edges = cv2.Canny(cv2.cvtColor(segmented, cv2.COLOR_RGB2GRAY), 50, 150)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        
        # Create edge overlay
        edge_overlay = segmented.copy()
        edge_overlay[edges > 0] = [0, 0, 0]
        
        # Step 5: Create final output
        result = {
            'processed_image': processed,
            'segmented_image': segmented,
            'preview_image': edge_overlay,
            'colors': [{'rgb': [int(c[0]), int(c[1]), int(c[2])], 'count': int(np.sum(labels == i))} for i, c in enumerate(centers)],
            'paintability': 90,  # Custom paintability score for pets
        }
        
        logger.info(f"Pet image processed with {num_colors} colors")
        return result

    def process_image(self, image_path, settings):
        """Process image with specialized handling based on image type"""
        logger = logging.getLogger('pbn-app')
        
        # Load image if path provided
        if isinstance(image_path, str):
            try:
                logger.info(f"Reading image from: {image_path}")
                file_size = os.path.getsize(image_path)
                logger.info(f"File size: {file_size} bytes")
                
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not read image: {image_path}")
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                logger.info(f"Image shape after reading: {image.shape}")
            except Exception as e:
                logger.error(f"Error reading image: {e}")
                raise
        else:
            image = image_path
        
        # CRITICAL: Get image type from settings properly
        image_type = settings.get('image_type', 'general')
        preset_type = settings.get('preset_type', '')
        
        # Double-check preset type for pet
        if preset_type == 'pet':
            image_type = 'pet'
            settings['image_type'] = 'pet'
            logger.info("Using pet preset - forcing image_type to pet")
        
        logger.info(f"Processing image as type: {image_type}")
        
        # Route to appropriate processing method
        try:
            if image_type == 'pet':
                logger.info("*** USING PET IMAGE PROCESSING PATH ***")
                return self.process_pet_image(image, settings)
            elif image_type == 'portrait':
                return self.process_portrait_image(image, settings)
            elif image_type == 'landscape':
                return self.process_landscape_image(image, settings)
            else:
                return self.process_generic_image(image, settings)
        except Exception as e:
            logger.error(f"Error in specialized processing: {str(e)}", exc_info=True)
            # Fallback to generic processing
            return self.process_generic_image(image, settings)
    
    def process_portrait_image(self, image, settings):
        """Process portrait image with facial feature preservation"""
        # This is a placeholder - implement your portrait-specific processing
        return self.process_with_feature_preservation(image, settings, 'portrait')
    
    def process_landscape_image(self, image, settings):
        """Process landscape image with horizon/structure preservation"""
        # This is a placeholder - implement your landscape-specific processing
        return self.process_with_feature_preservation(image, settings, 'landscape')
    
    def process_still_life_image(self, image, settings):
        """Process still life image with object boundary preservation"""
        # This is a placeholder - implement your still life-specific processing
        return self.process_with_feature_preservation(image, settings, 'still_life')
    
    def process_generic_image(self, image, settings):
        """Process generic image without specialized handling"""
        # This is a placeholder - implement your generic processing
        return self.process_with_feature_preservation(image, settings, 'general')

    def _create_basic_result(self, image, settings):
        """Create a basic result if all other processing fails"""
        logger = logging.getLogger('pbn-app')
        logger.info("Using fallback basic processing")
        
        # Make a copy of the image
        processed = image.copy()
        
        # Apply basic blur
        processed = cv2.GaussianBlur(processed, (5, 5), 0)
        
        # Simple color quantization
        num_colors = settings.get('colors', 10)
        pixels = processed.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to 8-bit image
        centers = np.uint8(centers)
        segmented_flat = centers[labels.flatten()]
        segmented = segmented_flat.reshape(image.shape)
        
        # Create a basic preview
        preview = segmented.copy()
        
        # Return the result
        result = {
            'processed_image': processed,
            'segmented_image': segmented,
            'preview_image': preview,
            'colors': [{'rgb': [int(c[0]), int(c[1]), int(c[2])], 'count': int(np.sum(labels == i))} for i, c in enumerate(centers)],
            'paintability': 50,  # Default value for fallback processing
        }
        
        logger.info(f"Basic fallback processing complete with {num_colors} colors")
        return result

    def _optimize_colors_for_pet_features(self, image, colors, eye_regions):
        """Ensure eyes get appropriate colors"""
        new_colors = colors.copy()
        
        # Extract eye region colors
        eye_colors = []
        for (x, y, w, h) in eye_regions:
            eye_area = image[y:y+h, x:x+w]
            if eye_area.size > 0:
                # Get average color of eye region
                avg_color = np.mean(eye_area.reshape(-1, 3), axis=0)
                eye_colors.append(avg_color)
        
        # Make sure at least one eye color is in the palette
        if eye_colors:
            replaced = False
            for eye_color in eye_colors:
                # Find similar color in palette
                distances = [np.sum((c - eye_color)**2) for c in new_colors]
                min_dist_idx = np.argmin(distances)
                
                # If no similar color exists, replace least used color
                if distances[min_dist_idx] > 1000:  # Threshold for similarity
                    new_colors[min_dist_idx] = eye_color
                    replaced = True
                    
            if not replaced:
                # Ensure eye color is distinct enough
                idx = np.argmin(distances)
                new_colors[idx] = eye_colors[0]
        
        return new_colors