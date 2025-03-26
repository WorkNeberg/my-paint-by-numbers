import cv2
import numpy as np
import logging
from enum import Enum

logger = logging.getLogger('pbn-app.image_preprocessor')

class PreprocessingMode(Enum):
    STANDARD = "standard"
    ENHANCED = "enhanced"
    ARTISTIC = "artistic"

class ImagePreprocessor:
    """
    Handles image preprocessing operations with support for different modes
    and parameter-based configuration.
    """
    
    def __init__(self):
        """Initialize the image preprocessor"""
        logger.info("ImagePreprocessor initialized")
        
    def preprocess(self, image, settings):
        """
        Preprocess an image according to the specified settings
        
        Args:
            image: Input image as numpy array (BGR format)
            settings: Dictionary of preprocessing settings
            
        Returns:
            Preprocessed image
        """
        logger.info("Starting image preprocessing")
        
        # Make a copy to avoid modifying the original
        processed = image.copy()
        
        # Extract preprocessing settings
        preprocessing = settings.get('preprocessing', {})
        mode = preprocessing.get('preprocessing_mode', 'standard')
        sharpen_level = preprocessing.get('sharpen_level', 0.5)
        contrast_boost = preprocessing.get('contrast_boost', 0.0)
        noise_reduction = preprocessing.get('noise_reduction_level', 0.3)
        detail_preservation = preprocessing.get('detail_preservation', 'medium')
        
        # Create preprocessing pipeline based on mode
        if mode == PreprocessingMode.STANDARD.value:
            processed = self._standard_preprocessing(processed, 
                                                    sharpen_level,
                                                    contrast_boost,
                                                    noise_reduction)
            
        elif mode == PreprocessingMode.ENHANCED.value:
            processed = self._enhanced_preprocessing(processed, 
                                                   sharpen_level,
                                                   contrast_boost,
                                                   noise_reduction,
                                                   detail_preservation)
            
        elif mode == PreprocessingMode.ARTISTIC.value:
            processed = self._artistic_preprocessing(processed,
                                                   sharpen_level,
                                                   contrast_boost)
        
        # Extract other settings that might affect preprocessing
        dark_enhancement = settings.get('color_control', {}).get('dark_area_enhancement', 0.5)
        light_protection = settings.get('color_control', {}).get('light_area_protection', 0.5)
        
        # Apply common enhancements
        if dark_enhancement > 0:
            processed = self._enhance_dark_areas(processed, dark_enhancement)
            
        if light_protection > 0:
            processed = self._protect_light_areas(processed, light_protection)
            
        logger.info("Image preprocessing completed")
        return processed
    
    def _standard_preprocessing(self, image, sharpen_level, contrast_boost, noise_reduction):
        """Standard preprocessing pipeline"""
        logger.debug("Applying standard preprocessing")
        
        # Apply noise reduction if needed
        if noise_reduction > 0:
            image = self._apply_noise_reduction(image, noise_reduction)
        
        # Apply contrast enhancement if needed
        if contrast_boost > 0:
            image = self._enhance_contrast(image, contrast_boost)
            
        # Apply sharpening if needed
        if sharpen_level > 0:
            image = self._apply_sharpening(image, sharpen_level)
            
        return image
    
    def _enhanced_preprocessing(self, image, sharpen_level, contrast_boost, 
                               noise_reduction, detail_preservation):
        """Enhanced preprocessing with better detail preservation"""
        logger.debug("Applying enhanced preprocessing")
        
        # Create a detail mask to preserve important features
        detail_mask = self._create_detail_mask(image, detail_preservation)
        
        # Apply noise reduction with detail preservation
        if noise_reduction > 0:
            image = self._apply_adaptive_noise_reduction(image, noise_reduction, detail_mask)
            
        # Apply local contrast enhancement
        image = self._enhance_local_contrast(image, contrast_boost, detail_mask)
        
        # Apply adaptive sharpening
        if sharpen_level > 0:
            image = self._apply_adaptive_sharpening(image, sharpen_level, detail_mask)
            
        return image
    
    def _artistic_preprocessing(self, image, sharpen_level, contrast_boost):
        """Artistic preprocessing with stylized effects"""
        logger.debug("Applying artistic preprocessing")
        
        # Apply bilateral filter for painterly effect
        image = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Boost color saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype("float32")
        (h, s, v) = cv2.split(hsv)
        s = s * (1.0 + contrast_boost * 0.5)
        s = np.clip(s, 0, 255)
        hsv = cv2.merge([h, s, v])
        image = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
        
        # Apply edge-preserving filter
        image = cv2.edgePreservingFilter(image, flags=1, sigma_s=60, sigma_r=0.4)
        
        # Apply stylization filter
        image = cv2.stylization(image, sigma_s=60, sigma_r=0.07)
        
        return image
    
    def _apply_noise_reduction(self, image, strength):
        """Apply basic noise reduction"""
        if strength <= 0:
            return image
        
        strength = min(1.0, strength)
        kernel_size = int(3 + strength * 4)
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
            
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def _apply_adaptive_noise_reduction(self, image, strength, detail_mask):
        """Apply noise reduction while preserving details"""
        if strength <= 0:
            return image
            
        # Apply stronger blur to non-detail areas
        blurred = cv2.GaussianBlur(image, (9, 9), 0)
        
        # Use detail mask to blend original and blurred image
        blend_factor = np.clip(1.0 - detail_mask * (1.0 - strength * 0.5), 0, 1)
        blend_factor = np.dstack([blend_factor, blend_factor, blend_factor])
        
        result = image * (1.0 - blend_factor) + blurred * blend_factor
        return result.astype(np.uint8)
    
    def _enhance_contrast(self, image, strength):
        """Enhance global contrast"""
        if strength <= 0:
            return image
            
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to lightness channel
        clahe = cv2.createCLAHE(clipLimit=2.0+strength*2, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge channels
        enhanced_lab = cv2.merge((cl, a, b))
        
        # Convert back to BGR
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Blend with original based on strength
        alpha = min(1.0, strength)
        return cv2.addWeighted(image, 1-alpha, enhanced_bgr, alpha, 0)
    
    def _enhance_local_contrast(self, image, strength, detail_mask):
        """Enhance local contrast with detail awareness"""
        if strength <= 0:
            return image
            
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE with adaptive clip limit
        clip_limit = 2.0 + strength * 3.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Create enhanced image
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Blend based on detail mask and strength
        alpha = np.clip(detail_mask * strength, 0, 1)
        alpha = np.dstack([alpha, alpha, alpha])
        
        result = image * (1.0 - alpha) + enhanced_bgr * alpha
        return result.astype(np.uint8)
    
    def _apply_sharpening(self, image, strength):
        """Apply basic unsharp mask sharpening"""
        if strength <= 0:
            return image
            
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        return sharpened
    
    def _apply_adaptive_sharpening(self, image, strength, detail_mask):
        """Apply sharpening with detail-aware mask"""
        if strength <= 0:
            return image
            
        # Create sharpen kernel
        kernel = np.array([[-1, -1, -1], 
                           [-1,  9, -1], 
                           [-1, -1, -1]], dtype=np.float32)
                           
        # Apply filter
        sharpened = cv2.filter2D(image, -1, kernel * strength)
        
        # Blend based on detail mask
        alpha = np.clip(detail_mask * strength, 0, 1)
        alpha = np.dstack([alpha, alpha, alpha])
        
        result = image * (1.0 - alpha) + sharpened * alpha
        return result.astype(np.uint8)
    
    def _create_detail_mask(self, image, detail_level):
        """
        Create a mask highlighting detailed areas in the image
        Higher values in the mask indicate more detail
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian to find edges/details
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        
        # Take absolute value and normalize
        lap = np.abs(lap)
        lap_norm = cv2.normalize(lap, None, 0, 1, cv2.NORM_MINMAX)
        
        # Apply different thresholds based on detail preservation level
        if detail_level == 'low':
            threshold = 0.7
        elif detail_level == 'medium':
            threshold = 0.5
        elif detail_level == 'high':
            threshold = 0.3
        else:  # very_high
            threshold = 0.1
        
        # Create binary mask
        mask = lap_norm > threshold
        
        # Dilate to cover detail areas better
        kernel_size = 3
        if detail_level == 'high' or detail_level == 'very_high':
            kernel_size = 5
            
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        
        # Blur the mask for smoother transitions
        mask = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 0)
        
        return mask
    
    def _enhance_dark_areas(self, image, strength):
        """Enhance details in dark areas"""
        if strength <= 0:
            return image
            
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Create mask for dark areas
        dark_mask = np.clip(1.0 - l.astype(np.float32) / 128.0, 0, 1)
        
        # Apply CLAHE to lightness channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Create enhanced image
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Blend based on dark mask and strength
        alpha = dark_mask * strength
        alpha = np.clip(alpha, 0, 1)
        alpha = np.dstack([alpha, alpha, alpha])
        
        result = image * (1.0 - alpha) + enhanced_bgr * alpha
        return result.astype(np.uint8)
    
    def _protect_light_areas(self, image, strength):
        """Protect details in bright areas"""
        if strength <= 0:
            return image
            
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Create mask for light areas
        light_mask = np.clip(l.astype(np.float32) / 128.0, 0, 1)
        
        # Find edges in light areas
        edges = cv2.Canny(l, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        
        # Combine with light mask
        light_detail_mask = light_mask * (edges.astype(np.float32) / 255.0)
        
        # Sharpen light details
        kernel = np.array([[-1, -1, -1], 
                          [-1,  9, -1], 
                          [-1, -1, -1]], dtype=np.float32)
                           
        # Apply filter
        sharpened = cv2.filter2D(image, -1, kernel * 0.5)
        
        # Blend based on light detail mask and strength
        alpha = light_detail_mask * strength
        alpha = np.clip(alpha, 0, 1)
        alpha = np.dstack([alpha, alpha, alpha])
        
        result = image * (1.0 - alpha) + sharpened * alpha
        return result.astype(np.uint8)