import cv2
import numpy as np
import logging

# Setup logging
logger = logging.getLogger('pbn-app')

class PetPatternPreserver:
    """
    Specialized class to detect and preserve important pet patterns and markings
    especially thin dark lines and textures that might otherwise be lost.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('pbn-app')
        self.min_pattern_size = 5  # Minimum size of pattern features to preserve
        self.pattern_contrast_threshold = 40  # Minimum contrast for pattern detection
        self.logger.info("PetPatternPreserver initialized")
    
    def detect_and_preserve(self, image, mask=None):
        """
        Detect important pet patterns and create a preservation mask
        
        Args:
            image: Input BGR or RGB image
            mask: Optional existing mask to enhance
            
        Returns:
            pattern_mask: Binary mask highlighting important pattern areas
            enhanced_image: Image with patterns enhanced for better segmentation
        """
        self.logger.info("Detecting and preserving pet patterns")
        
        # Convert to various color spaces for better detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Create feature importance mask
        feature_mask = np.zeros_like(gray, dtype=np.float32)
        
        # 1. Detect eyes - try multiple approaches
        eyes = []
        # Try Haar cascade first
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        detected_eyes = eye_cascade.detectMultiScale(gray, 1.1, 3, minSize=(15, 15))
        if len(detected_eyes) > 0:
            eyes = detected_eyes
        else:
            # Fallback: Look for dark circular regions
            v_channel = hsv[:,:,2]
            dark_mask = v_channel < 60
            # Find contours in dark regions
            dark_binary = dark_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(dark_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 500:  # Size filter for eyes
                    x, y, w, h = cv2.boundingRect(contour)
                    if 0.5 < w/h < 2.0:  # Eyes should be roughly circular
                        eyes.append((x, y, w, h))
        
        # 2. Mark eyes with higher importance in feature mask
        for (x, y, w, h) in eyes:
            # Create a circular mask around the eye (enlarged for better coverage)
            cv2.circle(feature_mask, (x + w//2, y + h//2), int(max(w, h) * 1.5), 1.0, -1)
        
        # Convert to grayscale for pattern analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        h, w = gray.shape[:2]
        
        # Initialize mask if none provided
        if mask is None:
            mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create pattern mask
        try:
            pattern_mask = self._detect_stripe_patterns(gray)
            pattern_mask = cv2.bitwise_or(pattern_mask, self._detect_spot_patterns(gray))
            
            # Enhance dark markings in the image
            enhanced_image = self._enhance_dark_markings(image, pattern_mask)
            
            # Combine with input mask if provided
            final_mask = cv2.bitwise_or(mask, pattern_mask)
            
            # Log pattern detection results
            pattern_ratio = np.sum(pattern_mask > 0) / (h * w)
            self.logger.info(f"Detected pet patterns covering {pattern_ratio:.2%} of image")
        except Exception as e:
            self.logger.error(f"Error in pattern detection: {e}")
            # Return original image and empty mask if error
            return mask, image
        
        return final_mask, enhanced_image
    
    def _detect_stripe_patterns(self, gray):
        """Detect stripe/tabby patterns using oriented filters"""
        mask = np.zeros_like(gray)
        
        # Apply multiple directional Gabor filters to detect stripes at different orientations
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            try:
                kernel = cv2.getGaborKernel((15, 15), 2.0, theta, 9.0, 0.8, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                
                # Threshold to get pattern regions
                local_thresh = cv2.adaptiveThreshold(
                    filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
                
                # Combine with the accumulating mask
                mask = cv2.bitwise_or(mask, local_thresh)
            except Exception as e:
                self.logger.error(f"Error in stripe pattern detection: {e}")
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _detect_spot_patterns(self, gray):
        """Detect spot patterns like those on tabby cats"""
        try:
            # Create local contrast map
            blur = cv2.GaussianBlur(gray, (15, 15), 0)
            contrast = cv2.absdiff(gray, blur)
            
            # Threshold contrast map to get high-contrast areas
            _, spots = cv2.threshold(contrast, self.pattern_contrast_threshold, 255, cv2.THRESH_BINARY)
            
            # Clean up the mask with morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            spots = cv2.morphologyEx(spots, cv2.MORPH_OPEN, kernel)
            spots = cv2.morphologyEx(spots, cv2.MORPH_CLOSE, kernel)
            
            return spots
        except Exception as e:
            self.logger.error(f"Error in spot pattern detection: {e}")
            return np.zeros_like(gray)
    
    def _enhance_dark_markings(self, image, pattern_mask):
        """Enhance dark markings in the image for better preservation"""
        try:
            # Create a copy of the image for enhancement
            enhanced = image.copy()
            
            # Convert to LAB color space for better luminance manipulation
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Create mask of dark areas that are part of patterns
            dark_mask = (l < 100).astype(np.uint8) * 255
            dark_pattern = cv2.bitwise_and(dark_mask, pattern_mask)
            
            # Dilate dark pattern slightly to ensure coverage
            kernel = np.ones((3, 3), np.uint8)
            dark_pattern = cv2.dilate(dark_pattern, kernel, iterations=1)
            
            # Create a mapping mask for the enhancement
            mapping_mask = cv2.GaussianBlur(dark_pattern.astype(float) / 255.0, (5, 5), 0)
            mapping_mask = np.stack([mapping_mask] * 3, axis=2)  # 3-channel mask
            
            # Increase local contrast in pattern areas
            sharp_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(image, -1, sharp_kernel)
            
            # Blend original with sharpened based on the mapping mask
            enhanced = (image * (1 - mapping_mask) + sharpened * mapping_mask).astype(np.uint8)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error enhancing dark markings: {e}")
            return image.copy()  # Return original if enhancement fails

    def _enhance_eye_regions(self, image, eye_regions):
        """Special processing for eye regions to maintain clarity"""
        enhanced = image.copy()
        
        for (x, y, w, h) in eye_regions:
            # Extract eye region with padding
            pad = 5
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(image.shape[1], x+w+pad), min(image.shape[0], y+h+pad)
            
            eye_region = image[y1:y2, x1:x2]
            
            # Special processing for eyes - increase contrast
            if eye_region.size > 0:
                # Convert to LAB color space
                lab = cv2.cvtColor(eye_region, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # CLAHE on L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
                cl = clahe.apply(l)
                
                # Merge channels
                enhanced_lab = cv2.merge((cl, a, b))
                enhanced_eye = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
                
                # Apply Gaussian blur to reduce noise
                enhanced_eye = cv2.GaussianBlur(enhanced_eye, (3, 3), 0)
                
                # Place back into the image
                enhanced[y1:y2, x1:x2] = enhanced_eye
        
        return enhanced

    def detect_pet_patterns(self, image):
        # Current code detects patterns but doesn't emphasize key features
        
        # Add this code to enhance detection of eyes and facial features
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect eye regions with more aggressive parameters
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 3)
        
        # Create a mask for important features
        feature_mask = np.zeros_like(gray)
        
        # If faces are detected, focus on those regions
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Mark face region
                feature_mask[y:y+h, x:x+w] = 150
                
                # Detect eyes within face region
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                
                # Mark eye regions with higher importance
                for (ex, ey, ew, eh) in eyes:
                    cv2.circle(feature_mask, (x+ex+ew//2, y+ey+eh//2), max(ew, eh), 255, -1)
        else:
            # If no faces detected, use contrast to find potential eye regions
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Find potential eye regions using contrast
            _, thresh = cv2.threshold(enhanced, 120, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for circular-ish contours that might be eyes
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 500:  # Size filter for eye regions
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    radius = int(radius * 1.5)  # Expand slightly
                    cv2.circle(feature_mask, center, radius, 255, -1)
        
        return feature_mask

    def process_pet_image(self, image):
        """
        Process a pet image to enhance key features and create a hierarchical outline
        
        Args:
            image: RGB or BGR pet image
            
        Returns:
            processed_image: Enhanced image with preserved features
            feature_mask: Importance mask highlighting key features
        """
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image[0,0,0] == image[0,0,2] else image
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Create grayscale for processing
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        
        # 1. Extract pet from background
        try:
            # Use GrabCut for background removal
            mask = np.zeros(gray.shape, dtype=np.uint8)
            
            # Initialize with likely foreground (middle of image)
            h, w = gray.shape
            center_y, center_x = h // 2, w // 2
            size_y, size_x = int(h * 0.6), int(w * 0.6)
            
            # Mark center region as probable foreground
            y1, y2 = max(0, center_y - size_y//2), min(h, center_y + size_y//2)
            x1, x2 = max(0, center_x - size_x//2), min(w, center_x + size_x//2)
            
            # Initialize mask: 0=background, 1=foreground, 2=probable background, 3=probable foreground
            mask[y1:y2, x1:x2] = 3  # Probable foreground
            mask_copy = mask.copy()
            
            # Run GrabCut
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            cv2.grabCut(rgb_image, mask_copy, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
            
            # Create binary mask where foreground = 1
            pet_mask = np.where((mask_copy == 1) | (mask_copy == 3), 255, 0).astype('uint8')
            
            # Clean up mask with morphological operations
            kernel = np.ones((5, 5), np.uint8)
            pet_mask = cv2.morphologyEx(pet_mask, cv2.MORPH_CLOSE, kernel)
            pet_mask = cv2.morphologyEx(pet_mask, cv2.MORPH_OPEN, kernel)
        except Exception as e:
            # Fallback to basic thresholding if GrabCut fails
            self.logger.error(f"Background extraction error: {e}, using fallback")
            _, pet_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2. Detect pet features (eyes, nose, ears)
        feature_mask = np.zeros_like(gray, dtype=np.float32)
        
        # Try to detect eyes
        try:
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)
            
            # If eyes detected, mark them in the feature mask
            for (x, y, w, h) in eyes:
                cv2.circle(feature_mask, (x + w//2, y + h//2), int(max(w, h) * 1.2), 1.0, -1)
                # Also mark the eye regions with higher value for better preservation
                feature_mask[y:y+h, x:x+w] = 1.0
            
            # In pet_pattern_preserver.py - enhance eye detection
            # After the eye detection step:
            if len(eyes) == 0:
                # If regular detection fails, look for dark circular regions
                hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
                v_channel = hsv[:,:,2]
                
                # Look for dark spots that could be eyes
                dark_areas = (v_channel < 60).astype(np.uint8) * 255
                
                # Find circular dark areas
                contours, _ = cv2.findContours(dark_areas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 50 < area < 500:
                        x, y, w, h = cv2.boundingRect(contour)
                        circularity = 4*np.pi*area/(cv2.arcLength(contour, True)**2) if cv2.arcLength(contour, True) > 0 else 0
                        
                        # Eyes tend to be somewhat circular
                        if circularity > 0.5:
                            cv2.circle(feature_mask, (x+w//2, y+h//2), int(max(w, h)*1.5), 1.0, -1)
        except Exception as e:
            self.logger.error(f"Eye detection error: {e}")
        
        # 3. Create a hierarchy of importance
        # Create pet outline - this will be the strongest line
        kernel = np.ones((7, 7), np.uint8)  # Larger kernel for thicker outline
        dilated_mask = cv2.dilate(pet_mask, kernel)
        outline_mask = dilated_mask - pet_mask
        
        # 4. Create a processed image with enhanced features
        processed_image = rgb_image.copy()
        
        # Enhance the contrast of the image
        lab = cv2.cvtColor(processed_image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l)
        
        # Create enhanced image
        enhanced_lab = cv2.merge((enhanced_l, a, b))
        processed_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # 5. Enhance eye regions specifically
        if 'eyes' in locals() and len(eyes) > 0:
            processed_image = self._enhance_eye_regions(processed_image, eyes)
        
        # 6. Boost the outline in the feature mask
        outline_feature = outline_mask.astype(np.float32) / 255.0
        feature_mask = np.maximum(feature_mask, outline_feature)
        
        # Make sure outline has high importance (1.0 instead of 0.8)
        feature_mask[outline_mask > 0] = 1.0
        
        # 7. Add pattern detection for fur texture
        pattern_mask, _ = self.detect_and_preserve(rgb_image)
        
        # Add pattern information to feature mask with lower importance
        pattern_importance = pattern_mask.astype(np.float32) / 255.0 * 0.3
        feature_mask = np.maximum(feature_mask, pattern_importance)
        
        self.logger.info(f"Pet image processing completed with feature highlighting")
        
        return processed_image, feature_mask

    def detect_pet_eyes(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        eye_importance = np.zeros_like(gray, dtype=np.float32)
        
        # Try multiple methods to detect eyes
        
        # 1. Haar cascade (traditional but sometimes misses pet eyes)
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, 1.05, 3, minSize=(15, 15))
        
        # 2. If Haar fails, try dark circular blob detection
        if len(eyes) < 2:
            # Dark circular regions are often eyes
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
            
            # Find circular dark regions
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, 1, 20,
                param1=50, param2=30, minRadius=5, maxRadius=30)
            
            if circles is not None:
                eyes = []
                circles = np.uint16(np.around(circles[0, :]))
                for (x, y, r) in circles:
                    # Filter by darkness (eyes are usually darker)
                    if np.mean(gray[y-r:y+r, x-r:x+r]) < 120:
                        eyes.append((x-r, y-r, 2*r, 2*r))
        
        # Mark eyes in the importance mask
        for (x, y, w, h) in eyes:
            cv2.circle(eye_importance, (x + w//2, y + h//2), int(max(w, h) * 1.5), 1.0, -1)
        
        return eye_importance

