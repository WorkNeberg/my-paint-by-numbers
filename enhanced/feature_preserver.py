import cv2
import numpy as np
import logging

# Set up logger
logger = logging.getLogger('pbn-app')

class FeaturePreserver:
    """
    Detects and preserves important features in images for paint-by-numbers generation.
    """
    
    def __init__(self):
        # Initialize detection parameters
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    def create_feature_mask(self, image, image_type=None):
        """
        Create a mask of important features to preserve during processing
        
        Args:
            image: Input image
            image_type: Type of image for specialized processing
            
        Returns:
            feature_mask: Mask of important features
            feature_regions: Dictionary of detected feature regions
        """
        # Convert image to proper format if needed
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not read image: {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Check if image is already RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = image
        else:
            # Convert grayscale to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Create grayscale version
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        
        # Initialize mask and feature regions
        h, w = gray.shape
        feature_mask = np.zeros((h, w), dtype=np.uint8)
        feature_regions = {}
        
        # Apply type-specific detection
        if image_type == 'portrait':
            feature_mask, feature_regions = self._detect_portrait_features(rgb_image, gray, feature_mask)
        elif image_type == 'pet':
            feature_mask, feature_regions = self._detect_pet_features(rgb_image, gray, feature_mask)
        elif image_type == 'landscape':
            feature_mask, feature_regions = self._detect_landscape_features(rgb_image, gray, feature_mask)
        else:
            # Generic feature detection
            feature_mask, feature_regions = self._detect_generic_features(rgb_image, gray, feature_mask)
        
        return feature_mask, feature_regions
    
    def _detect_portrait_features(self, rgb_image, gray, mask):
        """Detect important features in portrait images"""
        features = {}
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        features['faces'] = faces
        
        # For each face, detect eyes
        eyes = []
        for (x, y, w, h) in faces:
            # Mark face region with medium importance
            cv2.rectangle(mask, (x, y), (x+w, y+h), 150, -1)
            
            # Get face region
            roi_gray = gray[y:y+h, x:x+w]
            
            # Detect eyes
            face_eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in face_eyes:
                # Convert to global coordinates
                eye_x = x + ex
                eye_y = y + ey
                eyes.append((eye_x, eye_y, ew, eh))
                
                # Mark eyes with high importance
                cv2.circle(mask, (eye_x + ew//2, eye_y + eh//2), max(ew, eh)//2, 255, -1)
        
        features['eyes'] = eyes
        
        return mask, features
    
    def _detect_pet_features(self, rgb_image, gray, mask):
        """Improved pet feature detection with focus on eyes and face"""
        features = {}
        
        # Use more sensitive parameters for pet eyes
        eyes = self.eye_cascade.detectMultiScale(gray, 1.05, 2, minSize=(10, 10))
        
        # If standard detection fails, try additional methods
        if len(eyes) == 0:
            # Try different parameters
            eyes = self.eye_cascade.detectMultiScale(gray, 1.02, 1)
        
        # Mark eyes with higher importance and larger area
        for (x, y, w, h) in eyes:
            # Increase detected area by 50%
            eye_radius = int(max(w, h) * 1.5)
            cv2.circle(mask, (x + w//2, y + h//2), eye_radius, 255, -1)
        
        # Detect nose region using color and position
        center_area = gray[int(gray.shape[0]*0.4):int(gray.shape[0]*0.7), 
                           int(gray.shape[1]*0.3):int(gray.shape[1]*0.7)]
        # Mark center area with medium importance
        mask[int(gray.shape[0]*0.4):int(gray.shape[0]*0.7), 
             int(gray.shape[1]*0.3):int(gray.shape[1]*0.7)] = 150
        
        return mask, features
    
    def _detect_generic_features(self, rgb_image, gray, feature_mask=None):
        """Detect generic features in any image"""
        logger = logging.getLogger('pbn-app')
        logger.info("DETECT_GENERIC_FEATURES METHOD CALLED")
        
        h, w = gray.shape
        
        if feature_mask is None:
            feature_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Simple implementation - detect edges
        edges = cv2.Canny(gray, 50, 150)
        feature_mask = cv2.bitwise_or(feature_mask, edges)
        
        # Return minimal results
        feature_regions = {'edges': []}
        
        return feature_mask, feature_regions