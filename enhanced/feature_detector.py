import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List

class FeatureDetector:
    """Advanced feature detection with confidence scoring and importance mapping"""
    
    def __init__(self):
        self.logger = logging.getLogger('pbn-app.feature_detector')
        
        # Initialize feature detection cascades
        try:
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            self.cat_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')
            self.pet_face_cascade = self.cat_face_cascade  # For consistent naming
            self.models_loaded = True  # Mark models as loaded
            self.logger.info("Feature detection cascades initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing cascades: {str(e)}")
            self.models_loaded = False  # Mark models as not loaded
    
    def detect_and_create_importance_map(self, image: np.ndarray, settings: Dict[str, Any], image_type: str) -> np.ndarray:
        """
        Detect features and create an importance map
        
        Args:
            image: Input RGB image
            settings: Dictionary of feature detection settings
            image_type: Type of image ('pet', 'portrait', 'landscape', 'general')
            
        Returns:
            Importance map (float32 0.0-1.0)
        """
        self.logger.info(f"Creating feature importance map for {image_type}")
        h, w = image.shape[:2]
        
        # Initialize empty importance map (0.0 = unimportant, 1.0 = very important)
        importance_map = np.zeros((h, w), dtype=np.float32)
        
        # Check if feature detection is enabled
        feature_detection = settings.get('feature_detection', {})
        if not feature_detection.get('feature_detection_enabled', True):
            self.logger.info("Feature detection disabled in settings")
            return importance_map
        
        # Prepare grayscale image for detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 1. Add base importance from image details
        importance_map = self._add_detail_importance(image, gray, importance_map)
        
        # 2. Apply image type specific feature detection
        if image_type == 'pet':
            importance_map = self._detect_pet_features(image, gray, importance_map, feature_detection)
        elif image_type == 'portrait':
            importance_map = self._detect_portrait_features(image, gray, importance_map, feature_detection)
        elif image_type == 'landscape':
            importance_map = self._detect_landscape_features(image, gray, importance_map, feature_detection)
        
        # 3. Normalize importance map to 0.0-1.0 range
        if np.max(importance_map) > 0:
            importance_map = importance_map / np.max(importance_map)
        
        # 4. Apply global importance factor
        feature_importance = feature_detection.get('feature_importance', 1.0)
        if feature_importance != 1.0:
            # Scale non-zero values by importance factor
            mask = importance_map > 0
            importance_map[mask] = importance_map[mask] * feature_importance
            importance_map = np.clip(importance_map, 0.0, 1.0)
        
        self.logger.info(f"Feature importance map created successfully")
        return importance_map
    
    def _add_detail_importance(self, image: np.ndarray, gray: np.ndarray, importance_map: np.ndarray) -> np.ndarray:
        """Add base importance from image details"""
        # Detect edges using Sobel operator
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_magnitude = cv2.magnitude(sobel_x, sobel_y)
        
        # Normalize edge magnitude
        if np.max(edge_magnitude) > 0:
            edge_magnitude = edge_magnitude / np.max(edge_magnitude) * 0.3  # Max importance 0.3 for edges
        
        # Add edge importance to map
        importance_map = np.maximum(importance_map, edge_magnitude)
        
        # Add importance based on local contrast
        local_std = cv2.GaussianBlur(gray, (5, 5), 0)
        local_std = cv2.GaussianBlur(np.abs(gray - local_std), (21, 21), 0) / 255.0 * 0.2
        importance_map = np.maximum(importance_map, local_std)
        
        return importance_map
    
    def _detect_pet_features(self, image: np.ndarray, gray: np.ndarray, importance_map: np.ndarray, 
                            settings: Dict[str, Any]) -> np.ndarray:
        """Detect and prioritize pet features"""
        eye_sensitivity = settings.get('eye_detection_sensitivity', 'medium')
        face_mode = settings.get('face_detection_mode', 'basic')
        protection_radius = settings.get('feature_protection_radius', 5)
        
        # Convert sensitivity to scale factors (higher = more sensitive)
        scale_factors = {
            'low': 1.3,
            'medium': 1.2,
            'high': 1.1
        }.get(eye_sensitivity, 1.2)
        
        # Convert detection mode to min neighbors (lower = more detections but more false positives)
        min_neighbors = {
            'basic': 5,
            'enhanced': 3,
            'ML': 2
        }.get(face_mode, 3)
        
        features_detected = False
        
        # Try cat face detection first
        if self.cat_face_cascade is not None:
            try:
                cat_faces = self.cat_face_cascade.detectMultiScale(gray, scale_factors, min_neighbors)
                
                for (x, y, w, h) in cat_faces:
                    features_detected = True
                    
                    # Create importance gradient for the face (highest in center, fading out)
                    y_coords, x_coords = np.mgrid[y:y+h, x:x+w]
                    face_center_y, face_center_x = y + h/2, x + w/2
                    
                    # Calculate distance from center (normalized)
                    dist = np.sqrt(((x_coords - face_center_x)/(w/2))**2 + 
                                  ((y_coords - face_center_y)/(h/2))**2)
                    
                    # Convert to importance (1.0 at center, fading to 0.4 at edges)
                    face_importance = np.clip(1.0 - dist*0.6, 0.4, 1.0)
                    
                    # Apply to importance map
                    importance_map[y:y+h, x:x+w] = np.maximum(
                        importance_map[y:y+h, x:x+w],
                        face_importance
                    )
                    
                    # Estimate eye positions
                    eye_y = y + int(h * 0.4)
                    left_eye_x = x + int(w * 0.3)
                    right_eye_x = x + int(w * 0.7)
                    eye_size = max(int(h * 0.15), 5)
                    
                    # Add high importance for eyes
                    for eye_x in [left_eye_x, right_eye_x]:
                        cv2.circle(
                            importance_map, 
                            (eye_x, eye_y), 
                            eye_size, 
                            1.0, 
                            -1  # Filled circle
                        )
            except Exception as e:
                self.logger.warning(f"Error in cat face detection: {str(e)}")
        
        # If no cat faces found, try general eye detection
        if not features_detected:
            try:
                # Use different parameters based on sensitivity
                eyes = self.eye_cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factors,
                    minNeighbors=min_neighbors,
                    minSize=(10, 10)
                )
                
                # If standard detection fails with high sensitivity, try a more aggressive approach
                if len(eyes) == 0 and eye_sensitivity == 'high':
                    eyes = self.eye_cascade.detectMultiScale(gray, 1.05, 1, minSize=(5, 5))
                
                for (x, y, w, h) in eyes:
                    features_detected = True
                    
                    # Adjust protection radius based on settings
                    eye_radius = max(w, h) * (0.5 + protection_radius / 10.0)
                    
                    # Create circular gradient of importance (1.0 at center, fading outward)
                    center_x, center_y = x + w//2, y + h//2
                    y_coords, x_coords = np.mgrid[
                        max(0, center_y - int(eye_radius)):min(gray.shape[0], center_y + int(eye_radius)),
                        max(0, center_x - int(eye_radius)):min(gray.shape[1], center_x + int(eye_radius))
                    ]
                    
                    # Calculate normalized distance from center
                    dist = np.sqrt(((x_coords - center_x)/eye_radius)**2 + ((y_coords - center_y)/eye_radius)**2)
                    
                    # Convert to importance (1.0 at center, fading to 0)
                    eye_importance = np.clip(1.0 - dist, 0, 1.0)
                    
                    # Apply to importance map
                    region_y_start = max(0, center_y - int(eye_radius))
                    region_y_end = min(gray.shape[0], center_y + int(eye_radius))
                    region_x_start = max(0, center_x - int(eye_radius))
                    region_x_end = min(gray.shape[1], center_x + int(eye_radius))
                    
                    importance_map[region_y_start:region_y_end, region_x_start:region_x_end] = np.maximum(
                        importance_map[region_y_start:region_y_end, region_x_start:region_x_end],
                        eye_importance
                    )
            except Exception as e:
                self.logger.warning(f"Error in pet eye detection: {str(e)}")
        
        # If still no features, try using color/intensity to detect pet features
        if not features_detected and eye_sensitivity != 'low':
            try:
                # Detect dark regions that might be eyes
                v_channel = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:,:,2]
                dark_regions = v_channel < 60
                
                if np.any(dark_regions):
                    dark_mask = dark_regions.astype(np.uint8) * 255
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
                    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
                    
                    # Find contours in dark regions
                    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if 10 < area < 500:  # Size filter for possible eyes
                            x, y, w, h = cv2.boundingRect(contour)
                            aspect_ratio = float(w) / h
                            
                            # Eyes typically have aspect ratio close to 1
                            if 0.5 < aspect_ratio < 2.0:
                                features_detected = True
                                
                                # Add high importance for potential eye
                                cv2.circle(
                                    importance_map, 
                                    (x + w//2, y + h//2), 
                                    int(max(w, h) * 1.5), 
                                    0.8, 
                                    -1
                                )
            except Exception as e:
                self.logger.warning(f"Error in pet dark region detection: {str(e)}")
                
        # If still no features detected, add general importance to face region
        if not features_detected:
            # Add medium importance to the center-top area where a pet face likely is
            h, w = gray.shape
            center_area = np.zeros((h, w), dtype=np.float32)
            center_y_start = int(h * 0.3)
            center_y_end = int(h * 0.7)
            center_x_start = int(w * 0.25)
            center_x_end = int(w * 0.75)
            
            # Create gradient of importance (highest in center)
            y_coords, x_coords = np.mgrid[center_y_start:center_y_end, center_x_start:center_x_end]
            center_y = h / 2
            center_x = w / 2
            
            dist = np.sqrt(((x_coords - center_x)/(w/4))**2 + ((y_coords - center_y)/(h/4))**2)
            center_importance = np.clip(0.5 - dist*0.3, 0.2, 0.5)
            
            center_area[center_y_start:center_y_end, center_x_start:center_x_end] = center_importance
            importance_map = np.maximum(importance_map, center_area)
        
        return importance_map
    
    def _detect_portrait_features(self, image: np.ndarray, gray: np.ndarray, importance_map: np.ndarray, 
                                settings: Dict[str, Any]) -> np.ndarray:
        """Detect and prioritize portrait features"""
        eye_sensitivity = settings.get('eye_detection_sensitivity', 'medium')
        face_mode = settings.get('face_detection_mode', 'basic')
        protection_radius = settings.get('feature_protection_radius', 5)
        preserve_expressions = settings.get('preserve_expressions', True)
        
        # Convert sensitivity to scale factors (higher = more sensitive)
        scale_factors = {
            'low': 1.3,
            'medium': 1.2,
            'high': 1.1
        }.get(eye_sensitivity, 1.2)
        
        # Convert detection mode to min neighbors (lower = more detections)
        min_neighbors = {
            'basic': 5,
            'enhanced': 3,
            'ML': 2
        }.get(face_mode, 3)
        
        features_detected = False
        
        # 1. Face detection
        try:
            # Try frontal faces first
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factors,
                minNeighbors=min_neighbors
            )
            
            # If no frontal faces, try profile faces
            if len(faces) == 0 and face_mode != 'basic':
                faces = self.profile_cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factors,
                    minNeighbors=min_neighbors
                )
                
            for (x, y, w, h) in faces:
                features_detected = True
                
                # Create importance gradient for the face (highest in center, fading out)
                y_coords, x_coords = np.mgrid[y:y+h, x:x+w]
                face_center_y, face_center_x = y + h/2, x + w/2
                
                # Calculate distance from center (normalized)
                dist = np.sqrt(((x_coords - face_center_x)/(w/2))**2 + 
                              ((y_coords - face_center_y)/(h/2))**2)
                
                # Convert to importance (1.0 at center, fading to 0.4 at edges)
                face_importance = np.clip(1.0 - dist*0.6, 0.4, 1.0)
                
                # Apply to importance map
                importance_map[y:y+h, x:x+w] = np.maximum(
                    importance_map[y:y+h, x:x+w],
                    face_importance
                )
                
                # 2. Eye detection within face
                face_roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(
                    face_roi_gray,
                    scaleFactor=scale_factors,
                    minNeighbors=min_neighbors
                )
                
                for (ex, ey, ew, eh) in eyes:
                    # Convert to global coordinates
                    global_ex = x + ex
                    global_ey = y + ey
                    
                    # Adjust protection radius based on settings
                    eye_radius = max(ew, eh) * (0.5 + protection_radius / 10.0)
                    
                    # Create circular gradient of importance (1.0 at center, fading outward)
                    center_x, center_y = global_ex + ew//2, global_ey + eh//2
                    y_coords, x_coords = np.mgrid[
                        max(0, center_y - int(eye_radius)):min(gray.shape[0], center_y + int(eye_radius)),
                        max(0, center_x - int(eye_radius)):min(gray.shape[1], center_x + int(eye_radius))
                    ]
                    
                    # Calculate normalized distance from center
                    dist = np.sqrt(((x_coords - center_x)/eye_radius)**2 + ((y_coords - center_y)/eye_radius)**2)
                    
                    # Convert to importance (1.0 at center, fading to 0)
                    eye_importance = np.clip(1.0 - dist, 0, 1.0)
                    
                    # Apply to importance map
                    region_y_start = max(0, center_y - int(eye_radius))
                    region_y_end = min(gray.shape[0], center_y + int(eye_radius))
                    region_x_start = max(0, center_x - int(eye_radius))
                    region_x_end = min(gray.shape[1], center_x + int(eye_radius))
                    
                    importance_map[region_y_start:region_y_end, region_x_start:region_x_end] = np.maximum(
                        importance_map[region_y_start:region_y_end, region_x_start:region_x_end],
                        eye_importance
                    )
                
                # 3. Add importance to mouth region if preserve_expressions is enabled
                if preserve_expressions:
                    mouth_y = y + int(h * 0.7)  # Mouth is typically in the lower third of the face
                    mouth_height = int(h * 0.2)
                    mouth_y_end = min(mouth_y + mouth_height, gray.shape[0])
                    
                    # Create importance gradient for the mouth region
                    y_coords, x_coords = np.mgrid[mouth_y:mouth_y_end, x:x+w]
                    mouth_center_y = mouth_y + mouth_height/2
                    mouth_center_x = x + w/2
                    
                    # Calculate distance from mouth center (normalized)
                    dist = np.sqrt(((x_coords - mouth_center_x)/(w/2))**2 + 
                                  ((y_coords - mouth_center_y)/(mouth_height/2))**2)
                    
                    # Convert to importance (0.8 at center, fading to 0.4 at edges)
                    mouth_importance = np.clip(0.8 - dist*0.4, 0.4, 0.8)
                    
                    # Apply to importance map
                    importance_map[mouth_y:mouth_y_end, x:x+w] = np.maximum(
                        importance_map[mouth_y:mouth_y_end, x:x+w],
                        mouth_importance
                    )
        except Exception as e:
            self.logger.warning(f"Error in portrait feature detection: {str(e)}")
                
        # If no faces detected, try general approach for possible face area
        if not features_detected:
            # Add medium importance to the center area where a face likely is
            h, w = gray.shape
            center_area = np.zeros((h, w), dtype=np.float32)
            center_y_start = int(h * 0.2)
            center_y_end = int(h * 0.8)
            center_x_start = int(w * 0.25)
            center_x_end = int(w * 0.75)
            
            # Create gradient of importance (highest in center)
            y_coords, x_coords = np.mgrid[center_y_start:center_y_end, center_x_start:center_x_end]
            center_y = h / 2
            center_x = w / 2
            
            dist = np.sqrt(((x_coords - center_x)/(w/4))**2 + ((y_coords - center_y)/(h/4))**2)
            center_importance = np.clip(0.5 - dist*0.3, 0.2, 0.5)
            
            center_area[center_y_start:center_y_end, center_x_start:center_x_end] = center_importance
            importance_map = np.maximum(importance_map, center_area)
        
        return importance_map
    
    def _detect_landscape_features(self, image: np.ndarray, gray: np.ndarray, importance_map: np.ndarray, 
                                settings: Dict[str, Any]) -> np.ndarray:
        """Detect and prioritize landscape features"""
        # For landscapes, we mainly care about:
        # 1. Horizon lines
        # 2. Strong structural elements
        # 3. Focal points
        
        try:
            # Detect strong edges that might be horizon or structure
            edges = cv2.Canny(gray, 50, 150)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=50, maxLineGap=10)
            
            # Add importance to detected lines
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(importance_map, (x1, y1), (x2, y2), 0.5, 3)
            
            # Add importance to detected focal points
            h, w = gray.shape
            
            # Convert to HSV to help identify key areas
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Add importance to sky regions (typically at the top)
            sky_region = np.zeros((h, w), dtype=np.float32)
            
            # Simple sky detection (upper part of the image with primarily blue hue)
            upper_half = hsv[:h//2, :, :]
            blue_hue_mask = (upper_half[:, :, 0] > 100) & (upper_half[:, :, 0] < 140)
            
            if np.mean(blue_hue_mask) > 0.3:  # If significant blue detected in upper half
                gradient = np.zeros((h//2, w), dtype=np.float32)
                
                # Create vertical gradient (0.4 at top, fading to 0.1)
                for y in range(h//2):
                    gradient[y, :] = 0.4 - (y / (h//2)) * 0.3
                
                sky_region[:h//2, :][blue_hue_mask] = gradient[blue_hue_mask]
                
                importance_map = np.maximum(importance_map, sky_region)
            
            # Detect potentially interesting focal points using saturation variance
            sat_blurred = cv2.GaussianBlur(hsv[:,:,1], (21, 21), 0)
            sat_variance = cv2.GaussianBlur((hsv[:,:,1].astype(np.float32) - sat_blurred)**2, (21, 21), 0)
            
            # Normalize variance to 0-0.6 range for importance
            if np.max(sat_variance) > 0:
                sat_variance = sat_variance / np.max(sat_variance) * 0.6
                importance_map = np.maximum(importance_map, sat_variance)
            
        except Exception as e:
            self.logger.warning(f"Error in landscape feature detection: {str(e)}")
        
        return importance_map
    def load_models(self):
        """Load or reload detection models"""
        try:
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            self.cat_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')
            self.pet_face_cascade = self.cat_face_cascade  # For consistent naming
            self.models_loaded = True
            self.logger.info("Detection models loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading detection models: {e}")
            self.models_loaded = False
            raise  # Re-raise the exception to be caught by caller
    def detect_type(self, image):
        """
        Detect the type of image (portrait, pet, landscape, etc.)
        
        Args:
            image: Input image
            
        Returns:
            String indicating image type
        """
        # For now, we'll use a simplified detection approach
        # In a full implementation, this would use more sophisticated detection
        
        # Try to load models if needed
        if not hasattr(self, 'models_loaded') or not self.models_loaded:
            try:
                self.load_models()
            except:
                self.logger.warning("Could not load detection models, using generic type")
                return "generic"
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Look for human faces
        faces = []
        try:
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        except Exception as e:
            self.logger.warning(f"Error detecting faces: {e}")
            
        # Look for pet faces
        pet_faces = []
        try:
            pet_faces = self.pet_face_cascade.detectMultiScale(gray, 1.1, 5)
        except Exception as e:
            self.logger.warning(f"Error detecting pet faces: {e}")
        
        # Determine image type based on detections
        if len(faces) > 0:
            # If faces are proportionally large in the image, it's likely a portrait
            face_area = sum(w*h for (x, y, w, h) in faces)
            image_area = gray.shape[0] * gray.shape[1]
            if face_area > image_area * 0.1:  # Face is at least 10% of image
                return "portrait"
            else:
                return "people"
        elif len(pet_faces) > 0:
            return "pet"
        else:
            # Check if it might be a landscape
            # Simple heuristic: landscape images often have a horizontal composition
            h, w = gray.shape
            if w > h * 1.2:  # Width significantly larger than height
                return "landscape"
            
            # Default to generic type
            return "generic"