import cv2
import numpy as np
import os

class FeaturePreserver:
    def __init__(self):
        # Load pre-trained models for feature detection
        self.basedir = os.path.dirname(os.path.abspath(__file__))
        cascade_dir = cv2.data.haarcascades
        
        self.face_cascade = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_frontalface_default.xml'))
        self.eye_cascade = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_eye.xml'))
        self.smile_cascade = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_smile.xml'))
        
        # For pets
        self.pet_face_cascade = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_frontalface_alt.xml'))
    def detect_and_preserve_features(self, image, image_type):
        """
        Detect and create a map of important features that should be preserved
        
        Parameters:
        - image: Input RGB image
        - image_type: Type of image ('portrait', 'pet', 'landscape', etc.)
        
        Returns:
        - Feature importance map (0-1 float) where higher values indicate more important areas
        """
        h, w = image.shape[:2]
        feature_map = np.zeros((h, w), dtype=np.float32)
        
        # Convert to grayscale for feature detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Base feature detection from edges (important for all image types)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = cv2.GaussianBlur(edges.astype(float) / 255.0, (15, 15), 0)
        feature_map += edge_density * 0.5
        
        # Add local contrast as an indicator of important features
        local_std = np.zeros_like(gray, dtype=np.float32)
        cv2.boxFilter(np.square(gray - cv2.boxFilter(gray, -1, (5, 5))), -1, (5, 5), local_std)
        local_std = np.sqrt(local_std)
        local_std = local_std / np.max(local_std) if np.max(local_std) > 0 else local_std
        feature_map += local_std * 0.3
        
        # Type-specific feature detection
        if image_type == 'portrait':
            try:
                # Detect faces
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w_face, h_face) in faces:
                    # Create face importance mask (higher in center, fading outward)
                    y_grid, x_grid = np.mgrid[y:y+h_face, x:x+w_face]
                    face_center_y, face_center_x = y + h_face/2, x + w_face/2
                    
                    # Calculate distance from center (normalized)
                    dist = np.sqrt(((x_grid - face_center_x) / (w_face/2))**2 + 
                                ((y_grid - face_center_y) / (h_face/2))**2)
                    
                    # Create radial gradient (1.0 at center, fading outward)
                    face_importance = np.clip(1.0 - dist*0.8, 0, 1)
                    
                    # Apply to feature map
                    feature_map[y:y+h_face, x:x+w_face] = np.maximum(
                        feature_map[y:y+h_face, x:x+w_face],
                        face_importance * 0.8
                    )
                    
                    # Detect eyes within the face region
                    face_roi_gray = gray[y:y+h_face, x:x+w_face]
                    eyes = self.eye_cascade.detectMultiScale(face_roi_gray)
                    
                    for (ex, ey, ew, eh) in eyes:
                        # Create circular region of high importance for each eye
                        eye_center_y, eye_center_x = y + ey + eh/2, x + ex + ew/2
                        eye_radius = max(ew, eh) * 0.7
                        
                        # Create radial mask for eye
                        y_grid, x_grid = np.mgrid[0:h, 0:w]
                        dist = np.sqrt((x_grid - eye_center_x)**2 + (y_grid - eye_center_y)**2)
                        eye_mask = np.clip(1.0 - dist/eye_radius, 0, 1)
                        
                        # Apply to feature map
                        feature_map = np.maximum(feature_map, eye_mask * 0.95)
            
            except Exception as e:
                print(f"Warning: Error in portrait feature detection: {e}")
        
        elif image_type == 'pet':
            try:
                # Try to detect pet face
                pet_faces = self.pet_face_cascade.detectMultiScale(gray, 1.1, 3)
                
                if len(pet_faces) > 0:
                    # Pet face detected
                    for (x, y, w_face, h_face) in pet_faces:
                        # Create importance mask for pet face
                        y_grid, x_grid = np.mgrid[y:y+h_face, x:x+w_face]
                        face_center_y, face_center_x = y + h_face/2, x + w_face/2
                        
                        # Calculate distance from center (normalized)
                        dist = np.sqrt(((x_grid - face_center_x) / (w_face/2))**2 + 
                                    ((y_grid - face_center_y) / (h_face/2))**2)
                        
                        # Create radial gradient (1.0 at center, fading outward)
                        face_importance = np.clip(1.0 - dist*0.7, 0, 1)
                        
                        # Apply to feature map
                        feature_map[y:y+h_face, x:x+w_face] = np.maximum(
                            feature_map[y:y+h_face, x:x+w_face],
                            face_importance * 0.8
                        )
                        
                        # Typical eye positions for pets
                        eye_y = y + int(h_face * 0.4)
                        left_eye_x = x + int(w_face * 0.3)
                        right_eye_x = x + int(w_face * 0.7)
                        
                        # Create circular regions for eyes
                        eye_radius = h_face * 0.15
                        
                        # Create radial masks for eyes
                        y_grid, x_grid = np.mgrid[0:h, 0:w]
                        
                        # Left eye
                        left_dist = np.sqrt((x_grid - left_eye_x)**2 + (y_grid - eye_y)**2)
                        left_eye_mask = np.clip(1.0 - left_dist/eye_radius, 0, 1)
                        left_eye_mask[left_eye_mask < 0] = 0
                        feature_map = np.maximum(feature_map, left_eye_mask * 0.95)
                        
                        # Right eye
                        right_dist = np.sqrt((x_grid - right_eye_x)**2 + (y_grid - eye_y)**2)
                        right_eye_mask = np.clip(1.0 - right_dist/eye_radius, 0, 1)
                        right_eye_mask[right_eye_mask < 0] = 0
                        feature_map = np.maximum(feature_map, right_eye_mask * 0.95)
                
                else:
                    # Fallback - use blob detection to find eyes
                    params = cv2.SimpleBlobDetector_Params()
                    params.minThreshold = 10
                    params.maxThreshold = 200
                    params.filterByArea = True
                    params.minArea = 30
                    params.maxArea = 300
                    params.filterByCircularity = True
                    params.minCircularity = 0.5
                    
                    detector = cv2.SimpleBlobDetector_create(params)
                    keypoints = detector.detect(255 - gray)
                    
                    for kp in keypoints:
                        # Create circular region of high importance for potential eyes
                        center_x, center_y = int(kp.pt[0]), int(kp.pt[1])
                        radius = kp.size * 0.8
                        
                        y_grid, x_grid = np.mgrid[0:h, 0:w]
                        dist = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
                        eye_mask = np.clip(1.0 - dist/radius, 0, 1)
                        eye_mask[eye_mask < 0] = 0
                        
                        feature_map = np.maximum(feature_map, eye_mask * 0.95)
            
            except Exception as e:
                print(f"Warning: Error in pet feature detection: {e}")
        
        # Normalize feature map to range [0, 1]
        feature_map = np.clip(feature_map, 0, 1)
        
        return feature_map    
    def create_feature_mask(self, image, image_type):
        """
        Create a mask highlighting important features to preserve
        
        Parameters:
        - image: Input RGB image
        - image_type: Type of image ('portrait', 'pet', etc)
        
        Returns:
        - feature_mask: Binary mask where important features are 255, others 0
        - feature_regions: Dictionary with region info for important features
        """
        h, w = image.shape[:2]
        feature_mask = np.zeros((h, w), dtype=np.uint8)
        feature_regions = {}
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        if image_type == 'portrait':
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for i, (x, y, w_face, h_face) in enumerate(faces):
                # Create feature region
                feature_regions[f'face_{i}'] = {
                    'rect': (x, y, w_face, h_face),
                    'type': 'face'
                }
                
                # Mark face area in mask (lower priority)
                cv2.rectangle(feature_mask, (x, y), (x+w_face, y+h_face), 100, -1)
                
                # Define ROI for face
                roi_gray = gray[y:y+h_face, x:x+w_face]
                roi_color = image[y:y+h_face, x:x+w_face]
                
                # Detect eyes
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                for j, (ex, ey, ew, eh) in enumerate(eyes):
                    # Mark eye area in mask (high priority)
                    cv2.rectangle(feature_mask, (x+ex, y+ey), (x+ex+ew, y+ey+eh), 255, -1)
                    feature_regions[f'eye_{i}_{j}'] = {
                        'rect': (x+ex, y+ey, ew, eh),
                        'type': 'eye'
                    }
                
                # Detect mouth/smile
                smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
                for j, (sx, sy, sw, sh) in enumerate(smiles):
                    # Mark mouth area in mask (high priority)
                    cv2.rectangle(feature_mask, (x+sx, y+sy), (x+sx+sw, y+sy+sh), 255, -1)
                    feature_regions[f'mouth_{i}_{j}'] = {
                        'rect': (x+sx, y+sy, sw, sh),
                        'type': 'mouth'
                    }
                    
                # If no specific features detected, focus on central face area
                if len(eyes) == 0 and len(smiles) == 0:
                    # Central face area (eyes, nose, mouth region)
                    center_x, center_y = x + w_face//2, y + h_face//2
                    center_w, center_h = w_face//2, h_face//2
                    cv2.rectangle(feature_mask, 
                                 (center_x - center_w//2, center_y - center_h//2),
                                 (center_x + center_w//2, center_y + center_h//2),
                                 255, -1)
                    
        elif image_type == 'pet':
            # Pet detection is more challenging, use a combination of approaches
            
            # Try to detect pet face (may also detect human faces)
            pet_faces = self.pet_face_cascade.detectMultiScale(gray, 1.1, 3)
            
            if len(pet_faces) > 0:
                for i, (x, y, w_face, h_face) in enumerate(pet_faces):
                    # Create feature region
                    feature_regions[f'pet_face_{i}'] = {
                        'rect': (x, y, w_face, h_face),
                        'type': 'pet_face'
                    }
                    
                    # Mark pet face in mask
                    cv2.rectangle(feature_mask, (x, y), (x+w_face, y+h_face), 100, -1)
                    
                    # Try to detect eyes within the face region
                    roi_gray = gray[y:y+h_face, x:x+w_face]
                    eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                    
                    for j, (ex, ey, ew, eh) in enumerate(eyes):
                        # Mark eye area with high priority
                        cv2.rectangle(feature_mask, (x+ex, y+ey), (x+ex+ew, y+ey+eh), 255, -1)
                        
                        # Expand slightly around eyes for better context
                        expanded_x = max(0, x+ex-ew//2)
                        expanded_y = max(0, y+ey-eh//2)
                        expanded_w = min(w-expanded_x, ew*2)
                        expanded_h = min(h-expanded_y, eh*2)
                        
                        cv2.rectangle(feature_mask, 
                                     (expanded_x, expanded_y),
                                     (expanded_x+expanded_w, expanded_y+expanded_h),
                                     200, -1)
                        
                        feature_regions[f'pet_eye_{i}_{j}'] = {
                            'rect': (x+ex, y+ey, ew, eh),
                            'type': 'pet_eye'
                        }
            else:
                # If no face detected, use image center as fallback for pets
                # Many pet photos focus on the head/face in the center
                center_x, center_y = w//2, h//3  # Slightly above center
                region_w, region_h = w//3, h//3
                
                cv2.rectangle(feature_mask,
                             (center_x - region_w//2, center_y - region_h//2),
                             (center_x + region_w//2, center_y + region_h//2),
                             200, -1)
                
                feature_regions['pet_center'] = {
                    'rect': (center_x - region_w//2, center_y - region_h//2, region_w, region_h),
                    'type': 'pet_center'
                }
                
        elif image_type == 'still_life':
            # For still life, use contour detection to find distinct objects
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and prioritize significant contours
            significant_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > (h*w*0.01):  # Minimum size threshold
                    significant_contours.append(contour)
                    
            # Sort by area (largest first)
            significant_contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
            
            # Take top N contours
            for i, contour in enumerate(significant_contours[:5]):
                cv2.drawContours(feature_mask, [contour], -1, 200, -1)
                
                # Get bounding rectangle
                x, y, w_obj, h_obj = cv2.boundingRect(contour)
                feature_regions[f'object_{i}'] = {
                    'rect': (x, y, w_obj, h_obj),
                    'type': 'still_life_object',
                    'area': cv2.contourArea(contour)
                }
                
        # Apply distance transform to create gradient around features
        if np.any(feature_mask):
            binary_mask = feature_mask.copy()
            binary_mask[binary_mask > 0] = 255
            
            # Distance transform creates gradient based on distance from features
            dist = cv2.distanceTransform(255 - binary_mask, cv2.DIST_L2, 3)
            
            # Normalize and invert so closer to feature = higher value
            dist = cv2.normalize(dist, None, 0, 1, cv2.NORM_MINMAX)
            gradient_mask = (1 - dist) * 255
            
            # Blend with original mask for smoother transition
            feature_mask = cv2.addWeighted(feature_mask, 0.7, gradient_mask.astype(np.uint8), 0.3, 0)
            
        return feature_mask, feature_regions