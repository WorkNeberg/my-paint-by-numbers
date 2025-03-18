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