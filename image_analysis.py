import cv2
import numpy as np

class ImageAnalyzer:
    """Analyzes images for paint-by-numbers processing"""
    
    def analyze_image(self, image):
        """Analyze image and determine type and optimal parameters"""
        # Detect image type
        image_type = self.detect_image_type(image)
        
        # Calculate image characteristics
        brightness = self.calculate_brightness(image)
        contrast = self.calculate_contrast(image)
        complexity = self.analyze_complexity(image)
        
        # Determine optimal parameters
        params = self.determine_optimal_parameters(image_type, brightness, contrast, complexity)
        
        return {
            "image_type": image_type,
            "brightness": brightness, 
            "contrast": contrast,
            "complexity": complexity,
            "params": params
        }
    
    def detect_image_type(self, image):
        """Determine if image is portrait, landscape, pet, etc."""
        # Check for portrait (face detection)
        if self.detect_faces(image):
            return "portrait"
            
        # Check for landscape features
        if self.is_landscape(image):
            return "landscape"
            
        # Check for pets (simplified)
        if self.might_be_pet(image):
            return "pet"
            
        # Default type
        return "general"
    
    def detect_faces(self, image):
        """Detect faces in image"""
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Check if faces occupy significant portion of image
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    face_area = w * h
                    image_area = image.shape[0] * image.shape[1]
                    if face_area / image_area > 0.05:  # Face is >5% of image
                        return True
            return False
        except Exception:
            # Fallback in case face detection fails
            return False
    
    def is_landscape(self, image):
        """Check if image has landscape characteristics"""
        h, w = image.shape[:2]
        
        # Check aspect ratio
        if w / h > 1.4:
            # Check for sky and ground color distribution
            top_section = image[:h//3, :]
            bottom_section = image[2*h//3:, :]
            
            top_avg = np.mean(top_section, axis=(0, 1))
            bottom_avg = np.mean(bottom_section, axis=(0, 1))
            
            # In landscapes, top often has more blue (index 0 in BGR)
            # Bottom often has more green (index 1 in BGR)
            if top_avg[0] > top_avg[1] and bottom_avg[1] > bottom_avg[0]:
                return True
                
        return False
    
    def might_be_pet(self, image):
        """Simple heuristic for pet detection"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Check for fur-like textures using Laplacian
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = laplacian.var()
        
        # High texture variance often indicates fur
        return lap_var > 500
    
    def analyze_complexity(self, image):
        """Analyze image complexity on a scale of 1-10"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
        
        # Texture complexity
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_complexity = min(laplacian.var() / 1000, 1.0)
        
        # Combined complexity score
        complexity = (edge_density * 5 + texture_complexity * 5)
        
        return max(1, min(10, int(complexity * 10)))
    
    def calculate_brightness(self, image):
        """Calculate average brightness (0-100)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return (np.mean(gray) / 255) * 100
    
    def calculate_contrast(self, image):
        """Calculate image contrast (0-100)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray)
        return min(100, (std_dev / 80) * 100)
    
    def determine_optimal_parameters(self, image_type, brightness, contrast, complexity):
        """Determine optimal processing parameters"""
        # Base parameters
        params = {
            "smoothing": 5,             # Default smoothing (0-10)
            "edge_preservation": 5,     # Default edge preservation (0-10)
            "color_count": 24,          # Default number of colors
            "min_segment_size": 100,    # Default minimum segment size
            "detail_level": 5           # Default detail level (0-10)
        }
        
        # Adjust based on image type
        if image_type == "portrait":
            params["smoothing"] = 4         # Less smoothing to preserve facial features
            params["edge_preservation"] = 7  # Stronger edge preservation
            params["min_segment_size"] = 80  # Smaller segments for facial details
            params["detail_level"] = 7       # More detail for faces
            
        elif image_type == "landscape":
            params["smoothing"] = 6         # More smoothing for landscape
            params["edge_preservation"] = 4  # Less edge preservation
            params["min_segment_size"] = 120 # Larger segments for landscapes
            params["detail_level"] = 4       # Less detail for landscapes
            
        elif image_type == "pet":
            params["smoothing"] = 5         # Medium smoothing
            params["edge_preservation"] = 6  # Good edge preservation for fur
            params["min_segment_size"] = 90  # Medium segments
            params["detail_level"] = 6       # Good detail for pet features
        
        # Adjust for brightness
        if brightness < 30:  # Dark image
            params["smoothing"] = max(2, params["smoothing"] - 2)  # Less smoothing
            params["edge_preservation"] += 1                       # More edge preservation
            
        # Adjust for contrast
        if contrast < 20:  # Low contrast
            params["edge_preservation"] += 2  # Enhance edges more
            
        # Adjust for complexity
        complexity_factor = complexity / 5  # Normalize to 0-2 range
        params["detail_level"] = min(10, int(params["detail_level"] * complexity_factor))
        params["min_segment_size"] = max(50, int(params["min_segment_size"] / complexity_factor))
        
        return params