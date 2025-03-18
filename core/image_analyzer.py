import cv2
import numpy as np
from scipy import ndimage
from skimage import feature

class ImageAnalyzer:
    """Analyzes images to detect characteristics for optimal processing"""
    
    def __init__(self):
        # No deep learning model
        self.model = None
        print("Using heuristic-based image analysis (no TensorFlow needed)")
    
    def analyze(self, image):
        """
        Analyze image to determine characteristics and optimal processing parameters
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            dict with analysis results
        """
        results = {}
        
        # Analyze basic color distribution
        results.update(self._analyze_colors(image))
        
        # Analyze edge complexity
        results.update(self._analyze_edges(image))
        
        # Classify image type using heuristics
        results.update(self._classify_image_heuristic(results))
        
        # Generate optimized parameters based on analysis
        results['parameters'] = self._get_optimal_parameters(results)
        
        return results
    
    def _analyze_colors(self, image):
        """Analyze color distribution in the image"""
        # Convert to different color spaces for analysis
        rgb = image.copy()
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate brightness stats
        brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        # Calculate dark area percentage
        dark_threshold = 60  # Pixels below this are considered "dark"
        dark_percentage = np.sum(gray < dark_threshold) / gray.size * 100
        
        # Calculate saturation stats
        saturation = np.mean(hsv[:,:,1])
        saturation_std = np.std(hsv[:,:,1])
        
        # Calculate color diversity
        # (Simplified - could use clustering or histogram analysis)
        color_diversity_score = saturation_std * brightness_std / 128.0
        
        # Classify color diversity level
        if color_diversity_score < 0.2:
            color_diversity = "low"
        elif color_diversity_score < 0.4:
            color_diversity = "medium"
        else:
            color_diversity = "high"
            
        # Detect if the image has large uniform areas
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        _, uniform_mask = cv2.threshold(
            np.abs(gray - blurred), 
            10, 255, 
            cv2.THRESH_BINARY_INV
        )
        uniform_percentage = np.sum(uniform_mask > 0) / uniform_mask.size * 100
        
        return {
            'brightness': float(brightness),
            'brightness_std': float(brightness_std),
            'dark_percentage': float(dark_percentage),
            'saturation': float(saturation),
            'color_diversity': color_diversity,
            'uniform_area_percentage': float(uniform_percentage),
            'needs_dark_enhancement': dark_percentage > 20
        }
    
    def _analyze_edges(self, image):
        """Analyze edge characteristics in the image"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Detect edges using Canny
        edges = feature.canny(gray, sigma=2)
        
        # Calculate edge density
        edge_density = np.sum(edges) / edges.size * 100
        
        # Calculate edge complexity (using gradient magnitude variance)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_complexity = np.std(gradient_magnitude) / np.mean(gradient_magnitude) if np.mean(gradient_magnitude) > 0 else 0
        
        # Classify edge complexity
        if edge_complexity < 1.0:
            complexity_level = "low"
        elif edge_complexity < 2.0:
            complexity_level = "medium"
        else:
            complexity_level = "high"
            
        return {
            'edge_density': float(edge_density),
            'edge_complexity': complexity_level,
            'edge_complexity_score': float(edge_complexity)
        }
    
    def _classify_image_heuristic(self, analysis):
        """Classify image type using simple heuristics"""
        # Get values from analysis
        dark_percentage = analysis.get('dark_percentage', 0)
        edge_density = analysis.get('edge_density', 0)
        brightness = analysis.get('brightness', 128)
        saturation = analysis.get('saturation', 128)
        edge_complexity_score = analysis.get('edge_complexity_score', 1.0)
        uniform_percentage = analysis.get('uniform_area_percentage', 0)
        
        # Initialize with default confidence
        confidence = 0.6
        
        # Basic heuristics for classification
        if edge_density > 15 and dark_percentage > 30:
            # High edge density + dark areas often indicates pets (fur, etc.)
            suggested_type = "pet"
            confidence = 0.7
            
        elif edge_complexity_score < 1.2 and saturation > 80 and uniform_percentage > 40:
            # Low complexity, high saturation, uniform areas = cartoon/illustration
            suggested_type = "cartoon"
            confidence = 0.8
            
        elif edge_density < 10 and uniform_percentage > 60:
            # Low edge density with large uniform areas often means landscape
            suggested_type = "landscape"
            confidence = 0.7
            
        elif 5 < edge_density < 20 and 80 < brightness < 200 and saturation < 100:
            # Medium edge density, brightness and low saturation often indicates portraits
            suggested_type = "portrait"
            confidence = 0.6
            
        else:
            # Default to "general" if no specific pattern is detected
            suggested_type = "general"
            confidence = 0.5
        
        return {
            'suggested_type': suggested_type,
            'confidence': confidence
        }
    
    def _get_optimal_parameters(self, analysis):
        """Determine optimal processing parameters based on image analysis"""
        image_type = analysis.get('suggested_type', 'general')
        dark_percentage = analysis.get('dark_percentage', 0)
        edge_complexity = analysis.get('edge_complexity', 'medium')
        color_diversity = analysis.get('color_diversity', 'medium')
        
        # Start with base parameters
        params = {
            'num_colors': 15,
            'simplification_level': 'medium',
            'edge_strength': 1.0,
            'edge_width': 1,
            'enhance_dark_areas': False,
            'dark_threshold': 50,
        }
        
        # Adjust based on image type
        if image_type == 'portrait':
            params['num_colors'] = 12
            params['simplification_level'] = 'low'  # Preserve details
            params['edge_strength'] = 0.8
        
        elif image_type == 'pet':
            params['num_colors'] = 15
            params['simplification_level'] = 'low'
            params['edge_strength'] = 1.2
            params['enhance_dark_areas'] = True
            params['dark_threshold'] = 70
        
        elif image_type == 'landscape':
            params['num_colors'] = 18
            params['simplification_level'] = 'medium'
            params['edge_strength'] = 1.0
            params['edge_width'] = 2
        
        elif image_type == 'cartoon':
            params['num_colors'] = 10
            params['simplification_level'] = 'high'
            params['edge_strength'] = 1.5
            params['edge_width'] = 2
        
        # Further adjust based on detected characteristics
        if dark_percentage > 30:
            params['enhance_dark_areas'] = True
            params['dark_threshold'] = min(80, 40 + dark_percentage // 2)
        
        if color_diversity == 'high':
            params['num_colors'] = min(25, params['num_colors'] + 5)
        elif color_diversity == 'low':
            params['num_colors'] = max(8, params['num_colors'] - 3)
            
        if edge_complexity == 'high':
            params['simplification_level'] = 'medium'
            params['edge_width'] = max(1, params['edge_width'] - 1)
        elif edge_complexity == 'low':
            params['edge_width'] = min(3, params['edge_width'] + 1)
            
        return params