import os
import cv2
import numpy as np
import logging
from typing import Dict, Tuple, Union

# Setup logging
logger = logging.getLogger('pbn-app')

class ImageTypeDetector:
    """
    Detects the type of image (pet, portrait, landscape, still_life) using
    either ML models or heuristic-based approaches.
    """
    
    def __init__(self):
        # Type confidence thresholds
        self.confidence_threshold = 0.5
        self.types = ['pet', 'portrait', 'landscape', 'still_life']
        
        # Initialize any required models
        self.initialized = True
        logger.info("Using heuristic-based image analysis (no TensorFlow needed)")
        
    def detect_image_type(self, image_path_or_array: Union[str, np.ndarray], force_type=None) -> str:
        """
        Detect type of image from file path or image array
        
        Args:
            image_path_or_array: Path to image file or numpy array of image
            force_type: Optional override to force a specific type
            
        Returns:
            Detected image type as string
        """
        # If type is forced, use that
        if force_type:
            logger.info(f"Using forced image type: {force_type}")
            return force_type
            
        # Load image if path is provided
        if isinstance(image_path_or_array, str):
            try:
                image = cv2.imread(image_path_or_array)
                if image is None:
                    logger.error(f"Could not read image from path: {image_path_or_array}")
                    return 'general'  # Default if can't read image
            except Exception as e:
                logger.error(f"Error reading image: {str(e)}")
                return 'general'  # Default if error
        else:
            image = image_path_or_array
        
        # Resize for consistent analysis
        try:
            img = cv2.resize(image, (224, 224))
        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            return 'general'  # Default if resize fails
        
        # Use heuristic approach for detection
        scores = self._analyze_image_heuristics(img)
        
        # Get predicted type and confidence
        predicted_type = max(scores, key=scores.get)
        confidence = scores[predicted_type]
        
        logger.info(f"Auto-detected image type: {predicted_type} (confidence: {confidence:.2f})")
        
        # If confidence is low, fall back to general
        if confidence < self.confidence_threshold:
            logger.info(f"Low confidence, using general image type")
            return 'general'
            
        return predicted_type
        
    def _analyze_image_heuristics(self, image: np.ndarray) -> Dict[str, float]:
        """
        Analyzes image using heuristics to determine image type
        
        Args:
            image: Input image (224x224 RGB)
            
        Returns:
            Dictionary of confidence scores for each image type
        """
        # Convert to RGB if needed
        if image.shape[2] == 3 and image.dtype == np.uint8:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image
            
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        
        # Initialize scores
        scores = {
            'pet': 0.0,
            'portrait': 0.0,
            'landscape': 0.0,
            'still_life': 0.0
        }
        
        # Pet detection
        # For pets, look for face-like structures, fur textures, and common pet colors
        # Run object detection for eyes
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
        
        # If 2 eyes are detected close to each other, this increases pet or portrait probability
        if len(eyes) >= 2:
            scores['pet'] += 0.2
            scores['portrait'] += 0.3
            
        # Check for fur textures using edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        if 0.1 < edge_density < 0.3:
            scores['pet'] += 0.2  # Medium edge density often indicates fur
            
        # Analyze color distribution for common pet colors
        h_channel = hsv[:,:,0]
        s_channel = hsv[:,:,1]
        
        # Pets often have browns, grays, blacks
        pet_color_mask = ((h_channel > 5) & (h_channel < 30) & (s_channel > 50)) | (s_channel < 30)
        pet_color_ratio = np.sum(pet_color_mask) / (pet_color_mask.shape[0] * pet_color_mask.shape[1])
        scores['pet'] += pet_color_ratio * 0.4
        
        # Portrait detection
        # Run face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            scores['portrait'] += 0.5
            
        # Check skin tones
        skin_lower = np.array([0, 20, 70], dtype=np.uint8)
        skin_upper = np.array([20, 150, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
        skin_ratio = np.sum(skin_mask > 0) / (skin_mask.shape[0] * skin_mask.shape[1])
        
        if 0.1 < skin_ratio < 0.5:
            scores['portrait'] += skin_ratio * 0.5
            
        # Landscape detection
        # Check for horizontal lines and natural colors
        horizontal_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        horizontal_edge_strength = np.mean(np.abs(horizontal_sobel))
        
        vertical_sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        vertical_edge_strength = np.mean(np.abs(vertical_sobel))
        
        # Landscapes typically have stronger horizontal edges
        if horizontal_edge_strength > vertical_edge_strength:
            scores['landscape'] += 0.3
            
        # Check for sky colors (blues)
        sky_lower = np.array([90, 50, 100], dtype=np.uint8)
        sky_upper = np.array([130, 255, 255], dtype=np.uint8)
        sky_mask = cv2.inRange(hsv, sky_lower, sky_upper)
        sky_ratio = np.sum(sky_mask > 0) / (sky_mask.shape[0] * sky_mask.shape[1])
        
        if sky_ratio > 0.2:
            scores['landscape'] += sky_ratio * 0.4
            
        # Check for green vegetation
        green_lower = np.array([35, 50, 50], dtype=np.uint8)
        green_upper = np.array([85, 255, 255], dtype=np.uint8)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        green_ratio = np.sum(green_mask > 0) / (green_mask.shape[0] * green_mask.shape[1])
        
        if green_ratio > 0.15:
            scores['landscape'] += green_ratio * 0.3
            
        # Still life detection
        # Check for object boundaries and distinctive colors
        # Still life images often have clear object boundaries and high color diversity
        
        # Calculate color diversity
        h_bins = cv2.calcHist([hsv], [0], None, [30], [0, 180])
        s_bins = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        v_bins = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        h_nonzero = np.sum(h_bins > 0) / len(h_bins)
        s_nonzero = np.sum(s_bins > 0) / len(s_bins)
        color_diversity = (h_nonzero + s_nonzero) / 2
        
        scores['still_life'] += color_diversity * 0.5
        
        # Check for clear object boundaries
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = np.var(lap)
        
        if lap_var > 100:  # High variance indicates clear edges
            scores['still_life'] += min(lap_var / 1000, 0.4)
            
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            for k in scores:
                scores[k] = scores[k] / total
                
        return scores