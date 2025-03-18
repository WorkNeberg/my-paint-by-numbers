import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

class ImageTypeDetector:
    def __init__(self):
        # Initialize model for classification
        self.model = MobileNetV2(weights='imagenet', include_top=True)
        
        # Mapping of ImageNet classes to our categories
        self.pet_classes = ['dog', 'cat', 'hamster', 'rabbit', 'fox', 'pet']
        self.portrait_classes = ['face', 'person', 'woman', 'man', 'girl', 'boy']
        self.landscape_classes = ['mountain', 'beach', 'field', 'forest', 'sky', 'sea', 'valley', 'lake']
        self.cartoon_classes = ['cartoon', 'animation', 'comic']
        self.still_life_classes = ['vase', 'fruit', 'flower', 'bottle', 'table', 'chair', 'food']
        self.architecture_classes = ['building', 'house', 'tower', 'castle', 'church', 'bridge']
        
    def detect_image_type(self, image):
        """
        Detect the type of image using deep learning classification
        
        Parameters:
        - image: Input RGB image
        
        Returns:
        - image_type: Detected image type as string
        - confidence: Confidence level (0-1)
        """
        # Resize image for model input
        img = cv2.resize(image, (224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Make prediction
        predictions = self.model.predict(img_array)
        decoded = decode_predictions(predictions, top=5)[0]
        
        # Count matches for each category
        scores = {
            "pet": 0,
            "portrait": 0,
            "landscape": 0,
            "cartoon": 0,
            "still_life": 0,
            "architecture": 0,
            "general": 0
        }
        
        # Process top 5 predictions
        for _, label, score in decoded:
            label = label.lower()
            if any(cls in label for cls in self.pet_classes):
                scores["pet"] += score
            elif any(cls in label for cls in self.portrait_classes):
                scores["portrait"] += score
            elif any(cls in label for cls in self.landscape_classes):
                scores["landscape"] += score
            elif any(cls in label for cls in self.cartoon_classes):
                scores["cartoon"] += score
            elif any(cls in label for cls in self.still_life_classes):
                scores["still_life"] += score
            elif any(cls in label for cls in self.architecture_classes):
                scores["architecture"] += score
            else:
                scores["general"] += score
                
        # Alternative approach: Use face detection for portrait verification
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # If face detected, increase portrait score
            face_area = sum(w * h for (x, y, w, h) in faces)
            image_area = image.shape[0] * image.shape[1]
            face_ratio = face_area / image_area
            
            if face_ratio > 0.1:  # Significant face in image
                scores["portrait"] += 0.5
        
        # Find the category with highest score
        best_type = max(scores.items(), key=lambda x: x[1])
        
        return best_type[0], best_type[1]
    
    def detect_type_simple(self, image):
        """Simpler method that doesn't require TensorFlow"""
        # Convert to grayscale for faster processing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Calculate face area ratio
            face_area = sum(w * h for (x, y, w, h) in faces)
            image_area = image.shape[0] * image.shape[1]
            face_ratio = face_area / image_area
            
            if face_ratio > 0.1:
                # Check if it might be a pet face rather than human
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    # Try to detect eyes within face
                    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                    eyes = eye_cascade.detectMultiScale(face_roi)
                    
                    # If no eyes or unusual proportions, might be pet
                    if len(eyes) == 0 or w/h > 1.5:
                        return "pet", 0.7
                
                return "portrait", 0.8
                
        # Check for landscapes based on edge distribution
        edges = cv2.Canny(gray, 100, 200)
        
        # Calculate edge density in different regions
        h, w = edges.shape
        top_half = np.sum(edges[:h//2, :]) / (h * w // 2)
        bottom_half = np.sum(edges[h//2:, :]) / (h * w // 2)
        
        if top_half < bottom_half * 0.5:  # Much more detail in bottom than top
            return "landscape", 0.7
            
        # Check color distribution for still life
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # More saturated colors often indicate still life/flowers
        if np.mean(hsv[:,:,1]) > 100:
            return "still_life", 0.6
            
        # Default to general
        return "general", 0.5