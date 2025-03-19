import cv2
import numpy as np
from skimage import segmentation, measure, morphology

class Segmenter:
    """Creates segments with type-specific optimizations"""
    
    def __init__(self):
        pass
    
    def segment(self, image, image_type, params):
        """Create segments optimized for the image type"""
        # Initial segmentation using SLIC superpixels
        segments = self.create_base_segments(image, params["detail_level"], params["min_segment_size"])
        
        # Apply type-specific refinements
        if image_type == "portrait":
            segments = self.refine_portrait_segments(image, segments)
        elif image_type == "landscape":
            segments = self.refine_landscape_segments(image, segments)
            
        # Enforce minimum segment size
        segments = self.merge_small_segments(segments, params["min_segment_size"])
        
        # Smooth segment boundaries
        segments = self.smooth_segment_boundaries(segments)
        
        return segments
    
    def create_base_segments(self, image, detail_level, min_size):
        """Create initial segmentation"""
        # Calculate number of segments based on image size and detail level
        h, w = image.shape[:2]
        image_size = h * w
        
        # Adjust segment count based on detail level
        n_segments = int(image_size * detail_level / 50000)
        n_segments = max(50, min(2000, n_segments))
        
        # Create superpixels
        segments = segmentation.slic(
            image, 
            n_segments=n_segments,
            compactness=10,
            sigma=1,
            start_label=1
        )
        
        return segments
    
    def refine_portrait_segments(self, image, segments):
        """Refine segments for portrait images"""
        try:
            # Detect faces
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return segments
                
            # For portraits, we'll preserve the segmentation as is
            # since the face-aware processing was done in the image processing step
        except Exception:
            pass
            
        return segments
    
    def refine_landscape_segments(self, image, segments):
        """Refine segments for landscape images"""
        # No special refinement needed beyond the initial segmentation
        return segments
    
    def merge_small_segments(self, segments, min_size):
        """Merge segments smaller than minimum size"""
        # Get region properties
        props = measure.regionprops(segments)
        
        # Identify small regions
        small_regions = [prop.label for prop in props if prop.area < min_size]
        
        if not small_regions:
            return segments
            
        # Create a copy we can modify
        labeled = segments.copy()
        
        # Process each small region
        for region_label in small_regions:
            # Create mask for this region
            region_mask = labeled == region_label
            
            # Skip if already processed in a previous iteration
            if not np.any(region_mask):
                continue
                
            # Dilate to find neighbors
            dilated = morphology.binary_dilation(region_mask)
            neighbor_mask = dilated & ~region_mask
            
            # Find most common neighboring label
            if np.any(neighbor_mask):
                neighbor_labels = labeled[neighbor_mask]
                if len(neighbor_labels) > 0:
                    # Find most common neighbor that's not another small region
                    valid_neighbors = [l for l in neighbor_labels if l not in small_regions and l != region_label]
                    if valid_neighbors:
                        most_common = np.bincount(valid_neighbors).argmax()
                        labeled[region_mask] = most_common
        
        return labeled
    
    def smooth_segment_boundaries(self, segments):
        """Smooth segment boundaries for better painting experience"""
        # Process each label individually
        max_label = np.max(segments)
        smoothed = segments.copy()
        
        for label in range(1, max_label + 1):
            # Create binary mask for this label
            mask = (segments == label).astype(np.uint8)
            
            # Apply morphological closing to smooth boundaries
            kernel = np.ones((3, 3), np.uint8)
            smoothed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Update only where the smoothed mask is 1 but original mask is 0
            # and avoid conflicts with other labels
            update_mask = (smoothed_mask == 1) & (mask == 0)
            smoothed[update_mask] = label
                
        return smoothed