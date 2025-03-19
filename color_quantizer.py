import cv2
import numpy as np
from sklearn.cluster import KMeans

class ColorQuantizer:
    """Creates optimal color palettes for paint-by-numbers"""
    
    def quantize_colors(self, image, segments, image_type, color_count=24):
        """Generate optimal palette and map segments to colors
        
        Args:
            image: OpenCV image in BGR format
            segments: Segmented image regions
            image_type: Type of image (portrait, landscape, etc.)
            color_count: Number of colors to use
        
        Returns:
            tuple: (palette, color_map)
        """
        # Extract average colors from segments
        segment_colors = self.extract_segment_colors(image, segments)
        
        # Create initial palette
        initial_palette = self.cluster_colors(segment_colors, color_count, image_type)
        
        # Refine palette for better painting experience
        refined_palette = self.refine_palette(initial_palette, image_type)
        
        # Map segments to palette colors
        color_map = self.map_segments_to_palette(segment_colors, refined_palette)
        
        return refined_palette, color_map
    
    def extract_segment_colors(self, image, segments):
        """Extract average color for each segment"""
        # Convert image to LAB color space for perceptual distance
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Extract average color for each segment
        segment_colors = {}
        for segment_id in np.unique(segments):
            mask = segments == segment_id
            if np.sum(mask) > 0:
                # Calculate average color in LAB space
                segment_colors[segment_id] = np.mean(lab_image[mask], axis=0)
                
        return segment_colors
    
    def cluster_colors(self, segment_colors, color_count, image_type):
        """Cluster segment colors to create a palette"""
        # Convert dictionary to array
        colors = np.array(list(segment_colors.values()))
        
        # Apply weights based on image type
        weighted_colors = self.apply_weights(colors.copy(), image_type)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=color_count, random_state=42, n_init=10)
        kmeans.fit(weighted_colors)
        
        # Convert to BGR for output
        palette = []
        for center in kmeans.cluster_centers_:
            lab_color = np.array([center], dtype=np.float32).reshape(1, 1, 3)
            bgr_color = cv2.cvtColor(np.uint8(lab_color), cv2.COLOR_LAB2BGR)[0, 0]
            palette.append(bgr_color)
        
        return palette
    
    def apply_weights(self, colors, image_type):
        """Apply weights to color channels based on image type"""
        if image_type == "portrait":
            # For portraits, emphasize a/b channels (color) for skin tones
            colors[:, 1] *= 1.5  # a channel
            colors[:, 2] *= 1.5  # b channel
        elif image_type == "landscape":
            # For landscapes, emphasize L channel (luminance)
            colors[:, 0] *= 1.5  # L channel
        
        return colors
    
    def refine_palette(self, palette, image_type):
        """Refine palette for better painting experience"""
        # Convert to HSV for easier color manipulation
        hsv_palette = []
        for color in palette:
            hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0, 0]
            hsv_palette.append(hsv)
        
        hsv_palette = np.array(hsv_palette)
        
        # Check each color pair
        for i in range(len(hsv_palette)):
            for j in range(i + 1, len(hsv_palette)):
                # Calculate color difference in HSV space
                h1, s1, v1 = hsv_palette[i]
                h2, s2, v2 = hsv_palette[j]
                
                # Calculate hue difference (circular)
                h_diff = min(abs(h1 - h2), 180 - abs(h1 - h2))
                s_diff = abs(s1 - s2)
                v_diff = abs(v1 - v2)
                
                # If colors are too similar
                if h_diff < 10 and s_diff < 40 and v_diff < 40:
                    # Increase saturation difference
                    mean_s = (s1 + s2) / 2
                    hsv_palette[i][1] = min(255, mean_s + 20)
                    hsv_palette[j][1] = max(0, mean_s - 20)
                    
                    # Increase value difference
                    mean_v = (v1 + v2) / 2
                    hsv_palette[i][2] = min(255, mean_v + 15)
                    hsv_palette[j][2] = max(0, mean_v - 15)
        
        # Convert back to BGR
        bgr_palette = []
        for hsv in hsv_palette:
            bgr = cv2.cvtColor(np.uint8([[[hsv[0], hsv[1], hsv[2]]]]), cv2.COLOR_HSV2BGR)[0, 0]
            bgr_palette.append(bgr)
            
        return bgr_palette
    
    def map_segments_to_palette(self, segment_colors, palette):
        """Map each segment to the closest palette color"""
        # Convert palette colors to LAB for perceptual comparison
        lab_palette = []
        for color in palette:
            bgr_color = np.uint8([[color]])
            lab_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2LAB)[0, 0]
            lab_palette.append(lab_color)
        
        # Map segments to closest palette colors
        color_map = {}
        for segment_id, segment_color in segment_colors.items():
            # Calculate distance to each palette color
            distances = []
            for palette_color in lab_palette:
                dist = np.sqrt(np.sum((segment_color - palette_color)**2))
                distances.append(dist)
                
            # Assign to closest color
            closest_color = np.argmin(distances)
            color_map[segment_id] = closest_color
            
        return color_map