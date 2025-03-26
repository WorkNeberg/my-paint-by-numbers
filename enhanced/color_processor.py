import cv2
import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import colorsys

logger = logging.getLogger('pbn-app.color_processor')

class ColorProcessor:
    """
    Handles color processing operations including palette generation,
    color harmony, and region-specific color optimization.
    """
    
    def __init__(self):
        """Initialize the color processor"""
        logger.info("ColorProcessor initialized")
    
    def process_colors(self, image, segments, settings, feature_map=None):
        """
        Process colors for the image based on settings
        
        Args:
            image: Input image as numpy array (BGR format)
            segments: Segmented image regions
            settings: Dictionary of color processing settings
            feature_map: Optional feature importance map
            
        Returns:
            Dictionary containing palette and processed segments
        """
        logger.info("Starting color processing")
        
        # Extract settings
        color_settings = settings.get('color_control', {})
        num_colors = settings.get('colors', 15)
        harmony_type = color_settings.get('color_harmony', 'none')
        saturation_boost = color_settings.get('color_saturation_boost', 0.0)
        dark_enhancement = color_settings.get('dark_area_enhancement', 0.5)
        light_protection = color_settings.get('light_area_protection', 0.5)
        color_threshold = color_settings.get('color_grouping_threshold', 0.5)
        highlight_preservation = color_settings.get('highlight_preservation', 'medium')
        
        # Get image type
        image_type = settings.get('image_type', 'generic')
        
        # Generate initial palette
        palette = self._generate_initial_palette(image, segments, num_colors)
        
        # Apply color harmony if requested
        if harmony_type != 'none':
            palette = self._apply_color_harmony(palette, harmony_type)
        
        # Apply saturation adjustment
        if saturation_boost != 0:
            palette = self._adjust_saturation(palette, saturation_boost)
        
        # Apply type-specific optimizations
        if image_type == 'pet':
            palette = self._optimize_pet_palette(palette, dark_enhancement)
        elif image_type == 'portrait':
            palette = self._optimize_portrait_palette(palette)
        elif image_type == 'landscape':
            palette = self._optimize_landscape_palette(palette)
        
        # Apply dark area enhancement
        if dark_enhancement > 0:
            palette = self._enhance_dark_colors(palette, dark_enhancement)
        
        # Apply highlight preservation
        if highlight_preservation != 'low':
            palette = self._preserve_highlights(palette, highlight_preservation)
        
        # Assign colors to segments
        colored_segments = self._assign_colors_to_segments(image, segments, palette, 
                                                         color_threshold, feature_map)
        
        logger.info(f"Color processing complete: {len(palette)} colors in palette")
        return {
            'palette': palette,
            'colored_segments': colored_segments
        }
    
    def _generate_initial_palette(self, image, segments, num_colors):
        """Generate initial color palette using K-Means clustering"""
        logger.debug(f"Generating palette with {num_colors} colors")
        
        # Reshape image to list of pixels
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # Use K-Means to find color clusters
        kmeans = KMeans(n_clusters=num_colors, n_init=10)
        kmeans.fit(pixels)
        
        # Get the colors
        colors = kmeans.cluster_centers_.astype(np.uint8)
        
        return colors
    
    def _apply_color_harmony(self, palette, harmony_type):
        """Apply color harmony transformations to the palette"""
        logger.debug(f"Applying {harmony_type} color harmony")
        
        # Convert palette to HSV for easier manipulation
        hsv_palette = []
        for color in palette:
            # Convert from BGR to RGB
            rgb = color[::-1]
            # Convert RGB to HSV (range 0-1)
            h, s, v = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
            hsv_palette.append((h, s, v))
        
        # Apply harmony transformation
        harmonized_hsv = []
        
        if harmony_type == 'complementary':
            # Group colors around two complementary hues
            for h, s, v in hsv_palette:
                # Decide which of the two complementary colors this is closest to
                h1 = h
                h2 = (h + 0.5) % 1.0
                
                # Choose the closer one
                if abs(h - h1) < abs(h - h2):
                    h_new = h1 + (h - h1) * 0.5  # Move closer to primary hue
                else:
                    h_new = h2 + (h - h2) * 0.5  # Move closer to complementary hue
                    
                harmonized_hsv.append((h_new % 1.0, s, v))
                
        elif harmony_type == 'analogous':
            # Group colors in neighboring hues
            # Find average hue
            hues = [h for h, _, _ in hsv_palette]
            avg_hue = sum(hues) / len(hues)
            
            # Move all hues closer to the average +- 30 degrees
            for h, s, v in hsv_palette:
                h_diff = (h - avg_hue + 0.5) % 1.0 - 0.5  # Difference in range -0.5 to 0.5
                
                # Limit to +- 1/6 (60 degrees)
                if abs(h_diff) > 1/6:
                    h_new = avg_hue + (1/6 if h_diff > 0 else -1/6)
                else:
                    h_new = h
                    
                harmonized_hsv.append((h_new % 1.0, s, v))
                
        elif harmony_type == 'triadic':
            # Group colors around three hues, 120 degrees apart
            triads = [0, 1/3, 2/3]  # Three hues equally spaced
            
            for h, s, v in hsv_palette:
                # Find closest triad
                diffs = [(h - triad + 0.5) % 1.0 - 0.5 for triad in triads]
                closest_idx = np.argmin([abs(diff) for diff in diffs])
                closest_triad = triads[closest_idx]
                diff = diffs[closest_idx]
                
                # Move 70% closer to the triad
                h_new = (closest_triad + diff * 0.3) % 1.0
                harmonized_hsv.append((h_new, s, v))
                
        elif harmony_type == 'monochromatic':
            # Use a single hue with varying saturation and value
            hues = [h for h, _, _ in hsv_palette]
            avg_hue = sum(hues) / len(hues)
            
            for _, s, v in hsv_palette:
                harmonized_hsv.append((avg_hue, s, v))
        else:
            # No harmony, return original
            harmonized_hsv = hsv_palette
        
        # Convert back to BGR
        harmonized_palette = []
        for h, s, v in harmonized_hsv:
            # Convert HSV to RGB
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            # Convert to 0-255 range and BGR order
            harmonized_palette.append(np.array([b*255, g*255, r*255], dtype=np.uint8))
        
        return np.array(harmonized_palette)
    
    def _adjust_saturation(self, palette, boost):
        """Adjust saturation of all colors in the palette"""
        logger.debug(f"Adjusting saturation by {boost}")
        
        # Convert to HSV for easier saturation adjustment
        adjusted_palette = []
        
        for color in palette:
            # Convert BGR to RGB
            rgb = color[::-1]
            # Convert RGB to HSV
            h, s, v = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
            
            # Adjust saturation
            s = max(0, min(1, s + boost))
            
            # Convert back to RGB
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            # Convert to BGR and 0-255 range
            adjusted_palette.append(np.array([b*255, g*255, r*255], dtype=np.uint8))
        
        return np.array(adjusted_palette)
    
    def _optimize_pet_palette(self, palette, dark_enhancement):
        """Optimize palette specifically for pet images"""
        logger.debug("Applying pet-specific color optimizations")
        
        # For pets, we want to:
        # 1. Ensure good separation between similar fur colors
        # 2. Preserve eye colors
        # 3. Enhance detail in dark fur
        
        # First, convert to Lab color space for better perceptual distance
        lab_palette = np.zeros_like(palette, dtype=np.float32)
        for i, color in enumerate(palette):
            # Convert BGR to Lab
            bgr = color.reshape(1, 1, 3).astype(np.uint8)
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).reshape(3)
            lab_palette[i] = lab
        
        # Calculate distance matrix
        distances = pairwise_distances(lab_palette)
        
        # Find pairs that are too close
        min_distance = 10  # Minimum L*a*b* distance
        for i in range(len(palette)):
            for j in range(i+1, len(palette)):
                if distances[i, j] < min_distance:
                    # Move colors apart in Lab space
                    direction = lab_palette[j] - lab_palette[i]
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        offset = (min_distance - distances[i, j]) / 2
                        
                        lab_palette[i] = lab_palette[i] - direction * offset
                        lab_palette[j] = lab_palette[j] + direction * offset
        
        # For dark fur enhancement, boost L value of dark colors
        if dark_enhancement > 0:
            for i, lab in enumerate(lab_palette):
                # Check if this is a dark color (low L value)
                if lab[0] < 50:  # L channel
                    # Boost L value based on dark_enhancement
                    boost = 10 * dark_enhancement
                    lab_palette[i][0] = min(100, lab[0] + boost)
        
        # Convert back to BGR
        optimized_palette = np.zeros_like(palette, dtype=np.uint8)
        for i, lab in enumerate(lab_palette):
            # Clip to valid Lab ranges
            l, a, b = lab
            l = max(0, min(100, l))
            a = max(-127, min(127, a))
            b = max(-127, min(127, b))
            
            # Convert Lab to BGR
            lab_color = np.array([[[l, a, b]]], dtype=np.float32)
            bgr = cv2.cvtColor(lab_color, cv2.COLOR_Lab2BGR).reshape(3)
            optimized_palette[i] = bgr
        
        return optimized_palette
    
    def _optimize_portrait_palette(self, palette):
        """Optimize palette specifically for portrait images"""
        logger.debug("Applying portrait-specific color optimizations")
        
        # For portraits, we want to:
        # 1. Ensure accurate skin tones
        # 2. Maintain good separation between similar facial features
        # 3. Preserve highlight/shadow detail in faces
        
        # Convert to HSV for easier skin tone detection
        hsv_palette = []
        for color in palette:
            # Convert BGR to RGB
            rgb = color[::-1]
            # Convert RGB to HSV
            h, s, v = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
            hsv_palette.append((h, s, v))
        
        # Identify potential skin tones
        # Skin tones typically have hue in certain ranges
        skin_indices = []
        for i, (h, s, v) in enumerate(hsv_palette):
            # Approximate skin tone hue range (reddish-yellowish)
            is_skin_tone = (0 <= h <= 0.1) or (0.8 <= h <= 1.0)
            
            # Moderate saturation
            is_skin_tone = is_skin_tone and (0.2 <= s <= 0.6)
            
            # Not too dark, not too bright
            is_skin_tone = is_skin_tone and (0.2 <= v <= 0.9)
            
            if is_skin_tone:
                skin_indices.append(i)
        
        # Enhance skin tone separation
        if len(skin_indices) > 0:
            # Get average skin tone
            avg_h = sum(hsv_palette[i][0] for i in skin_indices) / len(skin_indices)
            avg_s = sum(hsv_palette[i][1] for i in skin_indices) / len(skin_indices)
            
            # Adjust skin tones to be more natural
            for i in skin_indices:
                h, s, v = hsv_palette[i]
                
                # Reduce saturation variation
                s_new = s * 0.7 + avg_s * 0.3
                
                # Keep hue close to average but maintain differences
                h_diff = (h - avg_h + 0.5) % 1.0 - 0.5
                h_new = (avg_h + h_diff * 0.7) % 1.0
                
                # Update palette
                hsv_palette[i] = (h_new, s_new, v)
        
        # Convert back to BGR
        optimized_palette = []
        for h, s, v in hsv_palette:
            # Convert HSV to RGB
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            # Convert to BGR and 0-255 range
            optimized_palette.append(np.array([b*255, g*255, r*255], dtype=np.uint8))
        
        return np.array(optimized_palette)
    
    def _optimize_landscape_palette(self, palette):
        """Optimize palette specifically for landscape images"""
        logger.debug("Applying landscape-specific color optimizations")
        
        # For landscapes, we want to:
        # 1. Enhance separation between sky and land
        # 2. Preserve natural colors
        # 3. Group similar vegetation colors
        
        # Convert to HSV for easier natural color detection
        hsv_palette = []
        for color in palette:
            # Convert BGR to RGB
            rgb = color[::-1]
            # Convert RGB to HSV
            h, s, v = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
            hsv_palette.append((h, s, v))
        
        # Identify potential sky colors (blueish, low saturation)
        sky_indices = []
        # Identify potential vegetation colors (greenish)
        vegetation_indices = []
        
        for i, (h, s, v) in enumerate(hsv_palette):
            # Sky hues (blue range)
            if 0.5 <= h <= 0.7 and s <= 0.5:
                sky_indices.append(i)
                
            # Vegetation hues (green range)
            if 0.2 <= h <= 0.4 and s >= 0.2:
                vegetation_indices.append(i)
        
        # Group vegetation colors slightly
        if len(vegetation_indices) > 0:
            # Get average vegetation hue
            avg_h = sum(hsv_palette[i][0] for i in vegetation_indices) / len(vegetation_indices)
            
            # Adjust vegetation colors to be more cohesive
            for i in vegetation_indices:
                h, s, v = hsv_palette[i]
                
                # Move hue slightly towards average
                h_diff = (h - avg_h + 0.5) % 1.0 - 0.5
                h_new = (avg_h + h_diff * 0.7) % 1.0
                
                # Update palette
                hsv_palette[i] = (h_new, s, v)
        
        # Make sky colors more cohesive and slightly more saturated
        if len(sky_indices) > 0:
            for i in sky_indices:
                h, s, v = hsv_palette[i]
                
                # Slightly boost saturation for more vibrant skies
                s_new = min(1.0, s * 1.2)
                
                # Update palette
                hsv_palette[i] = (h, s_new, v)
        
        # Convert back to BGR
        optimized_palette = []
        for h, s, v in hsv_palette:
            # Convert HSV to RGB
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            # Convert to BGR and 0-255 range
            optimized_palette.append(np.array([b*255, g*255, r*255], dtype=np.uint8))
        
        return np.array(optimized_palette)
    
    def _enhance_dark_colors(self, palette, strength):
        """Enhance dark colors for better visibility"""
        logger.debug(f"Enhancing dark colors with strength {strength}")
        
        # Convert to Lab color space
        lab_palette = np.zeros_like(palette, dtype=np.float32)
        for i, color in enumerate(palette):
            # Convert BGR to Lab
            bgr = color.reshape(1, 1, 3).astype(np.uint8)
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).reshape(3)
            lab_palette[i] = lab
        
        # Enhance dark colors
        for i, lab in enumerate(lab_palette):
            # Check if this is a dark color (low L value)
            if lab[0] < 50:  # L channel
                # Calculate boost based on darkness and strength
                darkness = 1.0 - (lab[0] / 50.0)
                boost = darkness * strength * 15.0
                
                # Apply boost to L channel
                lab_palette[i][0] = min(100, lab[0] + boost)
                
                # Slightly increase a,b channels for more vibrance
                boost_ab = darkness * strength * 5.0
                lab_palette[i][1] += boost_ab
                lab_palette[i][2] += boost_ab
        
        # Convert back to BGR
        enhanced_palette = np.zeros_like(palette, dtype=np.uint8)
        for i, lab in enumerate(lab_palette):
            # Clip to valid Lab ranges
            l, a, b = lab
            l = max(0, min(100, l))
            a = max(-127, min(127, a))
            b = max(-127, min(127, b))
            
            # Convert Lab to BGR
            lab_color = np.array([[[l, a, b]]], dtype=np.float32)
            bgr = cv2.cvtColor(lab_color, cv2.COLOR_Lab2BGR).reshape(3)
            enhanced_palette[i] = bgr
        
        return enhanced_palette
    
    def _preserve_highlights(self, palette, preservation_level):
        """Preserve highlight details based on preservation level"""
        logger.debug(f"Preserving highlights with level {preservation_level}")
        
        # Set threshold based on preservation level
        if preservation_level == 'very_high':
            threshold = 80
        elif preservation_level == 'high':
            threshold = 90
        elif preservation_level == 'medium':
            threshold = 95
        else:  # low
            return palette  # No preservation needed
        
        # Convert to Lab color space
        lab_palette = np.zeros_like(palette, dtype=np.float32)
        for i, color in enumerate(palette):
            # Convert BGR to Lab
            bgr = color.reshape(1, 1, 3).astype(np.uint8)
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).reshape(3)
            lab_palette[i] = lab
        
        # Find highlight colors (high L value)
        highlight_indices = []
        for i, lab in enumerate(lab_palette):
            if lab[0] > threshold:  # L channel
                highlight_indices.append(i)
        
        # If we found highlight colors, ensure they're well separated
        if len(highlight_indices) > 0:
            # Calculate pairwise distances between highlight colors
            highlight_labs = lab_palette[highlight_indices]
            distances = pairwise_distances(highlight_labs)
            
            # Ensure minimum distance between highlight colors
            min_distance = 10  # Minimum L*a*b* distance
            for i in range(len(highlight_indices)):
                for j in range(i+1, len(highlight_indices)):
                    if distances[i, j] < min_distance:
                        # Increase distance by adjusting a,b values
                        idx1, idx2 = highlight_indices[i], highlight_indices[j]
                        
                        # Calculate direction vector in a,b space
                        direction = lab_palette[idx2][1:] - lab_palette[idx1][1:]
                        if np.linalg.norm(direction) > 0:
                            direction = direction / np.linalg.norm(direction)
                            offset = (min_distance - distances[i, j]) / 2
                            
                            # Move colors apart in a,b space
                            lab_palette[idx1][1:] = lab_palette[idx1][1:] - direction * offset
                            lab_palette[idx2][1:] = lab_palette[idx2][1:] + direction * offset
        
        # If we don't have enough highlight colors, adjust the brightest colors
        if len(highlight_indices) < 2 and len(palette) > 5:
            # Sort by brightness (L value)
            brightness_order = np.argsort([-lab[0] for lab in lab_palette])
            
            # Take top 2 and ensure they're bright enough
            for i in range(2):
                if i < len(brightness_order):
                    idx = brightness_order[i]
                    if lab_palette[idx][0] < threshold:
                        # Increase brightness
                        lab_palette[idx][0] = min(100, threshold + 5)
        
        # Convert back to BGR
        preserved_palette = np.zeros_like(palette, dtype=np.uint8)
        for i, lab in enumerate(lab_palette):
            # Clip to valid Lab ranges
            l, a, b = lab
            l = max(0, min(100, l))
            a = max(-127, min(127, a))
            b = max(-127, min(127, b))
            
            # Convert Lab to BGR
            lab_color = np.array([[[l, a, b]]], dtype=np.float32)
            bgr = cv2.cvtColor(lab_color, cv2.COLOR_Lab2BGR).reshape(3)
            preserved_palette[i] = bgr
        
        return preserved_palette
    
    def _assign_colors_to_segments(self, image, segments, palette, threshold, feature_map=None):
        """Assign colors to segments with adaptive thresholding and feature awareness"""
        logger.debug("Assigning colors to segments")
        
        # Create output colored segments
        colored_segments = np.zeros_like(segments)
        
        # For each segment, assign the best matching color
        unique_segments = np.unique(segments[segments > 0])
        
        for segment_id in unique_segments:
            # Create mask for this segment
            mask = segments == segment_id
            
            # Extract color information from the segment
            segment_pixels = image[mask]
            
            # Skip very small segments
            if len(segment_pixels) < 10:
                continue
            
            # Calculate average color of segment
            avg_color = np.mean(segment_pixels, axis=0)
            
            # Check if this segment contains important features
            if feature_map is not None:
                feature_importance = np.mean(feature_map[mask])
            else:
                feature_importance = 0
                
            # Apply adaptive threshold based on feature importance
            adaptive_threshold = threshold * (1.0 - feature_importance * 0.5)
            
            # Find best matching color in palette
            best_match = self._find_best_color_match(avg_color, palette, adaptive_threshold)
            
            # Assign color
            colored_segments[mask] = best_match
        
        return colored_segments
    
    def _find_best_color_match(self, color, palette, threshold):
        """Find best matching color in palette with adaptive thresholding"""
        # Convert color to Lab for better perceptual matching
        color_bgr = color.reshape(1, 1, 3).astype(np.uint8)
        color_lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2Lab).reshape(3)
        
        # Convert palette to Lab
        palette_lab = np.zeros((len(palette), 3), dtype=np.float32)
        for i, pal_color in enumerate(palette):
            bgr = pal_color.reshape(1, 1, 3).astype(np.uint8)
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).reshape(3)
            palette_lab[i] = lab
        
        # Calculate color distances
        distances = np.sqrt(np.sum((palette_lab - color_lab) ** 2, axis=1))
        
        # Find colors within threshold
        within_threshold = distances < threshold
        
        if np.any(within_threshold):
            # Return closest color within threshold
            closest_idx = np.argmin(distances)
            return closest_idx + 1  # +1 because 0 is background
        else:
            # If no colors within threshold, return closest
            closest_idx = np.argmin(distances)
            return closest_idx + 1  # +1 because 0 is background