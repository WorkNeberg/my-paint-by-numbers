import numpy as np
import cv2
import logging
import math
from scipy import ndimage
from enum import Enum

logger = logging.getLogger('pbn-app.number_placer')

class NumberSizeStrategy(Enum):
    UNIFORM = "uniform"
    PROPORTIONAL = "proportional"
    ADAPTIVE = "adaptive"

class NumberPlacementStrategy(Enum):
    CENTER = "center"
    WEIGHTED = "weighted"
    AVOID_FEATURES = "avoid_features"

class NumberOverlapStrategy(Enum):
    SHRINK = "shrink"
    MOVE = "move"
    PRIORITIZE = "prioritize"

class NumberPlacer:
    """
    Handles intelligent placement of numbers within paint-by-numbers regions
    """
    
    def __init__(self):
        """Initialize the number placer"""
        logger.info("NumberPlacer initialized")
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.default_font_scale = 0.5
        self.default_thickness = 1
        self.base_font_color = (0, 0, 0)  # Black
        self.outline_color = (255, 255, 255)  # White
        
    def place_numbers(self, label_image, settings, feature_map=None, colors=None):
        """
        Place numbers in regions based on settings
        
        Args:
            label_image: Image with region labels
            settings: Dictionary with number placement settings
            feature_map: Optional feature importance map
            colors: Optional list of region colors
            
        Returns:
            Image with placed numbers and data about placements
        """
        logger.info("Starting number placement")
        
        # Extract settings - handle both string and dictionary cases
        number_settings = settings.get('number_placement', {})
        
        # Handle case where number_settings is a string, not a dictionary
        if isinstance(number_settings, str):
            # Use string value as placement_strategy and use defaults for other settings
            placement_strategy = number_settings
            size_strategy = settings.get('number_size_strategy', 'proportional')
            contrast_level = settings.get('number_contrast', 'medium')
            legibility_priority = settings.get('number_legibility_priority', 0.5)
            min_number_size = settings.get('min_number_size', 10)
            overlap_strategy = settings.get('number_overlap_strategy', 'shrink')
        else:
            # Original code path for dictionary settings
            size_strategy = number_settings.get('number_size_strategy', 'proportional')
            placement_strategy = number_settings.get('number_placement', 'center')
            contrast_level = number_settings.get('number_contrast', 'medium')
            legibility_priority = number_settings.get('number_legibility_priority', 0.5)
            min_number_size = number_settings.get('min_number_size', 10)
            overlap_strategy = number_settings.get('number_overlap_strategy', 'shrink')
        
        # Create output image (3 channel for colored text)
        h, w = label_image.shape
        numbers_image = np.zeros((h, w, 4), dtype=np.uint8)
        numbers_image[:, :, 3] = 0  # Fully transparent
        
        # Get unique regions
        regions = np.unique(label_image)
        if 0 in regions:  # Skip background (0)
            regions = regions[1:]
            
        # Calculate region properties
        region_data = self._analyze_regions(label_image, regions)
        
        # Determine number sizes
        number_sizes = self._calculate_number_sizes(
            region_data, 
            size_strategy, 
            min_number_size,
            legibility_priority
        )
        
        # Determine number positions
        number_positions = self._calculate_number_positions(
            label_image,
            region_data,
            placement_strategy,
            feature_map
        )
        
        # Detect and resolve overlaps
        if len(regions) > 1:
            number_positions, number_sizes = self._resolve_overlaps(
                number_positions,
                number_sizes,
                overlap_strategy,
                region_data
            )
        
        # Determine appropriate font colors for contrast
        font_colors = self._determine_font_colors(region_data, contrast_level, colors)
        
        # Place numbers on the image
        number_metadata = []
        for region_id in regions:
            if region_id == 0:  # Skip background
                continue
                
            idx = region_id - 1  # Convert to 0-based index
            
            if idx >= len(number_positions) or idx >= len(number_sizes):
                continue  # Skip if data is missing
                
            pos = number_positions[idx]
            size = number_sizes[idx]
            color = font_colors[idx] if idx < len(font_colors) else self.base_font_color
            
            # Place the number with outline for better visibility
            self._place_number_with_outline(
                numbers_image,
                str(region_id),
                pos,
                size,
                color
            )
            
            # Store metadata
            number_metadata.append({
                'region_id': int(region_id),
                'position': (int(pos[0]), int(pos[1])),
                'size': float(size),
                'color': color
            })
        
        logger.info(f"Placed {len(number_metadata)} numbers")
        return numbers_image, number_metadata
        
    def _analyze_regions(self, label_image, regions):
        """Analyze region properties"""
        logger.debug("Analyzing region properties")
        
        region_data = []
        
        for region_id in regions:
            mask = (label_image == region_id)
            
            # Skip empty regions
            if not np.any(mask):
                region_data.append({
                    'region_id': region_id,
                    'area': 0,
                    'centroid': (0, 0),
                    'width': 0,
                    'height': 0,
                    'bbox': (0, 0, 0, 0),
                    'aspect_ratio': 1.0,
                    'complexity': 0.0,
                    'avg_color': None
                })
                continue
            
            # Calculate region properties
            area = np.sum(mask)
            
            # Calculate centroid
            y_coords, x_coords = np.nonzero(mask)
            centroid_y = np.mean(y_coords)
            centroid_x = np.mean(x_coords)
            
            # Calculate bounding box
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            width = max_x - min_x + 1
            height = max_y - min_y + 1
            bbox = (min_x, min_y, width, height)
            
            # Calculate aspect ratio
            aspect_ratio = width / height if height > 0 else 1.0
            
            # Calculate region complexity
            perimeter = self._calculate_perimeter(mask)
            # Use isoperimetric quotient (4π*area/perimeter²)
            complexity = (4 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 1.0
            complexity = 1.0 - complexity  # Invert so higher value = more complex
            
            region_data.append({
                'region_id': region_id,
                'area': area,
                'centroid': (centroid_x, centroid_y),
                'width': width,
                'height': height,
                'bbox': bbox,
                'aspect_ratio': aspect_ratio,
                'complexity': complexity
            })
            
        return region_data
        
    def _calculate_perimeter(self, mask):
        """Calculate the perimeter of a binary mask"""
        # Use morphological gradient to find perimeter
        struct = ndimage.generate_binary_structure(2, 2)
        dilated = ndimage.binary_dilation(mask, struct)
        perimeter = np.sum(dilated) - np.sum(mask)
        return perimeter
        
    def _calculate_number_sizes(self, region_data, size_strategy, min_size, legibility_priority):
        """Calculate appropriate font sizes based on region properties"""
        logger.debug(f"Calculating number sizes using {size_strategy} strategy")
        
        number_sizes = []
        
        # Get areas
        areas = np.array([r['area'] for r in region_data])
        
        # Skip empty regions
        if len(areas) == 0 or np.max(areas) == 0:
            return number_sizes
            
        # Normalize areas to range 0-1
        norm_areas = areas / np.max(areas)
        
        # Calculate base size scaling factor
        min_area_factor = 0.3  # Minimum size factor for smallest region
        
        for i, region in enumerate(region_data):
            area = region['area']
            complexity = region['complexity']
            width = region['width']
            height = region['height']
            
            # Start with base size calculation based on strategy
            if size_strategy == NumberSizeStrategy.UNIFORM.value:
                # Uniform size for all numbers
                size = self.default_font_scale * 1.5
                
            elif size_strategy == NumberSizeStrategy.PROPORTIONAL.value:
                # Size proportional to square root of normalized area
                size = min_area_factor + (1.0 - min_area_factor) * np.sqrt(norm_areas[i])
                size *= self.default_font_scale * 2.0
                
            elif size_strategy == NumberSizeStrategy.ADAPTIVE.value:
                # Adaptive sizing based on area and complexity
                # For complex shapes, we need to be more careful with size
                complexity_factor = 1.0 - complexity * 0.5  # Reduce size for complex shapes
                size = min_area_factor + (1.0 - min_area_factor) * np.sqrt(norm_areas[i])
                size *= self.default_font_scale * 2.5 * complexity_factor
                
                # Consider region shape too
                min_dimension = min(width, height)
                if min_dimension > 0:
                    # Ensure number won't be too big for the region
                    text_height = min_dimension * 0.7  # Leave some margin
                    # Convert to font scale
                    text_size_factor = text_height / 20.0  # Approximate conversion
                    size = min(size, text_size_factor)
            else:
                # Default to proportional
                size = min_area_factor + (1.0 - min_area_factor) * np.sqrt(norm_areas[i])
                size *= self.default_font_scale * 2.0
                
            # Apply legibility priority (increase size for better legibility)
            legibility_boost = 1.0 + legibility_priority * 0.5
            size *= legibility_boost
            
            # Ensure minimum size
            min_font_size = min_size / 20.0  # Convert pixel size to font scale
            size = max(size, min_font_size)
            
            number_sizes.append(size)
            
        return number_sizes
        
    def _calculate_number_positions(self, label_image, region_data, placement_strategy, feature_map=None):
        """Calculate optimal number positions based on placement strategy"""
        logger.debug(f"Calculating number positions using {placement_strategy} strategy")
        
        positions = []
        
        for region in region_data:
            centroid = region['centroid']
            region_id = region['region_id']
            area = region['area']
            
            # Skip empty regions
            if area == 0:
                positions.append((0, 0))
                continue
                
            if placement_strategy == NumberPlacementStrategy.CENTER.value:
                # Use region centroid
                positions.append(centroid)
                
            elif placement_strategy == NumberPlacementStrategy.WEIGHTED.value:
                # Find position in widest part of region
                pos = self._find_widest_point(label_image, region_id)
                positions.append(pos)
                
            elif placement_strategy == NumberPlacementStrategy.AVOID_FEATURES.value and feature_map is not None:
                # Avoid important features if possible
                pos = self._find_position_avoiding_features(label_image, region_id, feature_map)
                positions.append(pos)
                
            else:
                # Default to centroid
                positions.append(centroid)
                
        return positions
        
    def _find_widest_point(self, label_image, region_id):
        """Find the widest point of a region using distance transform"""
        # Create mask for this region
        mask = (label_image == region_id)
        
        # Skip empty regions
        if not np.any(mask):
            return (0, 0)
            
        # Calculate distance transform (distance to nearest boundary)
        dist_transform = ndimage.distance_transform_edt(mask)
        
        # Find the point with maximum distance (widest point)
        max_idx = np.argmax(dist_transform)
        max_point = np.unravel_index(max_idx, dist_transform.shape)
        
        # Convert to (x, y) coordinates
        return (max_point[1], max_point[0])
        
    def _find_position_avoiding_features(self, label_image, region_id, feature_map):
        """Find a position that avoids important features"""
        # Create mask for this region
        mask = (label_image == region_id)
        
        # Skip empty regions
        if not np.any(mask):
            return (0, 0)
            
        # Calculate centroid as fallback
        y_coords, x_coords = np.nonzero(mask)
        centroid_y = np.mean(y_coords)
        centroid_x = np.mean(x_coords)
        centroid = (centroid_x, centroid_y)
        
        # If no feature map, return centroid
        if feature_map is None:
            return centroid
            
        # Create an inverted feature importance map (high values = good for number placement)
        feature_mask = mask * (1.0 - feature_map)
        
        # Also consider distance from edges
        dist_transform = ndimage.distance_transform_edt(mask)
        normalized_dist = dist_transform / np.max(dist_transform) if np.max(dist_transform) > 0 else dist_transform
        
        # Combine distance and feature avoidance
        placement_score = feature_mask * normalized_dist
        
        # If all scores are zero, return centroid
        if np.max(placement_score) == 0:
            return centroid
            
        # Find best position
        best_idx = np.argmax(placement_score)
        best_point = np.unravel_index(best_idx, placement_score.shape)
        
        # Convert to (x, y) coordinates
        return (best_point[1], best_point[0])
        
    def _resolve_overlaps(self, positions, sizes, strategy, region_data):
        """Detect and resolve number placement overlaps"""
        logger.debug(f"Resolving overlaps using {strategy} strategy")
        
        if not positions or not sizes:
            return positions, sizes
            
        # Create copies for modification
        new_positions = positions.copy()
        new_sizes = sizes.copy()
        
        # Get number sizes in pixels for collision detection
        pixel_sizes = []
        for i, size in enumerate(sizes):
            # Estimate text size
            text = str(region_data[i]['region_id'])
            (text_width, text_height), _ = cv2.getTextSize(
                text, self.font, size, self.default_thickness
            )
            pixel_sizes.append((text_width, text_height))
        
        # Check for overlapping numbers
        num_numbers = len(positions)
        for i in range(num_numbers):
            pos_i = new_positions[i]
            size_i = new_sizes[i]
            width_i, height_i = pixel_sizes[i]
            
            # Skip if invalid position
            if pos_i[0] == 0 and pos_i[1] == 0:
                continue
            
            rect_i = (
                pos_i[0] - width_i/2, 
                pos_i[1] - height_i/2, 
                width_i, 
                height_i
            )
            
            for j in range(i+1, num_numbers):
                pos_j = new_positions[j]
                size_j = new_sizes[j]
                width_j, height_j = pixel_sizes[j]
                
                # Skip if invalid position
                if pos_j[0] == 0 and pos_j[1] == 0:
                    continue
                
                rect_j = (
                    pos_j[0] - width_j/2, 
                    pos_j[1] - height_j/2, 
                    width_j, 
                    height_j
                )
                
                # Check if rectangles overlap
                if self._check_rect_overlap(rect_i, rect_j):
                    # Resolve overlap based on strategy
                    if strategy == NumberOverlapStrategy.SHRINK.value:
                        # Shrink both numbers
                        new_sizes[i] *= 0.85
                        new_sizes[j] *= 0.85
                        
                        # Update sizes in pixels for future checks
                        text_i = str(region_data[i]['region_id'])
                        (width_i, height_i), _ = cv2.getTextSize(
                            text_i, self.font, new_sizes[i], self.default_thickness
                        )
                        pixel_sizes[i] = (width_i, height_i)
                        
                        text_j = str(region_data[j]['region_id'])
                        (width_j, height_j), _ = cv2.getTextSize(
                            text_j, self.font, new_sizes[j], self.default_thickness
                        )
                        pixel_sizes[j] = (width_j, height_j)
                        
                    elif strategy == NumberOverlapStrategy.MOVE.value:
                        # Move numbers away from each other
                        # Calculate direction vector
                        dx = pos_j[0] - pos_i[0]
                        dy = pos_j[1] - pos_i[1]
                        distance = np.sqrt(dx*dx + dy*dy)
                        
                        if distance > 0:
                            # Normalize and scale
                            dx /= distance
                            dy /= distance
                            
                            # Required separation
                            required_separation = (width_i + width_j) / 2 * 1.1  # Add 10% margin
                            
                            # Move proportionally to region areas
                            area_i = region_data[i]['area']
                            area_j = region_data[j]['area']
                            total_area = area_i + area_j
                            
                            if total_area > 0:
                                move_i = required_separation * (area_j / total_area)
                                move_j = required_separation * (area_i / total_area)
                                
                                # Update positions
                                new_positions[i] = (
                                    pos_i[0] - dx * move_i,
                                    pos_i[1] - dy * move_i
                                )
                                new_positions[j] = (
                                    pos_j[0] + dx * move_j,
                                    pos_j[1] + dy * move_j
                                )
                        
                    elif strategy == NumberOverlapStrategy.PRIORITIZE.value:
                        # Prioritize number with larger region
                        area_i = region_data[i]['area']
                        area_j = region_data[j]['area']
                        
                        if area_i > area_j:
                            # Shrink j
                            new_sizes[j] *= 0.7
                        else:
                            # Shrink i
                            new_sizes[i] *= 0.7
                        
                        # Update sizes in pixels
                        text_i = str(region_data[i]['region_id'])
                        (width_i, height_i), _ = cv2.getTextSize(
                            text_i, self.font, new_sizes[i], self.default_thickness
                        )
                        pixel_sizes[i] = (width_i, height_i)
                        
                        text_j = str(region_data[j]['region_id'])
                        (width_j, height_j), _ = cv2.getTextSize(
                            text_j, self.font, new_sizes[j], self.default_thickness
                        )
                        pixel_sizes[j] = (width_j, height_j)
        
        return new_positions, new_sizes
        
    def _check_rect_overlap(self, rect1, rect2):
        """Check if two rectangles overlap"""
        # rect format: (x, y, width, height) where (x,y) is the top-left
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        # Check if one rectangle is to the left of the other
        if x1 + w1 <= x2 or x2 + w2 <= x1:
            return False
            
        # Check if one rectangle is above the other
        if y1 + h1 <= y2 or y2 + h2 <= y1:
            return False
            
        # Rectangles overlap
        return True
        
    def _determine_font_colors(self, region_data, contrast_level, colors=None):
        """Determine font colors for best contrast"""
        logger.debug(f"Determining font colors with contrast level: {contrast_level}")
        
        font_colors = []
        
        # If no colors provided, use default black
        if colors is None:
            return [self.base_font_color] * len(region_data)
            
        for i, region in enumerate(region_data):
            region_id = region['region_id']
            
            # If region_id is out of range for colors array
            if region_id <= 0 or region_id > len(colors):
                font_colors.append(self.base_font_color)
                continue
                
            # Get region color (assuming BGR format)
            color = colors[region_id-1]
            
            # Calculate luminance
            # Convert BGR to RGB
            r, g, b = color[2]/255.0, color[1]/255.0, color[0]/255.0
            
            # Relative luminance (ITU-R BT.709)
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            
            # Choose text color based on contrast level and background luminance
            if contrast_level == "very_high":
                # Maximum contrast - black on light, white on dark
                font_color = (0, 0, 0) if luminance > 0.5 else (255, 255, 255)
                
            elif contrast_level == "high":
                # High contrast - similar to very high but with some gradation
                if luminance > 0.7:
                    font_color = (0, 0, 0)  # Black on very light
                elif luminance < 0.3:
                    font_color = (255, 255, 255)  # White on very dark
                else:
                    # For middle range, choose the more contrasting option
                    font_color = (0, 0, 0) if luminance > 0.5 else (255, 255, 255)
                    
            elif contrast_level == "medium":
                # Medium contrast - softer colors
                if luminance > 0.7:
                    font_color = (40, 40, 40)  # Dark gray on light
                elif luminance < 0.3:
                    font_color = (215, 215, 215)  # Light gray on dark
                else:
                    # For middle range colors, check individual channels
                    # and use complementary-ish colors
                    font_color = []
                    for c in color:
                        # Invert with some moderation
                        font_color.append(255 - c if c > 128 else 0)
                    font_color = tuple(font_color)
                    
            else:  # low
                # Low contrast - subtle difference
                if luminance > 0.8:
                    font_color = (60, 60, 60)  # Softer dark on very light
                elif luminance < 0.2:
                    font_color = (200, 200, 200)  # Softer light on very dark
                else:
                    # For middle range, use a slightly shifted color
                    font_color = []
                    for c in color:
                        # Small shift
                        font_color.append(max(0, min(255, c - 40 if c > 128 else c + 40)))
                    font_color = tuple(font_color)
            
            font_colors.append(font_color)
            
        return font_colors
        
    def _place_number_with_outline(self, image, text, position, font_scale, color):
        """Place a number with outline for better visibility"""
        thickness = self.default_thickness
        
        # Calculate text size for centering
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, font_scale, thickness
        )
        
        # Adjust position to center text
        text_x = int(position[0] - text_width / 2)
        text_y = int(position[1] + text_height / 2)
        
        # Draw white outline
        outline_thickness = thickness + 1
        cv2.putText(
            image, text, (text_x, text_y), 
            self.font, font_scale, self.outline_color, 
            outline_thickness, cv2.LINE_AA
        )
        
        # Draw text in specified color
        cv2.putText(
            image, text, (text_x, text_y), 
            self.font, font_scale, color, 
            thickness, cv2.LINE_AA
        )
        
        # Update alpha channel (make text area opaque)
        # Create text mask
        text_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.putText(
            text_mask, text, (text_x, text_y), 
            self.font, font_scale, 255, 
            outline_thickness, cv2.LINE_AA
        )
        
        # Apply mask to alpha channel
        image[:,:,3] = np.maximum(image[:,:,3], text_mask)