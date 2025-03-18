import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology 

class NumberPlacer:
    def __init__(self):
        pass
    
    def place_numbers(self, label_image, region_data, style='classic'):
        """
        Place region ID numbers on a template image
        
        Parameters:
        - label_image: Image with region labels
        - region_data: List of dictionaries containing region information
        - style: Placement style ('classic', 'minimal', or 'detailed')
        
        Returns:
        - Image with numbers placed on it
        """
        # Create empty white image
        h, w = label_image.shape
        template = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        print(f"Placing numbers using {style} style")
        
        # Handle each placement style differently
        if style == 'minimal':
            # Minimal style: Only place numbers outside the shape
            self._place_numbers_minimal(template, region_data)
        elif style == 'detailed':
            # Detailed style: Place numbers inside each region
            self._place_numbers_detailed(template, label_image, region_data)
        else:
            # Classic style: Mixed approach
            self._place_numbers_classic(template, label_image, region_data)
        
        return template
    
    def _place_numbers_minimal(self, template, region_data):
        """Place numbers along the outside edges of the template"""
        h, w = template.shape[:2]
        
        # Sort regions by their centers (top to bottom, left to right)
        left_regions = sorted([r for r in region_data if r['center'][0] < w/3], 
                             key=lambda r: r['center'][1])
        
        right_regions = sorted([r for r in region_data if r['center'][0] > 2*w/3], 
                              key=lambda r: r['center'][1])
        
        top_regions = sorted([r for r in region_data if r['center'][1] < h/3 and r['center'][0] >= w/3 and r['center'][0] <= 2*w/3], 
                            key=lambda r: r['center'][0])
        
        bottom_regions = sorted([r for r in region_data if r['center'][1] > 2*h/3 and r['center'][0] >= w/3 and r['center'][0] <= 2*w/3], 
                               key=lambda r: r['center'][0])
        
        # Middle regions (place last if space permits)
        middle_regions = [r for r in region_data if r not in left_regions + right_regions + top_regions + bottom_regions]
        
        # Place numbers along the edges
        # Left edge
        y_offset = 30
        for r in left_regions:
            y_pos = min(max(30, r['center'][1]), h - 10)
            cv2.putText(template, str(r['id']), (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_offset = y_pos + 20
        
        # Right edge
        y_offset = 30
        for r in right_regions:
            y_pos = min(max(30, r['center'][1]), h - 10)
            text_size = cv2.getTextSize(str(r['id']), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.putText(template, str(r['id']), (w - text_size[0] - 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_offset = y_pos + 20
        
        # Top edge
        x_offset = 30
        for r in top_regions:
            x_pos = min(max(30, r['center'][0]), w - 30)
            cv2.putText(template, str(r['id']), (x_pos, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            x_offset = x_pos + 20
        
        # Bottom edge
        x_offset = 30
        for r in bottom_regions:
            x_pos = min(max(30, r['center'][0]), w - 30)
            cv2.putText(template, str(r['id']), (x_pos, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            x_offset = x_pos + 20

    def optimize_number_placement(self, label_image, region_data, min_font_size=8, 
                                max_font_size=15, preserve_features=True, feature_map=None):
        """
        Optimized version of number placement with better readability and feature preservation
        
        Parameters:
        - label_image: Image with region labels
        - region_data: Information about each region
        - min_font_size: Minimum font size for numbers
        - max_font_size: Maximum font size for numbers
        - preserve_features: Whether to avoid placing numbers over important features
        - feature_map: Optional map of important features to avoid
        
        Returns:
        - Dictionary mapping region labels to number placement coordinates and sizes
        """
        print("Optimizing number placement for better readability...")
        
        h, w = label_image.shape
        number_placements = {}
        
        # For each region, find the optimal placement for its number
        for region_label, data in enumerate(region_data):
            # Skip tiny regions that will be merged or discarded
            if data['area'] < 100:
                continue
                
            # Create a mask for this region
            region_mask = (label_image == data['id'])
            
            # Calculate appropriate font size based on region size
            region_size_factor = min(1.0, data['area'] / 10000)  # Cap at 1.0
            font_size = int(min_font_size + region_size_factor * (max_font_size - min_font_size))
            
            # Initialize best placement position and score
            best_pos = None
            best_score = float('-inf')
            
            # Get region's centroid as starting point
            centroid_x, centroid_y = data['center']
            
            # Generate candidate positions including centroid and distributed alternatives
            candidate_positions = []
            
            # Add centroid
            candidate_positions.append((centroid_x, centroid_y))
            
            # Add positions along the medial axis (skeleton)
            from skimage import morphology
            skeleton = morphology.medial_axis(region_mask)
            skeleton_points = np.where(skeleton)
            
            # Sample points along the skeleton
            if len(skeleton_points[0]) > 0:
                num_samples = min(10, len(skeleton_points[0]))
                sample_indices = np.linspace(0, len(skeleton_points[0]) - 1, num_samples).astype(int)
                for idx in sample_indices:
                    candidate_positions.append((skeleton_points[1][idx], skeleton_points[0][idx]))
            
            # Evaluate each candidate position
            for pos_x, pos_y in candidate_positions:
                # Check if position is within the label region
                pos_y_int, pos_x_int = int(pos_y), int(pos_x)
                if not (0 <= pos_y_int < h and 0 <= pos_x_int < w) or not region_mask[pos_y_int][pos_x_int]:
                    continue
                    
                # Calculate the space available around this position
                number_width = font_size * len(str(data['id'])) * 0.6
                number_height = font_size
                
                # Define the region that would be occupied by the number
                half_width = number_width / 2
                half_height = number_height / 2
                
                num_box_x1 = max(0, int(pos_x - half_width))
                num_box_y1 = max(0, int(pos_y - half_height))
                num_box_x2 = min(w - 1, int(pos_x + half_width))
                num_box_y2 = min(h - 1, int(pos_y + half_height))
                
                # Calculate how well the number fits within the region
                num_box = region_mask[num_box_y1:num_box_y2, num_box_x1:num_box_x2]
                fit_score = np.sum(num_box) / num_box.size if num_box.size > 0 else 0
                
                # Penalize if number would overlay important features
                feature_penalty = 0
                if preserve_features and feature_map is not None:
                    feature_box = feature_map[num_box_y1:num_box_y2, num_box_x1:num_box_x2]
                    feature_density = np.mean(feature_box) if feature_box.size > 0 else 0
                    feature_penalty = feature_density * 0.5  # Scale penalty
                
                # Calculate distance from centroid (prefer positions closer to center)
                distance_from_centroid = np.sqrt((pos_x - centroid_x)**2 + (pos_y - centroid_y)**2)
                distance_penalty = distance_from_centroid / np.sqrt(data['area']) * 0.1
                
                # Calculate final score
                score = fit_score - distance_penalty - feature_penalty
                
                # Update best position if this is better
                if score > best_score:
                    best_score = score
                    best_pos = (pos_x, pos_y)
            
            # If we found a valid position, add it to the results
            if best_pos is not None:
                number_placements[data['id']] = {
                    'position': best_pos,
                    'font_size': font_size,
                    'fit_score': best_score
                }
            else:
                # Fallback to centroid if no better position found
                number_placements[data['id']] = {
                    'position': (centroid_x, centroid_y),
                    'font_size': font_size,
                    'fit_score': 0.0
                }
        
        return number_placements
    def _place_numbers_detailed(self, template, label_image, region_data):
        """Place numbers inside each region"""
        # For each region
        for region in region_data:
            region_id = region['id']
            
            # Calculate text size
            text = str(region_id)
            font_scale = 0.4
            thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Adjust position to center of region
            cx, cy = region['center']
            
            # Place multiple numbers for large regions
            if region['area'] > 5000:  # Large region
                # Calculate how many numbers to place based on area
                num_placements = max(1, min(10, region['area'] // 5000))
                
                # Try to find good positions using distance transform or simple grid
                positions = self._find_placement_points(label_image, region, num_placements)
                
                # Place number at each position
                for x, y in positions:
                    # Check if position is valid
                    if 0 <= x < template.shape[1] and 0 <= y < template.shape[0]:
                        tx = max(0, min(template.shape[1] - text_width, x - text_width//2))
                        ty = max(text_height, min(template.shape[0], y + text_height//2))
                        cv2.putText(template, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
            else:
                # Smaller region - place one number
                # Check if center is valid
                tx = max(0, min(template.shape[1] - text_width, cx - text_width//2))
                ty = max(text_height, min(template.shape[0], cy + text_height//2))
                cv2.putText(template, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    
    def _place_numbers_classic(self, template, label_image, region_data):
        """Mixed approach - numbers inside large regions, one per small region"""
        # Sort regions by size - large to small
        sorted_regions = sorted(region_data, key=lambda r: r['area'], reverse=True)
        
        # Large regions (top 30%) - use detailed approach
        large_regions = sorted_regions[:max(1, len(sorted_regions) // 3)]
        small_regions = sorted_regions[len(large_regions):]
        
        # Use detailed approach for large regions
        for region in large_regions:
            region_id = region['id']
            
            # Calculate text size
            text = str(region_id)
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Place at center of region
            cx, cy = region['center']
            tx = max(0, min(template.shape[1] - text_width, cx - text_width//2))
            ty = max(text_height, min(template.shape[0], cy + text_height//2))
            
            cv2.putText(template, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        # For small regions, try to place numbers near the center but offset from each other
        # to avoid overlap
        grid_size = 20  # Grid cell size for collision detection
        h, w = template.shape[:2]
        grid = np.zeros((h // grid_size + 1, w // grid_size + 1), dtype=bool)
        
        for region in small_regions:
            region_id = region['id']
            
            # Calculate text size
            text = str(region_id)
            font_scale = 0.4
            thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Try to place near center
            cx, cy = region['center']
            
            # Check grid for collision
            gx, gy = cx // grid_size, cy // grid_size
            
            # Try center first, then nearby positions
            positions = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), 
                        (-1, -1), (1, -1), (-1, 1), (1, 1)]
            
            placed = False
            for dx, dy in positions:
                ngx, ngy = gx + dx, gy + dy
                
                # Check if position is valid and not occupied
                if 0 <= ngx < grid.shape[1] and 0 <= ngy < grid.shape[0] and not grid[ngy, ngx]:
                    # Found a spot
                    tx = ngx * grid_size
                    ty = ngy * grid_size + text_height
                    
                    # Ensure position is within bounds
                    if tx >= 0 and tx + text_width < w and ty >= text_height and ty < h:
                        cv2.putText(template, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
                        grid[ngy, ngx] = True
                        placed = True
                        break
            
            # If all nearby spots are taken, place at center anyway
            if not placed:
                tx = max(0, min(w - text_width, cx - text_width//2))
                ty = max(text_height, min(h, cy + text_height//2))
                cv2.putText(template, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    
    def _find_placement_points(self, label_image, region, num_points=3):
        """Find good points to place numbers in a region using distance transform"""
        # Create a binary mask for this region
        region_id = region['id'] - 1  # Adjust for 0-indexed label image
        mask = np.zeros(label_image.shape, dtype=np.uint8)
        
        # Try to use mask based on region ID if available
        if np.any(label_image == region_id):
            mask[label_image == region_id] = 255
        else:
            # Fallback - use a circle centered on region center
            cx, cy = region['center']
            radius = int(np.sqrt(region['area'] / np.pi))
            cv2.circle(mask, (cx, cy), radius, 255, -1)
        
        # Calculate distance transform to find points far from edges
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # Get positions for text placement
        positions = []
        
        if num_points == 1:
            # Just use the center
            positions.append(region['center'])
        else:
            # Find local maxima
            # Simple approach: blur and find peaks
            blurred = cv2.GaussianBlur(dist, (15, 15), 0)
            
            # Find the maximum distance point
            max_val = np.max(blurred)
            
            # If max value is too small, just use center
            if max_val < 5:
                positions.append(region['center'])
                return positions
                
            # Create a threshold at 70% of maximum distance

            threshold = 0.7 * max_val
            peaks = blurred > threshold
            
            # Dilate to get wider peaks and avoid too close placements
            kernel = np.ones((25, 25), np.uint8)
            dilated = cv2.dilate(peaks.astype(np.uint8), kernel)
            
            # Find connected components in the dilated peaks
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated)
            
            # Skip background (label 0)
            for i in range(1, min(num_labels, num_points + 1)):
                x, y = int(centroids[i][0]), int(centroids[i][1])
                positions.append((x, y))
            
            # If not enough points found, add the region center
            if len(positions) < num_points:
                positions.append(region['center'])
        
        return positions