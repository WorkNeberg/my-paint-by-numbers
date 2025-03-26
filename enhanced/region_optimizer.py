import cv2
import numpy as np
import logging
from collections import defaultdict
from scipy import ndimage
import time

logger = logging.getLogger('pbn-app.region_optimizer')

class RegionOptimizer:
    """
    Optimizes regions for better paintability and visual quality
    """
    
    def __init__(self):
        """Initialize the region optimizer"""
        logger.info("RegionOptimizer initialized")
        
    def optimize_regions(self, segments, settings, feature_map=None):
        """
        Optimize segmented regions based on settings
        
        Args:
            segments: Segmented image with region labels
            settings: Dictionary with optimization settings
            feature_map: Optional feature importance map
            
        Returns:
            Optimized segments
        """
        start_time = time.time()
        logger.info("Starting region optimization")
        
        # Extract optimization settings
        merge_level = settings.get('merge_regions_level', 'normal')
        simplification = settings.get('simplification_level', 'medium')
        
        # Convert to numeric values
        merge_values = {'low': 0.25, 'normal': 0.5, 'aggressive': 0.8}
        merge_strength = merge_values.get(merge_level, 0.5)
        
        simplification_values = {'low': 0.25, 'medium': 0.5, 'high': 0.75, 'very_high': 0.9}
        simplify_strength = simplification_values.get(simplification, 0.5)
        
        # Step 1: Handle small regions
        segments = self._handle_small_regions(segments, merge_strength)
        
        # Step 2: Simplify region boundaries
        segments = self._simplify_boundaries(segments, simplify_strength)
        
        # Step 3: Handle special features if feature map is provided
        if feature_map is not None:
            segments = self._preserve_features(segments, feature_map)
            
        # Step 4: Final cleanup
        segments = self._cleanup_regions(segments)
        
        # Calculate timing
        elapsed = time.time() - start_time
        logger.info(f"Region optimization completed in {elapsed:.2f}s")
        
        return segments
        
    def _handle_small_regions(self, segments, merge_strength):
        """
        Handle small regions by merging them with neighbors
        
        Args:
            segments: Segmented image
            merge_strength: How aggressively to merge (0.0-1.0)
            
        Returns:
            Optimized segments
        """
        h, w = segments.shape
        result = segments.copy()
        
        # Calculate region areas
        regions = np.unique(segments)
        region_areas = {}
        
        for region in regions:
            if region >= 0:  # Skip boundary label (-1)
                area = np.sum(segments == region)
                region_areas[region] = area
        
        # Calculate total area and median area
        total_area = h * w
        median_area = np.median(list(region_areas.values()))
        
        # Set size threshold based on merge_strength (smaller = more aggressive merging)
        # Base threshold on image size and merge_strength
        min_area_pct = 0.001 + merge_strength * 0.004  # 0.1% to 0.5% of image
        min_area = max(50, int(total_area * min_area_pct))  # At least 50 pixels
        
        # Identify small regions to merge
        small_regions = [r for r, area in region_areas.items() if area < min_area]
        logger.debug(f"Found {len(small_regions)} small regions to merge")
        
        # Sort small regions by size (smallest first)
        small_regions.sort(key=lambda r: region_areas[r])
        
        # For each small region
        for region in small_regions:
            # Skip if this region was already merged
            if region not in np.unique(result):
                continue
                
            # Find adjacent regions
            mask = result == region
            dilated = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
            neighbors = np.unique(result[dilated > 0])
            
            # Remove current region and boundary from neighbors
            neighbors = [n for n in neighbors if n != region and n >= 0]
            
            if neighbors:
                # Find best neighbor (largest shared boundary)
                best_neighbor = None
                best_count = 0
                
                for neighbor in neighbors:
                    # Dilate neighbor and count overlap with mask
                    neighbor_mask = result == neighbor
                    neighbor_dilated = cv2.dilate(
                        neighbor_mask.astype(np.uint8), 
                        np.ones((3, 3), np.uint8)
                    )
                    overlap = np.sum(neighbor_dilated & mask)
                    
                    if overlap > best_count:
                        best_count = overlap
                        best_neighbor = neighbor
                
                # Merge with best neighbor
                if best_neighbor is not None:
                    result[mask] = best_neighbor
        
        return result
        
    def _simplify_boundaries(self, segments, simplify_strength):
        """
        Simplify region boundaries to make them smoother
        
        Args:
            segments: Segmented image
            simplify_strength: How much to simplify (0.0-1.0)
            
        Returns:
            Segments with simplified boundaries
        """
        # Skip if no simplification
        if simplify_strength <= 0.01:
            return segments
            
        result = segments.copy()
        
        # Calculate boundary simplification iterations based on strength
        iterations = max(1, int(simplify_strength * 3))
        
        # Get unique regions
        regions = np.unique(result)
        regions = regions[regions >= 0]  # Skip boundary label (-1)
        
        # Process each region
        for region in regions:
            # Create mask for this region
            mask = result == region
            
            # Apply morphological operations to simplify boundary
            # Open operation removes small protrusions
            # Close operation fills small intrusions
            kernel_size = 3 + int(simplify_strength * 4)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            # Apply opening and closing
            simplified = mask
            for _ in range(iterations):
                simplified = cv2.morphologyEx(
                    simplified.astype(np.uint8), 
                    cv2.MORPH_OPEN, 
                    kernel
                )
                simplified = cv2.morphologyEx(
                    simplified, 
                    cv2.MORPH_CLOSE, 
                    kernel
                )
            
            # Update only the pixels that changed
            changed = (simplified > 0) != mask
            result[changed & (simplified > 0)] = region
            
        # Handle overlaps and gaps from independent processing
        # Apply watershed to clean up
        markers = np.ones_like(segments) * -1  # All background initially
        
        # Set seed points in the center of each region
        for region in regions:
            mask = result == region
            if np.any(mask):
                # Use distance transform to find centers
                dist = ndimage.distance_transform_edt(mask)
                # Get maximum distance point
                y, x = np.unravel_index(np.argmax(dist), dist.shape)
                markers[y, x] = region
                
        # Fix the watershed input format
        if len(segments.shape) == 2:
            # Create a proper 3-channel image as required by watershed
            watershed_image = np.zeros((segments.shape[0], segments.shape[1], 3), dtype=np.uint8)
        else:
            # If already multi-channel, ensure it's the right type
            watershed_image = segments.astype(np.uint8)
        
        # Ensure markers are proper format (int32)
        markers = markers.astype(np.int32)
        
        try:
            cv2.watershed(watershed_image, markers)
        except cv2.error as e:
            logger.error(f"Watershed error: {e}")
            # Fallback to original segments if watershed fails
            return segments
        
        # Convert watershed result back to segments
        final_result = np.zeros_like(segments)
        for region in regions:
            final_result[markers == region] = region
            
        return final_result
        
    def _preserve_features(self, segments, feature_map):
        """
        Preserve important features in the segmentation
        
        Args:
            segments: Segmented image
            feature_map: Feature importance map
            
        Returns:
            Segments with preserved features
        """
        # Skip if no feature map
        if feature_map is None:
            return segments
            
        # Threshold feature map to find important areas
        threshold = 0.5  # Adjust based on feature importance scale
        important_areas = feature_map > threshold
        
        # If no important areas, return original
        if not np.any(important_areas):
            return segments
            
        # Find regions with important features
        important_labels = np.unique(segments[important_areas])
        
        # Now just return the segments - we're not modifying them here
        # but in a real implementation, you could apply special handling
        # to regions with important features
        
        return segments
        
    def _cleanup_regions(self, segments):
        """
        Final cleanup of segmented regions
        
        Args:
            segments: Segmented image
            
        Returns:
            Cleaned-up segments
        """
        # Get unique region labels
        regions = np.unique(segments)
        regions = regions[regions >= 0]  # Skip boundary label (-1)
        
        # Create clean result
        result = np.zeros_like(segments)
        
        # Relabel regions to be consecutive integers starting from 0
        for i, region in enumerate(regions):
            result[segments == region] = i
            
        return result