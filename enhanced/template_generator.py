import cv2
import numpy as np
import logging
import time
from enum import Enum

logger = logging.getLogger('pbn-app.template_generator')

class TemplateStyle(Enum):
    CLASSIC = "classic"
    MINIMAL = "minimal"
    DETAILED = "detailed"
    ARTISTIC = "artistic"
    SKETCH = "sketch"
    BOLD = "bold"
    CLEAN = "clean"

class TemplateGenerator:
    """
    Enhanced template generator that produces high-quality paint-by-numbers templates
    with advanced segmentation and styling options.
    """
    
    def __init__(self):
        """Initialize the template generator"""
        logger.info("Initializing TemplateGenerator")
        self.last_settings = {}
        
    def generate_template(self, image, settings, feature_map=None):
        """
        Generate a complete paint-by-numbers template based on settings
        
        Args:
            image: Input image as numpy array (RGB format)
            settings: Dictionary with template generation settings
            feature_map: Optional feature importance map
            
        Returns:
            Dictionary containing template, segments, palette and metadata
        """
        self.last_settings = settings
        start_time = time.time()
        logger.info("Starting template generation")
        
        # Extract core settings
        num_colors = settings.get('colors', 15)
        edge_strength = settings.get('edge_strength', 1.0)
        edge_width = settings.get('edge_width', 2)
        simplification_level = settings.get('simplification_level', 'medium')
        edge_style = settings.get('edge_style', 'standard')
        merge_regions_level = settings.get('merge_regions_level', 'normal')
        
        # Convert simplification level to numeric value
        simplification_values = {
            'low': 0.25,
            'medium': 0.5, 
            'high': 0.75,
            'very_high': 0.9
        }
        simplification = simplification_values.get(simplification_level, 0.5)
        
        # Convert merge regions level to numeric value
        merge_values = {
            'low': 0.25,
            'normal': 0.5,
            'aggressive': 0.8
        }
        merge_level = merge_values.get(merge_regions_level, 0.5)
        
        # Step 1: Apply advanced segmentation to create regions
        segments, segment_stats = self._segment_image(
            image, num_colors, simplification, merge_level, feature_map
        )
        
        # Step 2: Extract and optimize color palette
        palette = self._extract_palette(image, segments, num_colors)
        
        # Step 3: Enhance and refine edges
        edges = self._enhance_edges(
            segments, edge_strength, edge_width, edge_style, feature_map
        )
        
        # Step 4: Create colored segments for visualization
        colored_segments = self._color_segments(segments, palette)
        
        # Step 5: Generate final template with styled appearance
        template_style = settings.get('template_style', 'classic')
        template = self._apply_template_style(
            colored_segments, edges, segments, palette, template_style
        )
        
        # Calculate timing
        elapsed = time.time() - start_time
        logger.info(f"Template generation completed in {elapsed:.2f}s")
        
        # Return results
        return {
            'template': template,
            'segments': segments,
            'colored_segments': colored_segments,
            'edges': edges,
            'palette': palette,
            'segment_stats': segment_stats,
            'processing_time': elapsed
        }
        
    def _segment_image(self, image, num_colors, simplification, merge_level, feature_map=None):
        """
        Apply advanced segmentation using watershed and intelligent merging
        
        Args:
            image: Input image
            num_colors: Target number of colors
            simplification: Level of simplification (0.0-1.0)
            merge_level: Level of region merging (0.0-1.0)
            feature_map: Optional feature importance map
            
        Returns:
            Segmented image and segment statistics
        """
        logger.info(f"Segmenting image with {num_colors} colors, simplification {simplification:.2f}")
        
        # Convert to LAB color space for better perceptual results
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply bilateral filtering for edge-aware smoothing
        # Adjust filter parameters based on simplification level
        # Higher simplification = stronger smoothing
        d = int(10 + simplification * 15)  # Filter diameter
        sigma_color = 20 + simplification * 60  # Range sigma
        sigma_space = 20 + simplification * 30  # Spatial sigma
        
        filtered = cv2.bilateralFilter(lab_image, d, sigma_color, sigma_space)
        
        # Apply feature-aware filtering if feature map provided
        if feature_map is not None:
            # Scale feature importance to determine amount of filtering
            # High importance areas get less filtering
            importance_factor = 1.0 - np.clip(feature_map, 0, 1)
            
            # Create a 3-channel importance map
            importance_3ch = np.stack([importance_factor] * 3, axis=2)
            
            # Blend filtered and original based on importance
            lab_image = (filtered * importance_3ch + lab_image * (1 - importance_3ch)).astype(np.uint8)
        else:
            lab_image = filtered
            
        # Apply mean shift segmentation for initial clustering
        # Adjust spatial radius based on simplification
        spatial_radius = int(10 + simplification * 30)
        color_radius = int(10 + simplification * 40)
        
        # Mean shift segmentation
        segmented = cv2.pyrMeanShiftFiltering(lab_image, spatial_radius, color_radius)
        
        # Convert back to RGB for K-means
        rgb_segmented = cv2.cvtColor(segmented, cv2.COLOR_LAB2RGB)
        
        # Reshape the image for k-means clustering
        pixels = rgb_segmented.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        # Determine effective number of colors based on simplification
        effective_colors = max(5, int(num_colors * (1.0 - simplification * 0.3)))
        
        # Apply k-means clustering to find color centers
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        _, labels, centers = cv2.kmeans(pixels, effective_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to integer labels
        labels = labels.flatten()
        
        # Reshape labels to original image dimensions
        segments = labels.reshape(image.shape[:2])
        
        # Apply region merging based on merge_level
        if merge_level > 0:
            segments = self._merge_similar_regions(segments, centers, merge_level)
            
        # Apply watershed to refine boundaries
        segments = self._refine_with_watershed(image, segments)
        
        # Calculate segment statistics
        segment_stats = self._calculate_segment_stats(segments)
        
        return segments, segment_stats
        
    def _merge_similar_regions(self, segments, color_centers, merge_level):
        """
        Merge similar adjacent regions based on color similarity
        
        Args:
            segments: Segmented image with labels
            color_centers: Color centers from k-means
            merge_level: How aggressively to merge (0.0-1.0)
            
        Returns:
            Optimized segments with merged regions
        """
        # Calculate color distances between all centers
        num_centers = len(color_centers)
        distances = np.zeros((num_centers, num_centers))
        
        for i in range(num_centers):
            for j in range(i+1, num_centers):
                # Euclidean distance in RGB space
                dist = np.linalg.norm(color_centers[i] - color_centers[j])
                distances[i, j] = dist
                distances[j, i] = dist
                
        # Determine merging threshold based on merge_level
        # Higher merge_level = more merging
        avg_distance = np.mean(distances[distances > 0])
        threshold = avg_distance * (0.3 + merge_level * 0.7)
        
        # Create adjacency map
        h, w = segments.shape
        adjacency = {}
        
        # Find adjacent regions
        for y in range(1, h):
            for x in range(1, w):
                current = segments[y, x]
                above = segments[y-1, x]
                left = segments[y, x-1]
                
                if current != above:
                    if current not in adjacency:
                        adjacency[current] = set()
                    if above not in adjacency:
                        adjacency[above] = set()
                    adjacency[current].add(above)
                    adjacency[above].add(current)
                    
                if current != left:
                    if current not in adjacency:
                        adjacency[current] = set()
                    if left not in adjacency:
                        adjacency[left] = set()
                    adjacency[current].add(left)
                    adjacency[left].add(current)
        
        # Identify regions to merge
        merges = []
        for region in adjacency:
            for adjacent in adjacency[region]:
                if region < adjacent:  # Avoid duplicates
                    if distances[region, adjacent] < threshold:
                        merges.append((region, adjacent))
        
        # Sort merges by distance (merge most similar first)
        merges.sort(key=lambda pair: distances[pair[0], pair[1]])
        
        # Apply merges
        label_map = {i: i for i in range(num_centers)}
        for src, dst in merges:
            # Map all regions that point to src to now point to dst
            for i in range(num_centers):
                if label_map[i] == label_map[src]:
                    label_map[i] = label_map[dst]
        
        # Create new segments with merged labels
        new_segments = np.zeros_like(segments)
        for i in range(num_centers):
            new_segments[segments == i] = label_map[i]
        
        # Relabel segments to be consecutive
        unique_labels = np.unique(new_segments)
        relabel_map = {label: i for i, label in enumerate(unique_labels)}
        for old_label, new_label in relabel_map.items():
            new_segments[new_segments == old_label] = new_label
            
        return new_segments
        
    def _refine_with_watershed(self, image, segments):
        """
        Refine segment boundaries using watershed algorithm
        
        Args:
            image: Original image
            segments: Initial segmentation
            
        Returns:
            Refined segments
        """
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply gradient to find boundaries
        gradient = cv2.morphologyEx(
            gray, cv2.MORPH_GRADIENT, 
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        )
        
        # Prepare markers from segments
        markers = segments + 1  # Add 1 to avoid 0 which watershed treats specially
        
        # Apply watershed
        cv2.watershed(image, markers)
        
        # Adjust labels back
        markers = markers - 1
        markers[markers == -2] = -1  # Fix watershed boundary marker
        
        # Clean up boundaries
        cleaned = np.zeros_like(segments)
        for label in np.unique(segments):
            if label >= 0:  # Skip boundary label (-1)
                mask = (markers == label)
                cleaned[mask] = label
                
        return cleaned
        
    def _calculate_segment_stats(self, segments):
        """
        Calculate statistics about each segment
        
        Args:
            segments: Segmented image with labels
            
        Returns:
            Dictionary with segment statistics
        """
        unique_labels = np.unique(segments)
        unique_labels = unique_labels[unique_labels >= 0]  # Skip boundary label (-1)
        
        stats = {}
        for label in unique_labels:
            # Create mask for this segment
            mask = (segments == label)
            
            # Calculate area
            area = np.sum(mask)
            
            # Find contours
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Calculate perimeter (contour length)
            perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
            
            # Calculate centroid
            m = cv2.moments(mask.astype(np.uint8))
            if m["m00"] > 0:
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
            else:
                cy, cx = np.where(mask)
                cy = np.mean(cy) if len(cy) > 0 else 0
                cx = np.mean(cx) if len(cx) > 0 else 0
                
            # Collect statistics
            stats[int(label)] = {
                "area": int(area),
                "perimeter": float(perimeter),
                "centroid": (int(cx), int(cy)),
                "complexity": float(perimeter**2 / (4 * np.pi * area)) if area > 0 else 0
            }
            
        return stats
        
    def _extract_palette(self, image, segments, num_colors):
        """
        Extract optimized color palette from segmented image
        
        Args:
            image: Original image
            segments: Segmented image
            num_colors: Target number of colors
            
        Returns:
            Array of RGB colors
        """
        # Get unique segment labels
        unique_labels = np.unique(segments)
        unique_labels = unique_labels[unique_labels >= 0]  # Skip boundary label (-1)
        
        # Create palette
        palette = []
        
        for label in unique_labels:
            # Create mask for this segment
            mask = (segments == label)
            
            # Extract pixels for this segment
            segment_pixels = image[mask]
            
            # If segment has pixels, calculate average color
            if len(segment_pixels) > 0:
                # Calculate median color (more robust than mean)
                avg_color = np.median(segment_pixels, axis=0).astype(np.uint8)
                palette.append(avg_color)
        
        # If we have more segments than requested colors, merge similar colors
        if len(palette) > num_colors:
            # Convert to numpy array
            palette = np.array(palette)
            
            # Apply k-means to find representative colors
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
            _, labels, centers = cv2.kmeans(
                np.float32(palette), 
                num_colors, 
                None, 
                criteria, 
                10, 
                cv2.KMEANS_RANDOM_CENTERS
            )
            
            # Use cluster centers as final palette
            palette = centers.astype(np.uint8)
        else:
            palette = np.array(palette)
            
        return palette
        
    def _enhance_edges(self, segments, edge_strength, edge_width, edge_style, feature_map=None):
        """
        Enhance and refine segment edges
        
        Args:
            segments: Segmented image
            edge_strength: Edge strength factor
            edge_width: Edge width in pixels
            edge_style: Edge style name
            feature_map: Optional feature importance map
            
        Returns:
            Edge mask
        """
        h, w = segments.shape
        edges = np.zeros((h, w), dtype=np.uint8)
        
        # Create edges by finding boundaries between different segments
        for y in range(1, h):
            for x in range(1, w):
                current = segments[y, x]
                above = segments[y-1, x]
                left = segments[y, x-1]
                
                if current != above or current != left:
                    edges[y, x] = 255
        
        # Apply edge style
        if edge_style == "soft":
            # Softer edges
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            edges = cv2.GaussianBlur(edges, (5, 5), 1)
            
        elif edge_style == "bold":
            # Bolder edges
            kernel = np.ones((edge_width+2, edge_width+2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
        elif edge_style == "thin":
            # Thinner, more precise edges
            kernel = np.ones((max(1, edge_width-1), max(1, edge_width-1)), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
            
        elif edge_style == "hierarchical" and feature_map is not None:
            # Create hierarchical edges based on feature importance
            # Basic edges
            base_kernel = np.ones((edge_width, edge_width), np.uint8)
            base_edges = cv2.dilate(edges, base_kernel, iterations=1)
            
            # Important edges (near important features)
            important_edges = np.zeros_like(edges)
            
            # Find areas with high importance
            if feature_map is not None:
                # Normalize feature map
                normalized_map = cv2.normalize(feature_map, None, 0, 1, cv2.NORM_MINMAX)
                
                # Threshold to find important areas
                _, important_areas = cv2.threshold(
                    (normalized_map * 255).astype(np.uint8), 
                    128, 255, 
                    cv2.THRESH_BINARY
                )
                
                # Dilate important areas to extend influence
                important_areas = cv2.dilate(
                    important_areas, 
                    np.ones((10, 10), np.uint8)
                )
                
                # Create stronger edges near important areas
                important_kernel = np.ones((edge_width+2, edge_width+2), np.uint8)
                important_edges = cv2.dilate(
                    edges & (important_areas > 0), 
                    important_kernel, 
                    iterations=1
                )
                
                # Combine edges
                edges = cv2.addWeighted(base_edges, 0.7, important_edges, 0.3, 0)
            else:
                edges = base_edges
        else:
            # Standard edges
            kernel = np.ones((edge_width, edge_width), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Apply edge strength
        if edge_strength != 1.0:
            # Adjust edge intensity
            edges = cv2.multiply(edges, edge_strength)
            
        return edges
        
    def _color_segments(self, segments, palette):
        """
        Create colored segmentation for visualization
        
        Args:
            segments: Segmented image
            palette: Color palette
            
        Returns:
            Colored segmentation (RGB)
        """
        h, w = segments.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Get unique segment labels
        unique_labels = np.unique(segments)
        unique_labels = unique_labels[unique_labels >= 0]  # Skip boundary label (-1)
        
        # Apply colors to segments
        for i, label in enumerate(unique_labels):
            if i < len(palette):
                colored[segments == label] = palette[i]
            else:
                # If we somehow have more segments than palette colors
                colored[segments == label] = [200, 200, 200]  # Default gray
                
        return colored
        
    def _apply_template_style(self, colored_segments, edges, segments, palette, style):
        """
        Apply template style to create final template
        
        Args:
            colored_segments: Colored segmentation
            edges: Edge mask
            segments: Segmented image
            palette: Color palette
            style: Template style name
            
        Returns:
            Final template image (RGB)
        """
        h, w = segments.shape
        template = np.copy(colored_segments)
        
        if style == TemplateStyle.CLASSIC.value:
            # Classic style: solid colors with black edges
            template[edges > 0] = [0, 0, 0]  # Black edges
            
        elif style == TemplateStyle.MINIMAL.value:
            # Minimal style: white background with thin black edges
            template = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background
            template[edges > 0] = [0, 0, 0]  # Black edges
            
        elif style == TemplateStyle.DETAILED.value:
            # Detailed style: slightly desaturated colors with strong black edges
            # Desaturate colors
            hsv = cv2.cvtColor(template, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = hsv[:, :, 1] * 0.7  # Reduce saturation
            template = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            # Apply edges
            template[edges > 0] = [0, 0, 0]  # Black edges
            
        elif style == TemplateStyle.ARTISTIC.value:
            # Artistic style: watercolor-like effect
            # Apply slight blur to colors
            template = cv2.GaussianBlur(template, (5, 5), 0)
            # Convert edges to grayscale gradient
            edge_gradient = cv2.GaussianBlur(edges, (5, 5), 2)
            # Apply softened edges
            for c in range(3):
                template[:, :, c] = np.where(
                    edge_gradient > 20,
                    np.clip(template[:, :, c] * (1 - edge_gradient/255 * 0.8), 0, 255),
                    template[:, :, c]
                )
                
        elif style == TemplateStyle.SKETCH.value:
            # Sketch style: white background with sketch-like edges
            template = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background
            # Create sketch-like edges
            edge_gradient = cv2.GaussianBlur(edges, (3, 3), 0.5)
            # Apply randomness to edges for sketch effect
            noise = np.random.normal(0, 15, edge_gradient.shape).astype(np.float32)
            edge_gradient = np.clip(edge_gradient + noise, 0, 255).astype(np.uint8)
            # Apply edges with varying darkness
            for c in range(3):
                template[:, :, c] = np.where(
                    edge_gradient > 30,
                    np.clip(255 - edge_gradient * 0.8, 0, 255),
                    template[:, :, c]
                )
                
        elif style == TemplateStyle.BOLD.value:
            # Bold style: vibrant colors with thick black edges
            # Enhance color saturation
            hsv = cv2.cvtColor(template, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)  # Increase saturation
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)  # Increase value
            template = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            # Apply thick edges
            dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
            template[dilated_edges > 0] = [0, 0, 0]  # Black edges
            
        elif style == TemplateStyle.CLEAN.value:
            # Clean style: soft colors with thin gray edges
            # Soften colors
            hsv = cv2.cvtColor(template, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = hsv[:, :, 1] * 0.8  # Reduce saturation
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)  # Slight brightness boost
            template = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            # Apply thin gray edges
            template[edges > 0] = [80, 80, 80]  # Dark gray edges
            
        else:
            # Default style (classic)
            template[edges > 0] = [0, 0, 0]  # Black edges
            
        return template