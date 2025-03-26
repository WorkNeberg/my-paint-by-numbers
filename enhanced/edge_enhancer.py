import cv2
import numpy as np
import logging
import time

logger = logging.getLogger('pbn-app.edge_enhancer')

class EdgeStyle:
    STANDARD = "standard"
    SOFT = "soft"
    BOLD = "bold"
    THIN = "thin"
    HIERARCHICAL = "hierarchical"

class EdgeEnhancer:
    """
    Enhances and refines edges for paint-by-numbers templates
    with support for different styles and feature awareness.
    """
    
    def __init__(self):
        """Initialize the edge enhancer"""
        logger.info("EdgeEnhancer initialized")
        
    def enhance_edges(self, segments, settings, feature_map=None):
        """
        Enhance edges based on settings
        
        Args:
            segments: Segmented image with region labels
            settings: Dictionary with edge enhancement settings
            feature_map: Optional feature importance map
            
        Returns:
            Edge mask and edge metadata
        """
        start_time = time.time()
        logger.info("Starting edge enhancement")
        
        # Extract edge settings
        edge_strength = settings.get('edge_strength', 1.0)
        edge_width = settings.get('edge_width', 2)
        edge_style = settings.get('edge_style', EdgeStyle.STANDARD)
        
        # Step 1: Extract raw edges
        raw_edges = self._extract_edges(segments)
        
        # Step 2: Apply smoothing
        smooth_edges = self._smooth_edges(raw_edges, edge_width)
        
        # Step 3: Apply style-specific enhancements
        styled_edges = self._apply_edge_style(smooth_edges, edge_style, edge_width, feature_map)
        
        # Step 4: Apply strength adjustment
        final_edges = self._adjust_edge_strength(styled_edges, edge_strength)
        
        # Calculate timing
        elapsed = time.time() - start_time
        logger.info(f"Edge enhancement completed in {elapsed:.2f}s")
        
        return final_edges
        
    def _extract_edges(self, segments):
        """
        Extract edges between different segment regions
        
        Args:
            segments: Segmented image
            
        Returns:
            Binary edge mask
        """
        h, w = segments.shape
        edges = np.zeros((h, w), dtype=np.uint8)
        
        # Find boundaries between different segments
        for y in range(1, h):
            for x in range(w-1):
                # Check horizontal neighbors
                if segments[y, x] != segments[y, x+1]:
                    edges[y, x] = 255
                    edges[y, x+1] = 255
                    
                # Check vertical neighbors
                if y < h-1 and segments[y, x] != segments[y+1, x]:
                    edges[y, x] = 255
                    edges[y+1, x] = 255
                    
                # Check diagonal neighbors
                if y < h-1 and x < w-1 and segments[y, x] != segments[y+1, x+1]:
                    edges[y, x] = 255
                    edges[y+1, x+1] = 255
        
        return edges
        
    def _smooth_edges(self, edges, edge_width):
        """
        Smooth edges to make them less jagged
        
        Args:
            edges: Binary edge mask
            edge_width: Target edge width
            
        Returns:
            Smoothed edge mask
        """
        # Skip smoothing for very thin edges
        if edge_width <= 1:
            return edges
            
        # Apply slight blur to soften jagged edges
        blurred = cv2.GaussianBlur(edges, (3, 3), 0.8)
        
        # Threshold to restore binary mask
        _, smoothed = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
        
        return smoothed
        
    def _apply_edge_style(self, edges, style, edge_width, feature_map=None):
        """
        Apply a specific edge style
        
        Args:
            edges: Binary edge mask
            style: Edge style name
            edge_width: Edge width in pixels
            feature_map: Optional feature importance map
            
        Returns:
            Styled edge mask
        """
        if style == EdgeStyle.STANDARD:
            # Standard edges: consistent width
            kernel = np.ones((edge_width, edge_width), np.uint8)
            styled = cv2.dilate(edges, kernel, iterations=1)
            
        elif style == EdgeStyle.SOFT:
            # Soft edges: blurred
            styled = cv2.GaussianBlur(edges, (edge_width*2+1, edge_width*2+1), edge_width/2)
            # Normalize to keep contrast
            styled = cv2.normalize(styled, None, 0, 255, cv2.NORM_MINMAX)
            
        elif style == EdgeStyle.BOLD:
            # Bold edges: thicker with high contrast
            kernel = np.ones((edge_width+2, edge_width+2), np.uint8)
            styled = cv2.dilate(edges, kernel, iterations=1)
            
        elif style == EdgeStyle.THIN:
            # Thin edges: precise and thin
            kernel = np.ones((max(1, edge_width-1), max(1, edge_width-1)), np.uint8)
            styled = cv2.dilate(edges, kernel, iterations=1)
            styled = cv2.erode(styled, np.ones((3, 3), np.uint8), iterations=1)
            
        elif style == EdgeStyle.HIERARCHICAL and feature_map is not None:
            # Hierarchical edges: different widths based on feature importance
            styled = self._create_hierarchical_edges(edges, edge_width, feature_map)
            
        else:
            # Default to standard if style not recognized
            kernel = np.ones((edge_width, edge_width), np.uint8)
            styled = cv2.dilate(edges, kernel, iterations=1)
            
        return styled
        
    def _create_hierarchical_edges(self, edges, base_width, feature_map):
        """
        Create hierarchical edges with varying widths based on feature importance
        
        Args:
            edges: Binary edge mask
            base_width: Base edge width
            feature_map: Feature importance map
            
        Returns:
            Hierarchical edge mask
        """
        # Normalize feature map to 0-1 range
        if feature_map is not None:
            feature_norm = cv2.normalize(feature_map, None, 0, 1, cv2.NORM_MINMAX)
        else:
            # If no feature map, return standard edges
            kernel = np.ones((base_width, base_width), np.uint8)
            return cv2.dilate(edges, kernel, iterations=1)
            
        # Create three levels of edges
        # Level 1: Thin edges for low-importance areas (base_width-1)
        # Level 2: Standard edges for medium-importance areas (base_width)
        # Level 3: Thick edges for high-importance areas (base_width+2)
        
        # Create importance thresholds
        low_thresh = 0.3
        high_thresh = 0.7
        
        # Create dilated edges for each level
        kernel_thin = np.ones((max(1, base_width-1), max(1, base_width-1)), np.uint8)
        kernel_std = np.ones((base_width, base_width), np.uint8)
        kernel_thick = np.ones((base_width+2, base_width+2), np.uint8)
        
        thin_edges = cv2.dilate(edges, kernel_thin, iterations=1)
        std_edges = cv2.dilate(edges, kernel_std, iterations=1)
        thick_edges = cv2.dilate(edges, kernel_thick, iterations=1)
        
        # Create masks for each importance level
        low_mask = feature_norm < low_thresh
        mid_mask = (feature_norm >= low_thresh) & (feature_norm < high_thresh)
        high_mask = feature_norm >= high_thresh
        
        # Dilate masks to affect nearby edges
        kernel_dilate = np.ones((15, 15), np.uint8)
        low_mask_dilated = cv2.dilate(low_mask.astype(np.uint8), kernel_dilate).astype(bool)
        mid_mask_dilated = cv2.dilate(mid_mask.astype(np.uint8), kernel_dilate).astype(bool)
        high_mask_dilated = cv2.dilate(high_mask.astype(np.uint8), kernel_dilate).astype(bool)
        
        # Combine edges according to masks
        hierarchical = np.zeros_like(edges)
        hierarchical[low_mask_dilated] = thin_edges[low_mask_dilated]
        hierarchical[mid_mask_dilated] = std_edges[mid_mask_dilated]
        hierarchical[high_mask_dilated] = thick_edges[high_mask_dilated]
        
        # Fill any remaining areas with standard edges
        hierarchical[hierarchical == 0] = std_edges[hierarchical == 0]
        
        return hierarchical
        
    def _adjust_edge_strength(self, edges, strength):
        """
        Adjust edge strength/opacity
        
        Args:
            edges: Edge mask
            strength: Edge strength factor
            
        Returns:
            Adjusted edge mask
        """
        if strength == 1.0:
            return edges
            
        # Apply strength as a multiplier
        adjusted = cv2.multiply(edges.astype(np.float32), strength)
        
        # Normalize back to 0-255 range
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        return adjusted