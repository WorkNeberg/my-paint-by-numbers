import cv2
import numpy as np
import logging
import time
from enum import Enum

logger = logging.getLogger('pbn-app.template_styler')

class TemplateStyle(Enum):
    CLASSIC = "classic"
    MINIMAL = "minimal"
    DETAILED = "detailed"
    ARTISTIC = "artistic"
    SKETCH = "sketch"
    BOLD = "bold"
    CLEAN = "clean"

class TemplateStyler:
    """
    Handles the styling and appearance of paint-by-numbers templates
    """
    
    def __init__(self):
        """Initialize the template styler"""
        logger.info("TemplateStyler initialized")
        
    def apply_style(self, colored_segments, edges, segments, palette, settings):
        """
        Apply a style to create the final template
        
        Args:
            colored_segments: Segments with colors applied
            edges: Edge mask
            segments: Original segment labels
            palette: Color palette
            settings: Style settings
            
        Returns:
            Styled template image
        """
        start_time = time.time()
        logger.info("Starting template styling")
        
        # Extract style settings
        style_name = settings.get('template_style', 'classic')
        style = TemplateStyle(style_name) if style_name in [e.value for e in TemplateStyle] else TemplateStyle.CLASSIC
        
        # Create appropriate style
        if style == TemplateStyle.CLASSIC:
            template = self._create_classic_style(colored_segments, edges)
        elif style == TemplateStyle.MINIMAL:
            template = self._create_minimal_style(segments, edges, palette)
        elif style == TemplateStyle.DETAILED:
            template = self._create_detailed_style(colored_segments, edges, segments)
        elif style == TemplateStyle.ARTISTIC:
            template = self._create_artistic_style(colored_segments, edges, segments)
        elif style == TemplateStyle.SKETCH:
            template = self._create_sketch_style(segments, edges, palette)
        elif style == TemplateStyle.BOLD:
            template = self._create_bold_style(colored_segments, edges)
        elif style == TemplateStyle.CLEAN:
            template = self._create_clean_style(colored_segments, edges)
        else:
            # Default to classic
            template = self._create_classic_style(colored_segments, edges)
            
        # Calculate timing
        elapsed = time.time() - start_time
        logger.info(f"Template styling completed in {elapsed:.2f}s")
        
        return template
        
    def _create_classic_style(self, colored_segments, edges):
        """
        Create classic template style: solid colors with black edges
        
        Args:
            colored_segments: Colored segments
            edges: Edge mask
            
        Returns:
            Classic style template
        """
        template = colored_segments.copy()
        
        # Apply black edges
        template[edges > 0] = [0, 0, 0]
        
        return template
        
    def _create_minimal_style(self, segments, edges, palette):
        """
        Create minimal template style: white background with thin black edges
        
        Args:
            segments: Segmented image
            edges: Edge mask
            palette: Color palette
            
        Returns:
            Minimal style template
        """
        h, w = segments.shape
        
        # Create white background
        template = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Apply black edges
        template[edges > 0] = [0, 0, 0]
        
        return template
        
    def _create_detailed_style(self, colored_segments, edges, segments):
        """
        Create detailed template style: slightly desaturated colors with strong edges
        
        Args:
            colored_segments: Colored segments
            edges: Edge mask
            segments: Segmented image
            
        Returns:
            Detailed style template
        """
        # Desaturate colors
        template = colored_segments.copy()
        hsv = cv2.cvtColor(template, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * 0.7  # Reduce saturation
        template = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Apply black edges
        template[edges > 0] = [0, 0, 0]
        
        return template
        
    def _create_artistic_style(self, colored_segments, edges, segments):
        """
        Create artistic template style: watercolor-like effect
        
        Args:
            colored_segments: Colored segments
            edges: Edge mask
            segments: Segmented image
            
        Returns:
            Artistic style template
        """
        # Start with base colors
        template = colored_segments.copy()
        
        # Apply slight blur to colors for watercolor effect
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
        
        return template
        
    def _create_sketch_style(self, segments, edges, palette):
        """
        Create sketch template style: white background with sketch-like edges
        
        Args:
            segments: Segmented image
            edges: Edge mask
            palette: Color palette
            
        Returns:
            Sketch style template
        """
        h, w = segments.shape
        
        # Create white background
        template = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Create sketch-like edges
        edge_gradient = cv2.GaussianBlur(edges, (3, 3), 0.5)
        
        # Add randomness to edges for sketch effect
        np.random.seed(42)  # For reproducibility
        noise = np.random.normal(0, 15, edge_gradient.shape).astype(np.float32)
        edge_gradient = np.clip(edge_gradient + noise, 0, 255).astype(np.uint8)
        
        # Apply edges with varying darkness
        for c in range(3):
            template[:, :, c] = np.where(
                edge_gradient > 30,
                np.clip(255 - edge_gradient * 0.8, 0, 255),
                template[:, :, c]
            )
        
        return template
        
    def _create_bold_style(self, colored_segments, edges):
        """
        Create bold template style: vibrant colors with thick black edges
        
        Args:
            colored_segments: Colored segments
            edges: Edge mask
            
        Returns:
            Bold style template
        """
        # Enhance color saturation
        template = colored_segments.copy()
        hsv = cv2.cvtColor(template, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)  # Increase saturation
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)  # Increase value
        template = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Apply thick edges
        dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        template[dilated_edges > 0] = [0, 0, 0]  # Black edges
        
        return template
        
    def _create_clean_style(self, colored_segments, edges):
        """
        Create clean template style: soft colors with thin gray edges
        
        Args:
            colored_segments: Colored segments
            edges: Edge mask
            
        Returns:
            Clean style template
        """
        # Soften colors
        template = colored_segments.copy()
        hsv = cv2.cvtColor(template, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * 0.8  # Reduce saturation
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)  # Slight brightness boost
        template = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Apply thin gray edges
        template[edges > 0] = [80, 80, 80]  # Dark gray edges
        
        return template