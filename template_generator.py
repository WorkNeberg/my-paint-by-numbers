import cv2
import numpy as np
from scipy import ndimage

class TemplateGenerator:
    """Generates paint-by-numbers templates"""
    
    def generate_template(self, image, segments, color_map, palette):
        """Generate paint-by-numbers template"""
        # Create segment outlines
        outlines = self.create_outlines(segments)
        
        # Calculate optimal positions for numbers
        number_positions = self.calculate_number_positions(segments)
        
        # Place numbers on template
        numbered_template = self.place_numbers(outlines.copy(), segments, number_positions, color_map)
        
        # Create colored result preview
        result_preview = self.create_result_preview(image, segments, color_map, palette)
        
        # Create color reference chart
        color_reference = self.create_color_reference(palette, color_map, segments)
        
        return {
            "outline_template": outlines,
            "numbered_template": numbered_template,
            "result_preview": result_preview,
            "color_reference": color_reference
        }
    
    def create_outlines(self, segments):
        """Create clean outlines for segments"""
        # Create an empty image for outlines
        outlines = np.ones(segments.shape, dtype=np.uint8) * 255
        
        # Find boundaries between different segments
        h, w = segments.shape
        for y in range(1, h):
            for x in range(1, w):
                # Check if pixel belongs to a different segment than any neighbor
                if (segments[y, x] != segments[y, x-1] or 
                    segments[y, x] != segments[y-1, x]):
                    outlines[y, x] = 0
        
        # Ensure clean, connected outlines
        outlines = cv2.dilate(outlines, np.ones((2, 2), np.uint8), iterations=1)
        
        # Convert to 3-channel for colored template
        outlines_3ch = cv2.cvtColor(outlines, cv2.COLOR_GRAY2BGR)
        
        return outlines_3ch
    
    def calculate_number_positions(self, segments):
        """Calculate optimal positions for placing numbers in each segment"""
        # Find the center of each segment using distance transform
        number_positions = {}
        
        for segment_id in np.unique(segments):
            # Create mask for this segment
            mask = (segments == segment_id).astype(np.uint8)
            
            # Calculate distance transform
            dist_transform = ndimage.distance_transform_edt(mask)
            
            # Find the point with maximum distance from boundaries
            max_dist_idx = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)
            
            # Store position and maximum distance (for sizing numbers)
            number_positions[segment_id] = {
                'position': max_dist_idx,
                'distance': dist_transform[max_dist_idx]
            }
            
        return number_positions
    
    def place_numbers(self, template, segments, number_positions, color_map):
        """Place numbers on the template"""
        # Font settings
        min_font_scale = 0.3
        max_font_scale = 0.8
        
        for segment_id, position_data in number_positions.items():
            # Skip very small segments
            if position_data['distance'] < 3:
                continue
                
            # Get color number from color map
            color_number = color_map.get(segment_id, 0) + 1  # +1 for 1-based numbering
            
            # Calculate appropriate font scale based on segment size
            font_scale = min(max_font_scale, 
                          max(min_font_scale, position_data['distance'] / 30))
            
            # Calculate text size
            text = str(color_number)
            thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Calculate position to center text
            y, x = position_data['position']
            text_x = int(x - text_size[0] / 2)
            text_y = int(y + text_size[1] / 2)
            
            # Ensure text is within image boundaries
            h, w = segments.shape
            text_x = max(0, min(text_x, w - text_size[0] - 1))
            text_y = max(text_size[1], min(text_y, h - 1))
            
            # Place number on template
            cv2.putText(template, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
            
        return template
    
    def create_result_preview(self, image, segments, color_map, palette):
        """Create a preview of the finished painting"""
        # Create an empty image
        h, w = segments.shape
        preview = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Fill each segment with its assigned color
        for segment_id in np.unique(segments):
            if segment_id in color_map:
                mask = segments == segment_id
                color_idx = color_map[segment_id]
                preview[mask] = palette[color_idx]
                