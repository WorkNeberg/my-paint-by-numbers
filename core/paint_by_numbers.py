import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from core.image_processor import ImageProcessor
from core.number_placement import NumberPlacer
from core.image_analyzer import ImageAnalyzer
from core.performance_optimizer import PerformanceOptimizer

class PaintByNumbersGenerator:
    def __init__(self, output_dir=None):
        # Initialize components
        self.processor = ImageProcessor()
        self.number_placer = NumberPlacer()
        self.analyzer = ImageAnalyzer()
        self.latest_analysis = None  # Store the most recent analysis
        
        # Color dictionary for naming
        self.color_dict = {
            'Red': [255, 0, 0],
            'Green': [0, 128, 0],
            'Blue': [0, 0, 255],
            'Yellow': [255, 255, 0],
            'Cyan': [0, 255, 255],
            'Magenta': [255, 0, 255],
            'Black': [0, 0, 0],
            'White': [255, 255, 255],
            'Gray': [128, 128, 128],
            'Orange': [255, 165, 0],
            'Purple': [128, 0, 128],
            'Brown': [165, 42, 42],
            'Pink': [255, 192, 203],
            'Navy': [0, 0, 128],
            'Teal': [0, 128, 128],
            'Olive': [128, 128, 0],
            'Maroon': [128, 0, 0],
            'Lime': [0, 255, 0],
            'Gold': [255, 215, 0],
            'Silver': [192, 192, 192],
            'Sky Blue': [135, 206, 235],
            'Tan': [210, 180, 140],
            'Lavender': [230, 230, 250],
            'Beige': [245, 245, 220],
        }
        self.performance_optimizer = PerformanceOptimizer()
    
    def get_color_name(self, rgb_value):
        """Find the nearest named color for an RGB value"""
        min_dist = float('inf')
        closest_color = "Unknown"
        
        for color_name, color_rgb in self.color_dict.items():
            dist = sum((a - b) ** 2 for a, b in zip(rgb_value, color_rgb))
            if dist < min_dist:
                min_dist = dist
                closest_color = color_name
        
        return closest_color
    
    def preprocess_image(self, image_path, max_dim=1200):
        """Preprocess image for segmentation"""
        # Check if file exists
        if not os.path.exists(image_path):
            raise ValueError(f"Image file does not exist: {image_path}")
            
        # Print debug info
        print(f"Reading image from: {image_path}")
        print(f"File size: {os.path.getsize(image_path)} bytes")
        
        # Read image with explicit error handling
        image = cv2.imread(image_path)
        
        if image is None:
            # Try reading with different flag
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}. The file may be corrupted or in an unsupported format.")
        
        # Check image dimensions
        print(f"Image shape after reading: {image.shape}")
        
        # Convert color space
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            raise ValueError(f"Error converting image color space: {str(e)}")
        
        # Resize if needed
        height, width = image.shape[:2]
        if height > max_dim or width > max_dim:
            scale = max_dim / max(height, width)
            new_width, new_height = int(width * scale), int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        return image
        
    def enrich_region_data(self, region_data, centers):
        """Add color names and optimize color representation"""
        enriched_data = []
        
        for i, region in enumerate(region_data):
            color = region['color'].astype(int)
            color_name = self.get_color_name(color)
            
            # Add color information
            enriched = {
                'id': i + 1,  # Reassign IDs starting from 1
                'center': region['center'],
                'area': region['area'],
                'color': color,
                'color_name': color_name,
                'color_hex': '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2]),
                'compactness': region.get('compactness', 0)
            }
            
            enriched_data.append(enriched)
            
        # Sort regions by brightness (light to dark)
        enriched_data.sort(key=lambda x: np.sum(x['color']), reverse=True)
        
        # Reassign IDs based on sorted order
        for i, region in enumerate(enriched_data):
            region['id'] = i + 1
            
        return enriched_data
    
    def create_template(self, shape, edges, region_data, label_image, style="classic", edge_style="normal"):
        """
        Create a paint-by-numbers template with numbers
        
        Parameters:
        - shape: Shape of the image (height, width)
        - edges: Edge map
        - region_data: Data about each region
        - label_image: Image with region labels
        - style: Template style ('classic', 'minimal', 'detailed')
        - edge_style: Style of edges ('normal', 'soft', 'thin')
        
        Returns:
        - Template image with numbers
        """
        h, w = shape[:2]
        
        # Create template with white background
        template = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Ensure edges are present and visible
        if np.max(edges) == 0:
            print("Warning: No edges detected. Generating edges from label image.")
            # Generate edges from label differences
            horiz_edges = (label_image[:-1, :] != label_image[1:, :])
            vert_edges = (label_image[:, :-1] != label_image[:, 1:])
            
            # Create edge map
            edges = np.zeros((h, w), dtype=np.uint8)
            edges[:-1, :][horiz_edges] = 255
            edges[1:, :][horiz_edges] = 255
            edges[:, :-1][vert_edges] = 255
            edges[:, 1:][vert_edges] = 255
        
        # Determine edge intensity based on style
        if edge_style == "soft":
            edge_intensity = 180  # MUCH lighter gray (was 120)
        elif edge_style == "thin":
            edge_intensity = 160  # MUCH lighter gray (was 50)
        elif edge_style == "hierarchical":
            # Create white background
            template = np.ones((h, w, 3), dtype=np.uint8) * 255
            
            # Pet outline - make it BLACK (not dark gray)
            outline_mask = (edges >= 240)  # Only the strongest edges
            template[outline_mask] = [0, 0, 0]  # Pure black
            
            # Feature edges (eyes, nose) - dark gray
            feature_mask = (edges >= 150) & (edges < 240)
            template[feature_mask] = [60, 60, 60]  # Darker than before
            
            # Internal detail edges - very light (almost invisible)
            detail_mask = (edges > 0) & (edges < 150)
            template[detail_mask] = [220, 220, 220]  # Much lighter
            
            return template
        else:  # "normal" or any other
            edge_intensity = 140  # MUCH lighter gray (was 30)
            
        # Apply edges with proper intensity and ensure they're visible
        for i in range(3):
            template[:,:,i] = np.where(edges > 0, edge_intensity, template[:,:,i])
        
        # Apply different styles
        if style == "minimal":
            # Minimal style has only edges and small numbers
            font_scale = 0.4
            thickness = 1
            number_color = (100, 100, 100)  # Gray numbers
        elif style == "detailed":
            # Detailed style has edges, numbers, and region shading
            font_scale = 0.5
            thickness = 1
            number_color = (0, 0, 0)  # Black numbers
            
            # Add light shading to alternate regions for better visibility
            for i, region in enumerate(region_data):
                mask = (label_image == region['id']).astype(np.uint8)
                if i % 2 == 0:
                    # Apply light gray shading
                    for c in range(3):
                        template[:,:,c] = np.where(mask == 1, 245, template[:,:,c])
        else:  # "classic" or default
            # Classic style has clear edges and normal sized numbers
            font_scale = 0.5
            thickness = 1
            number_color = (0, 0, 0)  # Black numbers
        
        # Sort regions by size, largest first
        region_data.sort(key=lambda x: x['area'], reverse=True)

        # Assign numbers 1-9 to largest regions first
        for i, region in enumerate(region_data):
            if i < 9:
                region['number'] = i + 1
            else:
                # For additional regions, use letters or repeated numbers
                # This keeps numbering simpler
                region['number'] = (i % 9) + 1

        # Make numbers larger and clearer
        font_scale = 0.6  # Increased from default
        font_thickness = 2  # Thicker for visibility

        # Add after sorting regions by size:
        # Don't place numbers on eyes or tiny regions
        min_area_for_number = h * w * 0.003  # Minimum 0.3% of image size

        # Place region numbers
        for region in region_data:
            if 'center' in region and region['area'] > min_area_for_number:
                # Calculate font size based on region area
                area = region['area']
                adaptive_scale = min(max(area / 5000, 0.3), 1.0) * font_scale
                
                # Get region center
                cx, cy = region['center']
                
                # Create the number text
                number = str(region['number'])  # Use 1-based indexing for user-friendliness
                
                # Calculate text size
                (text_width, text_height), _ = cv2.getTextSize(
                    number, cv2.FONT_HERSHEY_SIMPLEX, adaptive_scale, thickness
                )
                
                # Adjust position to center the text
                text_x = cx - text_width // 2
                text_y = cy + text_height // 2
                
                # Ensure text is within image bounds
                text_x = max(0, min(text_x, w - text_width))
                text_y = max(text_height, min(text_y, h))
                
                # Draw the number
                cv2.putText(
                    template, number, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, adaptive_scale, number_color, thickness
                )
        
        # Create a two-level hierarchy:
        
        # 1. Pet outline - very dark
        pet_outline = np.zeros_like(edges)
        # Calculate the outer boundary of the entire pet
        unique_labels = np.unique(label_image)
        if 0 in unique_labels and len(unique_labels) > 1:
            # 0 is typically background
            pet_mask = (label_image > 0).astype(np.uint8)
            # Dilate and subtract to find outline
            dilated = cv2.dilate(pet_mask, np.ones((3,3), np.uint8))
            pet_outline = dilated - pet_mask
            # Make pet outline very dark
            edges[pet_outline > 0] = 255  # Full black for pet outline
        
        # 2. Feature outlines (eyes, nose, etc.) - medium dark
        # Keep original edge detection for features
        
        # 3. Internal details - light gray or removed
        # Make non-feature, non-outline edges lighter
        internal_edges = (edges > 0) & (pet_outline == 0)
        edges[internal_edges] = 100  # Light gray for internal details
        
        return template
    
    def create_color_chart(self, region_data):
        """Create color reference chart image"""
        chart_height = max(600, len(region_data) * 30 + 50)
        color_chart = np.ones((chart_height, 400, 3), dtype=np.uint8) * 255
        
        # Add title to chart
        cv2.putText(color_chart, "Color Reference Chart", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        # Add colors to chart
        for i, region in enumerate(region_data):
            color = region['color'].astype(int)
            region_id = region['id']
            color_name = region['color_name']
            color_hex = region['color_hex']
            y_pos = i * 30 + 60
            
            # Draw color square
            cv2.rectangle(color_chart, (20, y_pos - 15), (50, y_pos + 5), color.tolist(), -1)
            cv2.rectangle(color_chart, (20, y_pos - 15), (50, y_pos + 5), (0, 0, 0), 1)
            
            # Add color info
            cv2.putText(color_chart, f"{region_id}: {color_name}", (60, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(color_chart, f"{color_hex} {tuple(color)}", (60, y_pos + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return color_chart
    
    def create_paintability_badge(self, paintability):
        """Create a paintability score badge"""
        badge = np.ones((100, 200, 3), dtype=np.uint8) * 255
        
        # Choose color based on score
        if paintability >= 80:
            color = (0, 180, 0)  # Green
            level = "Excellent"
        elif paintability >= 60:
            color = (0, 180, 180)  # Yellow-green
            level = "Good"
        elif paintability >= 40:
            color = (0, 120, 255)  # Orange
            level = "Moderate"
        else:
            color = (0, 0, 255)  # Red
            level = "Challenging"
        
        # Draw circular gauge
        cv2.ellipse(badge, (50, 50), (40, 40), 0, 0, 360, (220, 220, 220), -1)
        cv2.ellipse(badge, (50, 50), (40, 40), 0, 0, int(paintability * 3.6), color, -1)
        cv2.ellipse(badge, (50, 50), (40, 40), 0, 0, 360, (0, 0, 0), 2)
        cv2.circle(badge, (50, 50), 30, (255, 255, 255), -1)
        
        # Add score text
        cv2.putText(badge, str(paintability), (35, 58), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        # Add label
        cv2.putText(badge, "Paintability Score", (100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(badge, level, (100, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return badge
    
    def process_image(self, image_path, num_colors=None, output_dir='output',
                     simplification_level=None, edge_strength=None, edge_width=None,
                     template_style='classic', enhance_dark_areas=None, dark_threshold=None,
                     auto_detect=True, edge_style='normal', merge_regions_level='normal',
                     image_type='general'):
        """
        Process image to create paint-by-numbers template with smart analysis
        
        Parameters:
        - image_path: Path to the image file
        - num_colors: Number of colors to use (will override auto-detection)
        - output_dir: Directory to save output files
        - simplification_level: Level of detail ('low', 'medium', 'high')
        - edge_strength: Strength of edge lines (0.5-1.5)
        - edge_width: Width of edge lines in pixels
        - template_style: Style of numbers ('classic', 'minimal', 'detailed')
        - enhance_dark_areas: Whether to enhance dark areas
        - dark_threshold: Threshold for dark area enhancement
        - auto_detect: Whether to use smart analysis
        - edge_style: Style of edges ('normal', 'soft', 'thin')
        - merge_regions_level: Level of region merging ('none', 'low', 'normal', 'aggressive')
        - image_type: Type of image ('general', 'portrait', 'pet', 'cartoon', 'landscape')
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filenames
        basename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(basename)[0]
        timestamp = int(time.time())
        
        # Define output paths
        template_path = os.path.join(output_dir, f"{name_without_ext}_{timestamp}_template.png")
        minimal_path = os.path.join(output_dir, f"{name_without_ext}_{timestamp}_minimal.png")
        detailed_path = os.path.join(output_dir, f"{name_without_ext}_{timestamp}_detailed.png")
        chart_path = os.path.join(output_dir, f"{name_without_ext}_{timestamp}_chart.png")
        preview_path = os.path.join(output_dir, f"{name_without_ext}_{timestamp}_preview.png")
        processed_path = os.path.join(output_dir, f"{name_without_ext}_{timestamp}_processed.png")
        
        print(f"Processing image: {image_path}")
        
        try:
            # Step 1: Preprocess the image
            image = self.preprocess_image(image_path)
            
            # Step 2: Analyze the image to determine optimal parameters
            if auto_detect:
                print("Analyzing image to determine optimal parameters...")
                analysis = self.analyzer.analyze(image)
                self.latest_analysis = analysis  # Store for later reference
                
                # Use the suggested parameters
                params = analysis['parameters']
                
                # Print the analysis results
                print(f"Image Analysis Results:")
                print(f"  - Suggested image type: {analysis.get('suggested_type', 'unknown')}")
                print(f"  - Dark area percentage: {analysis.get('dark_percentage', 0):.1f}%")
                print(f"  - Edge complexity: {analysis.get('edge_complexity', 'medium')}")
                print(f"  - Color diversity: {analysis.get('color_diversity', 'medium')}")
                print(f"  - Suggested parameters: {params}")
                
                # Use detected image type
                if 'suggested_type' in analysis:
                    image_type = analysis['suggested_type']
                
                # Allow override of auto-detected parameters if specified
                if num_colors is not None:
                    params['num_colors'] = num_colors
                if simplification_level is not None:
                    params['simplification_level'] = simplification_level
                if edge_strength is not None:
                    params['edge_strength'] = edge_strength
                if edge_width is not None:
                    params['edge_width'] = edge_width
                if enhance_dark_areas is not None:
                    params['enhance_dark_areas'] = enhance_dark_areas
                if dark_threshold is not None:
                    params['dark_threshold'] = dark_threshold
                
                # Add new parameters with defaults if not present
                if 'edge_style' not in params:
                    params['edge_style'] = edge_style
                if 'merge_regions_level' not in params:
                    params['merge_regions_level'] = merge_regions_level
                if 'image_type' not in params:
                    params['image_type'] = image_type
            else:
                # Use provided parameters (or defaults)
                params = {
                    'num_colors': num_colors if num_colors is not None else 15,
                    'simplification_level': simplification_level if simplification_level is not None else 'medium',
                    'edge_strength': edge_strength if edge_strength is not None else 1.0,
                    'edge_width': edge_width if edge_width is not None else 1,
                    'enhance_dark_areas': enhance_dark_areas if enhance_dark_areas is not None else False,
                    'dark_threshold': dark_threshold if dark_threshold is not None else 50,
                    'edge_style': edge_style,
                    'merge_regions_level': merge_regions_level,
                    'image_type': image_type
                }
            
            # Step 3: Process the image with optimized parameters
            print(f"Processing with parameters: {params}")
            
            # Make sure we handle the case where processor doesn't accept all parameters
            try:
                # Try with all parameters first
                vectorized, label_image, edges, centers, raw_region_data, paintability = self.processor.process_image(
                    image, 
                    params['num_colors'],
                    params['simplification_level'],
                    params['edge_strength'],
                    params['edge_width'],
                    params['enhance_dark_areas'],
                    params['dark_threshold'],
                    params.get('edge_style', 'normal'),
                    params.get('merge_regions_level', 'normal'),
                    params.get('image_type', 'general')
                )
            except TypeError as e:
                print(f"Processor doesn't support all parameters, falling back to basic parameters: {str(e)}")
                # Fall back to original parameter set if method signature doesn't match
                vectorized, label_image, edges, centers, raw_region_data, paintability = self.processor.process_image(
                    image, 
                    params['num_colors'],
                    params['simplification_level'],
                    params['edge_strength'],
                    params['edge_width'],
                    params['enhance_dark_areas'],
                    params['dark_threshold']
                )
            
            # Save the processed version for reference
            plt.imsave(processed_path, vectorized)
            
            # Step 4: Enrich region data with color information
            print("Enriching region data...")
            region_data = self.enrich_region_data(raw_region_data, centers)
            
            # Step 5: Create templates with different styles
            print(f"Creating template with style: {template_style}")
            
            # Use the edge style from parameters
            current_edge_style = params.get('edge_style', 'normal')
            
            classic_template = self.create_template(
                vectorized.shape, edges, region_data, label_image, 
                style="classic", edge_style=current_edge_style
            )
            minimal_template = self.create_template(
                vectorized.shape, edges, region_data, label_image, 
                style="minimal", edge_style=current_edge_style
            )
            detailed_template = self.create_template(
                vectorized.shape, edges, region_data, label_image, 
                style="detailed", edge_style=current_edge_style
            )
            
            # Select the primary template based on requested style
            if template_style == 'minimal':
                template = minimal_template
            elif template_style == 'detailed':
                template = detailed_template
            else:  # 'classic'
                template = classic_template
            
            # Step 6: Create color reference chart
            print("Creating color chart...")
            color_chart = self.create_color_chart(region_data)
            
            # Step 7: Create paintability badge
            print(f"Creating paintability badge (score: {paintability})...")
            badge = self.create_paintability_badge(paintability)
            
            # Save outputs
            print("Saving output files...")
            plt.imsave(template_path, template)
            plt.imsave(minimal_path, minimal_template)
            plt.imsave(detailed_path, detailed_template)
            plt.imsave(chart_path, color_chart)
            
            # Create a comprehensive preview image
            print("Creating preview image...")
            h, w = template.shape[:2]
            badge_h, badge_w = badge.shape[:2]
            
            # Create preview with template, processed image and badge
            preview = np.ones((h, w*2 + badge_w + 20, 3), dtype=np.uint8) * 255
            
            # Add template and vectorized image
            preview[:, :w] = template
            preview[:, w+10:w*2+10] = vectorized
            
            # Add the badge at the right side
            y_offset = max(0, (h - badge_h) // 2)
            badge_h_actual = min(badge_h, h - y_offset)
            
            if badge_h_actual > 0 and badge_w > 0:
                preview[y_offset:y_offset+badge_h_actual, w*2+20:w*2+20+badge_w] = badge[:badge_h_actual]
            
            plt.imsave(preview_path, preview)
            print("Processing complete!")
            
            return {
                'template': template_path,
                'minimal': minimal_path,
                'detailed': detailed_path,
                'chart': chart_path,
                'preview': preview_path,
                'processed': processed_path,
                'region_data': region_data,
                'paintability': paintability
            }
            
        except Exception as e:
            print(f"Error in advanced processing: {str(e)}")
            import traceback
            traceback.print_exc()
            
            print("Falling back to basic processing...")
            
            # Basic processing as fallback
            try:
                # Step 1: Read and preprocess image
                if isinstance(image_path, str):
                    if not os.path.exists(image_path):
                        raise ValueError(f"Image file does not exist: {image_path}")
                        
                    # Read image directly since preprocess may have failed
                    image = cv2.imread(image_path)
                    if image is None:
                        raise ValueError(f"Could not read image: {image_path}")
                    
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    # If image was already loaded and passed
                    image = image_path
                    
                # 1. Reduce noise
                blurred = cv2.bilateralFilter(image, 9, 75, 75)
                
                # 2. Find edges
                gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
                edge_str = 1.0 if edge_strength is None else edge_strength
                edges = cv2.Canny(gray, 50 * edge_str, 150 * edge_str)
                
                # 3. Simple color clustering
                from sklearn.cluster import KMeans
                pixels = blurred.reshape(-1, 3)
                n_colors = 15 if num_colors is None else num_colors
                kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
                labels = kmeans.fit_predict(pixels)
                centers = kmeans.cluster_centers_
                
                # 4. Recreate image with clustered colors
                segmented = centers[labels].reshape(image.shape).astype(np.uint8)
                
                # 5. Create basic template using the label map for number placement
                label_map = labels.reshape(image.shape[:2])
                
                # 6. Create simple region data
                region_data = []
                for i in range(n_colors):
                    color = centers[i].astype(int)
                    color_name = self.get_color_name(color)
                    
                    # Create mask for this color
                    mask = (labels.reshape(image.shape[:2]) == i)
                    area = np.sum(mask)
                    
                    # Find centroid
                    indices = np.where(mask)
                    if len(indices[0]) > 0:
                        cy = int(np.mean(indices[0]))
                        cx = int(np.mean(indices[1]))
                    else:
                        cy, cx = 0, 0
                    
                    region_data.append({
                        'id': i + 1,
                        'center': (cx, cy),
                        'area': int(area),
                        'color': color,
                        'color_name': color_name,
                        'color_hex': '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
                    })
                
                # Create templates with proper edge style
                template = self.create_template(
                    image.shape, edges, region_data, label_map, 
                    style="classic", edge_style=edge_style
                )
                
                minimal_template = self.create_template(
                    image.shape, edges, region_data, label_map,
                    style="minimal", edge_style=edge_style
                )
                
                detailed_template = self.create_template(
                    image.shape, edges, region_data, label_map,
                    style="detailed", edge_style=edge_style
                )
                
                # Save the images
                plt.imsave(template_path, template)
                plt.imsave(processed_path, segmented)
                plt.imsave(minimal_path, minimal_template)
                plt.imsave(detailed_path, detailed_template)
                
                # Create simple color chart
                color_chart = self.create_color_chart(region_data)
                plt.imsave(chart_path, color_chart)
                
                # Create a simple preview (template + segmented)
                h, w = template.shape[:2]
                preview = np.ones((h, w*2, 3), dtype=np.uint8) * 255
                preview[:, :w] = template
                preview[:, w:] = segmented
                plt.imsave(preview_path, preview)
                
                return {
                    'template': template_path,
                    'minimal': minimal_path,
                    'detailed': detailed_path,
                    'chart': chart_path,
                    'preview': preview_path,
                    'processed': processed_path,
                    'region_data': region_data,
                    'paintability': 50  # Default value for basic processing
                }
            except Exception as nested_e:
                print(f"Even fallback processing failed: {str(nested_e)}")
                traceback.print_exc()
                raise

    def generate(self, image_path, **kwargs):
        """Generate paint by numbers from an image"""
        # Start timing for performance measurement
        start_time = time.time()
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to RGB for processing
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Compute hash for caching
        settings_hash = self.performance_optimizer.compute_image_hash(image, kwargs)
        
        # Try to load from cache first
        cached_result = self.performance_optimizer.cache_result(settings_hash)
        if cached_result is not None:
            print(f"Using cached result (saved {time.time() - start_time:.2f} seconds)")
            return cached_result
        
        # Optimize image size for processing
        processed_image, scale = self.performance_optimizer.optimize_image_size(image)
        
        # ...existing processing steps...
        
        # If you have any parallelizable steps, use process_in_parallel
        # Example:
        # segmented_regions = self.performance_optimizer.process_in_parallel(
        #     regions, self.process_region
        # )
        
        # For operations that benefit from multi-scale processing:
        # feature_map = self.performance_optimizer.multi_scale_process(
        #     processed_image, self.detect_features, scales=[0.5, 0.75, 1.0]
        # )
        
        # ...rest of existing code...
        
        # Save result to cache before returning
        self.performance_optimizer.cache_result(settings_hash, result)
        
        print(f"Processing completed in {time.time() - start_time:.2f} seconds")
        return

    def generate_numbered_template(self, segmented_image, colors, complexity='medium', feature_mask=None):
        """Generate a template with numbers from a segmented image
    
        Args:
            segmented_image: The color-segmented image
            colors: List of colors used in the segmentation
            complexity: Complexity level ('low', 'medium', 'high')
            feature_mask: Optional mask highlighting important features (eyes, pet outline)
    
        Returns:
            Template image with numbers
        """
        import cv2
        import numpy as np
        
        # Create label image from the segmented image
        h, w = segmented_image.shape[:2]
        label_image = np.zeros((h, w), dtype=np.uint8)
        
        # Enhanced edge detection - use higher thresholds to reduce "wool" effect
        gray = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
        
        # Use bilateral filter to reduce noise while preserving edges
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Create pet outline - this is critical for seeing the shape
        pet_outline = np.zeros_like(gray)
        
        # Use simpler edge detection with higher thresholds
        edges = cv2.Canny(blurred, 150, 250)  # Higher thresholds = fewer edges
        
        # If we have feature mask, use it to prioritize important edges
        if feature_mask is not None:
            # Keep edges near important features
            important_areas = (feature_mask > 0.6).astype(np.uint8)
            important_areas = cv2.dilate(important_areas, np.ones((5,5),np.uint8))
            
            # Combine edges - keep strong edges and edges in important areas
            pet_outline = cv2.bitwise_and(edges, important_areas*255)
            
            # Add this to the main edges
            edges = cv2.bitwise_or(edges, pet_outline)
        
        # Extract pet contour - create a clearer outline of the entire shape
        mask = np.zeros_like(gray)
        mask[gray < 250] = 255  # Assume white/near-white is background
        
        # Find contours (pet outline)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw the pet outline with higher intensity (255)
        cv2.drawContours(edges, contours, -1, 255, 2)

        # Create a stronger mask specifically for the pet outline
        pet_mask = np.zeros_like(gray)
        _, threshold = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY_INV)  # Better threshold for pet/background

        # Remove small holes and noise
        kernel = np.ones((5,5), np.uint8)
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

        # Find only the main large contour (the pet)
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Sort by area and keep only the largest (the pet)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

        # Create a separate pet outline edge map
        pet_outline = np.zeros_like(edges)
        cv2.drawContours(pet_outline, contours, -1, 255, 3)  # Thicker outline (3px)

        # Add the pet outline to the main edges but with maximum intensity
        edges = cv2.bitwise_or(edges, pet_outline)
        
        # Find regions by comparing pixels to color palette
        region_data = []
        for i, color_info in enumerate(colors):
            color_rgb = np.array(color_info['rgb'])
            
            # Increase tolerance for better region merging
            tolerance = 15
            mask = np.all((segmented_image >= (color_rgb - tolerance)) & 
                          (segmented_image <= (color_rgb + tolerance)), axis=2)
            
            # Assign region ID to label image
            label_image[mask] = i + 1
            
            # Calculate region area and center
            area = np.sum(mask)
            if area > 0:
                y_coords, x_coords = np.where(mask)
                cx = int(np.mean(x_coords))
                cy = int(np.mean(y_coords))
                
                region_data.append({
                    'id': i,
                    'center': (cx, cy),
                    'area': int(area),
                    'color': color_rgb
                })

        # Map complexity to template style 
        if complexity == 'low':
            style = 'minimal'
        elif complexity == 'high':
            style = 'detailed'
        else:
            style = 'classic'
            
        # Use hierarchical edge style for better outline visibility
        return self.create_template(
            segmented_image.shape, 
            edges, 
            region_data, 
            label_image, 
            style=style, 
            edge_style='hierarchical'  # Changed from 'soft' to 'hierarchical'
        )
