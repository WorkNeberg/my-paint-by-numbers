import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from core.image_processor import ImageProcessor
from core.number_placement import NumberPlacer
from core.image_analyzer import ImageAnalyzer

class PaintByNumbersGenerator:
    def __init__(self):
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
    
    def create_template(self, image_shape, edges, region_data, label_image, style="classic", edge_style="normal"):
        """
        Create the paint-by-numbers template image
        
        Parameters:
        - image_shape: Shape of the original image
        - edges: Edge map
        - region_data: Information about each region
        - label_image: Label map from segmentation (needed for number placement)
        - style: Style of number placement ('classic', 'minimal', 'detailed')
        - edge_style: Style of edges ('normal', 'soft', 'thin')
        """
        # Create white background
        template = np.ones((*image_shape[:2], 3), dtype=np.uint8) * 255
        
        # Place numbers according to selected style
        numbered_template = self.number_placer.place_numbers(
            label_image=label_image,  # FIXED: Use the actual label image
            region_data=region_data,
            style=style
        )
        
        # Apply edges with style
        if edge_style == 'soft':
            # Use gray edges instead of black
            for i in range(3):
                numbered_template[:,:,i] = np.where(edges > 0, 180, numbered_template[:,:,i])
        elif edge_style == 'thin':
            # Thin the edges before applying
            kernel = np.ones((2,2), np.uint8)
            thin_edges = cv2.erode(edges, kernel)
            for i in range(3):
                numbered_template[:,:,i] = np.where(thin_edges > 0, 0, numbered_template[:,:,i])
        else:  # normal
            # Standard black edges
            for i in range(3):
                numbered_template[:,:,i] = np.where(edges > 0, 0, numbered_template[:,:,i])
        
        return numbered_template
    
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