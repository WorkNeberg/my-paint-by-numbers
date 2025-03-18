import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy import ndimage

class ImageProcessor:
    def __init__(self):
        pass
        
    def process_image(self, image, num_colors, simplification_level='medium', 
                    edge_strength=1.0, edge_width=1, enhance_dark_areas=False, 
                    dark_threshold=50, edge_style='normal', merge_regions_level='normal',
                    image_type='general'):
        """
        Process image to create paint-by-numbers data
        
        Parameters:
        - image: Input image
        - num_colors: Number of colors to use
        - simplification_level: Level of detail ('low', 'medium', 'high')
        - edge_strength: Strength of edge lines (0.5-1.5)
        - edge_width: Width of edge lines in pixels
        - enhance_dark_areas: Whether to enhance dark areas
        - dark_threshold: Threshold for dark area enhancement
        - edge_style: Style of edges ('normal', 'soft', 'thin')
        - merge_regions_level: Level of region merging ('none', 'low', 'normal', 'aggressive')
        - image_type: Type of image ('general', 'portrait', 'pet', 'cartoon', 'landscape')
        
        Returns:
        - vectorized: Image with reduced colors
        - label_image: Image with region labels
        - edges: Edge map
        - centers: Color centers
        - region_data: Information about each region
        - paintability: Score indicating how suitable the image is for painting
        """
        print("Starting image processing pipeline...")
        
        # Step 1: Apply dark area enhancement if requested
        if enhance_dark_areas:
            image = self._enhance_dark_areas(image, dark_threshold)
        
        # Step 2: Apply blur based on simplification level
        if simplification_level == 'low':
            blur_size = 3
        elif simplification_level == 'medium':
            blur_size = 5
        elif simplification_level == 'high':
            blur_size = 7
        else:  # 'extreme'
            blur_size = 9
            
        # Apply bilateral filter for edge-preserving smoothing
        print(f"Applying bilateral filter with size {blur_size}")
        smoothed = cv2.bilateralFilter(image, blur_size, 75, 75)
        
        # Step 3: Color quantization with K-means
        print(f"Performing color quantization with {num_colors} colors")
        h, w = image.shape[:2]
        pixels = smoothed.reshape(-1, 3).astype(np.float32)
        
        # Use better K-means initialization for more stable results
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10, max_iter=300)
        kmeans.fit(pixels)
        
        # Get color centers and labels
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        # Create label image
        label_image = labels.reshape(h, w)
        
        # Step 4: Merge small regions if requested
        if merge_regions_level != 'none':
            print(f"Applying region merging level: {merge_regions_level}")
            
            # Set merging parameters based on level and image type
            if merge_regions_level == 'low':
                min_region_percent = 0.2
                color_threshold = 40
            elif merge_regions_level == 'aggressive':
                min_region_percent = 1.0
                color_threshold = 60
            else:  # 'normal'
                min_region_percent = 0.5
                color_threshold = 50
                
            # Adjust based on image type
            if image_type == 'pet':
                min_region_percent *= 1.5  # More aggressive merging for pets
            elif image_type == 'portrait':
                min_region_percent *= 1.2  # Somewhat aggressive for portraits
            
            # Perform the merging
            label_image = self.merge_small_regions(
                label_image, 
                centers, 
                min_region_percent=min_region_percent,
                color_similarity_threshold=color_threshold
            )
            
            # Get unique labels after merging
            unique_labels = np.unique(label_image)
            
            # Update centers and create mapping for merged labels
            new_centers = np.zeros((len(unique_labels), 3))
            for i, label in enumerate(unique_labels):
                mask = (label_image == label)
                new_centers[i] = centers[label]
                
            centers = new_centers
        
        # Step 5: Recreate the quantized image using updated labels
        flat_labels = label_image.flatten()
        quantized = centers[flat_labels].reshape(image.shape).astype(np.uint8)
        
        # Step 6: Detect edges with improved approach
        print(f"Detecting edges with style: {edge_style}")
        edges = self._detect_edges(label_image, edge_strength, edge_width, edge_style)
        
        # Step 7: Extract region data
        print("Extracting region data")
        region_data = self._extract_regions(label_image, centers)
        
        # Step 8: Calculate paintability score
        print("Calculating paintability score")
        paintability = self._calculate_paintability(region_data, len(centers))
        
        print(f"Processing complete. Paintability score: {paintability}")
        return quantized, label_image, edges, centers, region_data, paintability
    def _enhance_dark_areas(self, image, dark_threshold):
        """Enhance dark areas in the image to improve detail"""
        print(f"Enhancing dark areas (threshold: {dark_threshold})")
        
        # Create a copy to preserve original
        image_enhanced = image.copy()
        
        # Identify dark areas (below threshold)
        dark_mask = np.max(image, axis=2) < dark_threshold
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to dark areas
        if np.any(dark_mask):
            # Convert to LAB color space (better for contrast enhancement)
            lab = cv2.cvtColor(image_enhanced, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to the L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l_enhanced = clahe.apply(l)
            
            # Merge channels back
            lab_enhanced = cv2.merge((l_enhanced, a, b))
            enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
            
            # Only replace the dark areas
            image_enhanced[dark_mask] = enhanced_rgb[dark_mask]
            
            # Calculate enhancement percentage
            dark_pixel_percentage = np.sum(dark_mask) / dark_mask.size * 100
            print(f"Enhanced {dark_pixel_percentage:.1f}% of dark pixels")
            
            # Optional: add gamma correction for extra detail in dark areas
            if dark_pixel_percentage > 30:  # If image is very dark
                lookup_table = np.array([((i / 255.0) ** 0.8) * 255 for i in range(0, 256)]).astype("uint8")
                # Apply gamma correction only to dark areas
                for i in range(3):
                    image_enhanced[dark_mask, i] = cv2.LUT(image_enhanced[dark_mask, i], lookup_table)
        
        return image_enhanced
    
    def _detect_edges(self, label_image, edge_strength, edge_width, edge_style='normal'):
        """
        Detect edges between different regions with various styles
        
        Parameters:
        - label_image: Image with region labels
        - edge_strength: Strength of edges (0.5-1.5)
        - edge_width: Width of edges
        - edge_style: Style of edges ('normal', 'soft', 'thin', 'adaptive')
        
        Returns:
        - edges: Edge map
        """
        h, w = label_image.shape
        edges = np.zeros((h, w), dtype=np.uint8)
        
        # Find edges by comparing adjacent pixels
        horiz_edges = (label_image[:-1, :] != label_image[1:, :])
        vert_edges = (label_image[:, :-1] != label_image[:, 1:])
        
        # Combine horizontal and vertical edges
        edges[:-1, :][horiz_edges] = 255
        edges[1:, :][horiz_edges] = 255
        edges[:, :-1][vert_edges] = 255
        edges[:, 1:][vert_edges] = 255
        
        # Apply different edge styles
        if edge_style == 'soft':
            # Use a grayscale value instead of black
            edges = (edges * 0.7).astype(np.uint8)  # 70% intensity
            
            # Apply slight blur to soften edges
            edges = cv2.GaussianBlur(edges, (3, 3), 0)
            
        elif edge_style == 'thin':
            # Thin the edges
            if edge_width > 1:
                kernel = np.ones((2, 2), np.uint8)
                edges = cv2.erode(edges, kernel)
            
        elif edge_style == 'adaptive':
            # Create adaptive edges based on region size
            # This requires additional processing to identify region sizes
            # For now, we'll just use soft edges
            edges = (edges * 0.8).astype(np.uint8)
        
        # Apply edge width
        if edge_width > 1 and edge_style != 'thin':
            kernel = np.ones((edge_width, edge_width), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Apply edge strength
        if edge_strength != 1.0:
            edges = (edges * edge_strength).astype(np.uint8)
        
        return edges
    def _extract_regions(self, label_image, centers):
        """Extract region data from labeled image"""
        region_data = []
        
        for region_id in range(len(centers)):
            # Create binary mask for this region
            mask = (label_image == region_id).astype(np.uint8)
            
            # Calculate area
            area = np.sum(mask)
            
            if area > 0:
                # Find center
                cy, cx = ndimage.center_of_mass(mask)
                
                # Calculate compactness (circularity)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    perimeter = cv2.arcLength(largest_contour, True)
                    compactness = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
                else:
                    compactness = 0
                
                # Add region data
                region_data.append({
                    'id': region_id,
                    'center': (int(cx), int(cy)),
                    'area': int(area),
                    'color': centers[region_id],
                    'compactness': compactness
                })
        
        return region_data
    def merge_small_regions(self, label_image, centers, min_region_percent=0.5, color_similarity_threshold=30):
        """
        Merge small regions into neighboring regions with similar colors
        
        Parameters:
        - label_image: Image with region labels
        - centers: Color centers from KMeans
        - min_region_percent: Minimum region size as percentage of total pixels (0.1-5.0)
        - color_similarity_threshold: Maximum color distance for merging (0-255)
        
        Returns:
        - Updated label_image with merged regions
        """
        h, w = label_image.shape
        total_pixels = h * w
        min_pixels = int(total_pixels * min_region_percent / 100)
        
        print(f"Merging small regions (min size: {min_region_percent}%, threshold: {min_pixels} pixels)")
        
        # Find region sizes
        unique_labels, label_counts = np.unique(label_image, return_counts=True)
        
        # Identify small regions to merge
        small_regions = [label for label, count in zip(unique_labels, label_counts) 
                        if count < min_pixels]
        
        if not small_regions:
            print("No small regions found to merge")
            return label_image
            
        print(f"Found {len(small_regions)} regions to merge")
        
        # Create a copy of label_image for merging
        merged_labels = label_image.copy()
        regions_merged = 0
        
        # Sort small regions by size (smallest first)
        region_sizes = {label: count for label, count in zip(unique_labels, label_counts)}
        sorted_small_regions = sorted(small_regions, key=lambda x: region_sizes.get(x, 0))
        
        # Process each small region
        for small_label in sorted_small_regions:
            # Skip if this region was already merged
            if small_label not in np.unique(merged_labels):
                continue
                
            # Create mask for this region
            small_mask = (merged_labels == small_label)
            
            # Find boundaries by dilating the mask and subtracting original
            dilated = cv2.dilate(small_mask.astype(np.uint8), np.ones((3,3), np.uint8))
            boundary = dilated.astype(bool) & ~small_mask
            
            # Get neighboring labels
            neighbor_labels = np.unique(merged_labels[boundary])
            neighbor_labels = [n for n in neighbor_labels if n != small_label]
            
            if not neighbor_labels:
                continue
                
            # Find best neighbor based on color similarity
            best_neighbor = None
            min_color_diff = float('inf')
            
            for neighbor in neighbor_labels:
                # Calculate color difference
                color_diff = np.sum(np.abs(centers[small_label] - centers[neighbor]))
                
                if color_diff < min_color_diff:
                    min_color_diff = color_diff
                    best_neighbor = neighbor
            
            # Only merge if colors are similar enough
            if min_color_diff <= color_similarity_threshold:
                merged_labels[small_mask] = best_neighbor
                regions_merged += 1
        
        print(f"Merged {regions_merged} regions")
        
        # Ensure label indices are consecutive
        remaining_labels = np.unique(merged_labels)
        label_map = {old: new for new, old in enumerate(remaining_labels)}
        for old_label, new_label in label_map.items():
            merged_labels[merged_labels == old_label] = new_label
        
        # Verify the reduction
        final_regions = len(np.unique(merged_labels))
        original_regions = len(np.unique(label_image))
        print(f"Reduced from {original_regions} to {final_regions} regions")
        
        return merged_labels
    def _calculate_paintability(self, region_data, num_colors):
        """Calculate paintability score based on region properties"""
        if not region_data:
            return 50  # Default score
            
        avg_region_area = np.mean([r['area'] for r in region_data])
        min_region_area = np.min([r['area'] for r in region_data])
        avg_compactness = np.mean([r.get('compactness', 0) for r in region_data])
        
        # Factors in score:
        # 1. Number of colors (fewer is easier)
        # 2. Average region size (larger is easier)
        # 3. Smallest region size (larger is easier)
        # 4. Compactness (more compact regions are easier)
        
        color_factor = max(0, 100 - (num_colors - 5) * 3)
        size_factor = min(100, avg_region_area / 100)
        min_size_factor = min(100, min_region_area / 10)
        compactness_factor = min(100, avg_compactness * 100)
        
        # Calculate weighted score
        paintability = int(
            color_factor * 0.3 + 
            size_factor * 0.3 + 
            min_size_factor * 0.2 +
            compactness_factor * 0.2
        )
        
        return max(0, min(100, paintability))  # Clamp to 0-100