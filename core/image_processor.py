import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy import ndimage
import time

class ImageProcessor:
    def __init__(self):
        pass
        
    def process_image(self, image, num_colors, simplification_level='medium', 
                    edge_strength=1.0, edge_width=1, enhance_dark_areas=False, 
                    dark_threshold=50, edge_style='normal', merge_regions_level='normal',
                    image_type='general'):
        """
        Process image to create paint-by-numbers data with improved color handling
        
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
        print("Starting image processing pipeline with enhanced algorithms...")
        
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
        
        # Step 3: Enhanced Color quantization with weighted K-means
        print(f"Performing enhanced color quantization with {num_colors} colors")
        h, w = image.shape[:2]
        pixels = smoothed.reshape(-1, 3).astype(np.float32)
        
        # Create feature importance map for weighting pixels
        feature_importance = self._create_feature_importance_map(image, image_type)
        
        # Create pixel weights based on importance and darkness
        pixel_weights = np.ones(pixels.shape[0])
        
        # Weight by importance (if we have a feature map)
        if feature_importance is not None:
            importance_flat = feature_importance.flatten()
            pixel_weights = pixel_weights * (1 + importance_flat * 2)  # Up to 3x weight
        
        # Weight by darkness (dark colors often underrepresented)
        darkness = 1 - (np.mean(pixels, axis=1) / 255)
        dark_boost = 1 + darkness * 2  # Boost dark pixels (up to 3x)
        pixel_weights = pixel_weights * dark_boost
        
        # Additional color weighting based on image type
        if image_type == 'pet' or image_type == 'portrait':
            # For pets and portraits, boost skin/fur tone colors
            hsv_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
            
            if image_type == 'pet':
                # Boost common fur colors
                fur_mask = (
                    ((hsv_pixels[:, 0] >= 20) & (hsv_pixels[:, 0] <= 40) & (hsv_pixels[:, 1] >= 30)) |  # Brown/orange
                    ((hsv_pixels[:, 0] >= 0) & (hsv_pixels[:, 0] <= 15) & (hsv_pixels[:, 1] <= 40))      # Black/gray/white
                )
                pixel_weights[fur_mask] *= 1.5
                
            elif image_type == 'portrait':
                # Boost skin tone colors
                skin_mask = (
                    ((hsv_pixels[:, 0] >= 0) & (hsv_pixels[:, 0] <= 50) & (hsv_pixels[:, 1] >= 10) & (hsv_pixels[:, 1] <= 150))
                )
                pixel_weights[skin_mask] *= 1.5
        
        # Create weighted samples for k-means
        sample_weights = pixel_weights / np.sum(pixel_weights)
        sample_indices = np.random.choice(
            len(pixels), 
            size=min(10000, len(pixels)),  # Cap samples for performance
            replace=True, 
            p=sample_weights
        )
        weighted_samples = pixels[sample_indices]
        
        # Use better K-means initialization for more stable results
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10, max_iter=300)
        kmeans.fit(weighted_samples)
        
        # Get color centers and labels
        centers = kmeans.cluster_centers_
        
        # Optimize the color palette if needed
        centers = self._optimize_color_palette(centers, image_type)
        
        # Assign pixels to closest center
        labels = np.zeros(len(pixels), dtype=np.int32)
        for i, pixel in enumerate(pixels):
            # Find closest color center
            distances = np.sum((centers - pixel) ** 2, axis=1)
            labels[i] = np.argmin(distances)
        
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
            
            # Perform the improved merging
            label_image = self.merge_small_regions(
                label_image, 
                centers, 
                min_region_percent=min_region_percent,
                color_similarity_threshold=color_threshold,
                feature_importance=feature_importance,  # Pass feature importance to merging
                image_type=image_type,  # Pass image type for specialized handling
                merge_regions_level=merge_regions_level  # Pass merge regions level
            )
            
            # Get unique labels after merging
            unique_labels = np.unique(label_image)
            
            # Update centers and create mapping for merged labels
            new_centers = np.zeros((len(unique_labels), 3))
            for i, label in enumerate(unique_labels):
                mask = (label_image == label)
                new_centers[i] = centers[label]
                
            centers = new_centers
            
            # Added new code: Safety check to ensure label indices match centers array size
            unique_labels = np.unique(label_image)
            if np.max(unique_labels) >= len(centers):
                print("Warning: Label indices don't match centers array. Remapping...")
                # Create a mapping from old labels to new consecutive indices
                label_mapping = {old_label: new_index for new_index, old_label in enumerate(unique_labels)}
                
                # Update the label image with new consecutive indices
                new_label_image = np.zeros_like(label_image)
                for old_label, new_label in label_mapping.items():
                    new_label_image[label_image == old_label] = new_label
                    
                # Update label_image to use the new consecutive indices
                label_image = new_label_image
        
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
    def _create_feature_importance_map(self, image, image_type):
        """
        Create a map highlighting important features in the image
        
        Parameters:
        - image: Input RGB image
        - image_type: Type of image
        
        Returns:
        - importance_map: 2D array with importance values [0-1]
        """
        h, w = image.shape[:2]
        importance_map = np.zeros((h, w), dtype=np.float32)
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 1. Edge importance (more edges = more important)
        edges = cv2.Canny(image, 50, 150)
        edge_density = cv2.GaussianBlur(edges.astype(float) / 255.0, (25, 25), 0)
        importance_map += edge_density * 0.5  # Edge weight
        
        # 2. Local contrast (higher contrast = more important)
        local_std = cv2.GaussianBlur(gray, (5, 5), 0)
        local_std = cv2.GaussianBlur(np.abs(gray - local_std), (21, 21), 0) / 255.0
        importance_map += local_std * 0.3  # Contrast weight
        
        # 3. Special handling based on image type
        if image_type == 'portrait' or image_type == 'pet':
            try:
                # Try to detect face/eyes for importance
                if image_type == 'portrait':
                    # Use face detection
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    for (x, y, w, h) in faces:
                        # Create higher importance for face regions
                        face_mask = np.zeros((h, w), dtype=np.float32)
                        # Higher importance in center, fading to edges
                        y_coords, x_coords = np.mgrid[0:h, 0:w]
                        face_mask = 1 - np.sqrt(((x_coords - w/2) / (w/2)) ** 2 + 
                                            ((y_coords - h/2) / (h/2)) ** 2)
                        face_mask = np.clip(face_mask, 0, 1)
                        
                        # Apply to importance map
                        importance_map[y:y+h, x:x+w] = np.maximum(
                            importance_map[y:y+h, x:x+w],
                            face_mask * 0.8  # Face importance weight
                        )
                        
                        # Try to detect eyes within face region
                        face_roi_gray = gray[y:y+h, x:x+w]
                        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                        eyes = eye_cascade.detectMultiScale(face_roi_gray)
                        
                        for (ex, ey, ew, eh) in eyes:
                            # Very high importance for eyes
                            eye_y, eye_x = y + ey, x + ex
                            cv2.circle(importance_map, (eye_x + ew//2, eye_y + eh//2), 
                                    max(ew, eh), 1.0, -1)
                
                elif image_type == 'pet':
                    # For pets, we'll use a combination of techniques to find eyes and face
                    # Pet eye detection - look for circular features in dark areas
                    pet_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')
                    pet_faces = pet_eye_cascade.detectMultiScale(gray, 1.1, 3)
                    
                    if len(pet_faces) > 0:
                        # Cat/dog face detected
                        for (x, y, w, h) in pet_faces:
                            # Create importance mask for pet face
                            importance_map[y:y+h, x:x+w] += 0.6
                            
                            # Typical eye positions - enhance these areas
                            eye_y = y + int(h * 0.4)  # Eyes typically in upper region
                            left_eye_x = x + int(w * 0.3)
                            right_eye_x = x + int(w * 0.7)
                            
                            # Add circular regions of importance for potential eyes
                            cv2.circle(importance_map, (left_eye_x, eye_y), h//8, 0.9, -1)
                            cv2.circle(importance_map, (right_eye_x, eye_y), h//8, 0.9, -1)
                    else:
                        # Fallback - use blob detection to find potential eyes
                        # particularly effective for dark eyes on light fur
                        
                        # Convert to HSV to isolate dark regions
                        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                        v_channel = hsv[:,:,2]
                        
                        # Look for dark blobs that could be eyes
                        _, dark_mask = cv2.threshold(v_channel, 50, 255, cv2.THRESH_BINARY_INV)
                        dark_mask = cv2.erode(dark_mask, np.ones((3, 3), np.uint8), iterations=1)
                        dark_mask = cv2.dilate(dark_mask, np.ones((5, 5), np.uint8), iterations=1)
                        
                        # Find contours that could be eyes
                        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        for contour in contours:
                            area = cv2.contourArea(contour)
                            if 50 < area < 500:  # Size filter for typical pet eyes
                                x, y, w, h = cv2.boundingRect(contour)
                                aspect_ratio = w / h
                                
                                # Eyes are typically somewhat circular
                                if 0.5 <= aspect_ratio <= 2.0:
                                    # Add importance to this region
                                    cv2.drawContours(importance_map, [contour], -1, 0.9, -1)
                                    # Add a larger region of importance around it
                                    cv2.drawContours(importance_map, [contour], -1, 0.6, 10)
            
            except Exception as e:
                print(f"Warning: Error in feature detection: {e}")
                # Continue without specialized detection
                pass
            
        # 4. Detect and emphasize high-detail regions
        detail = cv2.Laplacian(gray, cv2.CV_64F)
        detail = np.abs(detail)
        detail = detail / np.max(detail) if np.max(detail) > 0 else detail
        detail_map = cv2.GaussianBlur(detail, (15, 15), 0)
        importance_map += detail_map * 0.2
        
        # 5. Add importance to dark regions in bright images and vice versa (for contrast)
        avg_brightness = np.mean(gray) / 255.0
        if avg_brightness > 0.6:  # Bright image
            # Dark areas are important
            darkness = 1 - (gray / 255.0)
            importance_map += darkness * 0.2
        elif avg_brightness < 0.4:  # Dark image
            # Bright areas are important
            brightness = gray / 255.0
            importance_map += brightness * 0.2
        
        # Normalize importance to [0, 1] range
        importance_map = np.clip(importance_map, 0, 1)
        
        return importance_map

    def _optimize_color_palette(self, centers, image_type):
        """
        Optimize color palette for better visual results
        
        Parameters:
        - centers: Color centers from KMeans
        - image_type: Type of image
        
        Returns:
        - optimized_centers: Improved color centers
        """
        # Make a copy to avoid modifying original
        optimized_centers = centers.copy()
        
        # Ensure we have black and white in high-contrast images
        has_black = np.any(np.sum(centers, axis=1) < 100)
        has_white = np.any(np.sum(centers, axis=1) > 650)
        
        # For portrait and pet images, correct color balance
        if image_type in ['pet', 'portrait']:
            # Convert to HSV for better color manipulation
            centers_rgb = centers.reshape(-1, 1, 3).astype(np.uint8)
            centers_hsv = cv2.cvtColor(centers_rgb, cv2.COLOR_RGB2HSV).reshape(-1, 3)
            
            # Improve saturation for better visual impact
            for i in range(len(centers_hsv)):
                # Don't over-saturate near-black or near-white colors
                brightness = centers_hsv[i, 2]
                if 30 < brightness < 220:
                    # Boost saturation slightly
                    centers_hsv[i, 1] = min(255, int(centers_hsv[i, 1] * 1.1))
                    
            # Convert back to RGB
            optimized_centers_rgb = cv2.cvtColor(centers_hsv.reshape(-1, 1, 3), cv2.COLOR_HSV2RGB)
            optimized_centers = optimized_centers_rgb.reshape(-1, 3)
        
        # For landscapes, enhance color harmony
        elif image_type == 'landscape':
            # Calculate color statistics
            avg_color = np.mean(optimized_centers, axis=0)
            
            # Enhance green/blue balance for natural scenes
            for i in range(len(optimized_centers)):
                # Enhance blues in skies
                if optimized_centers[i, 2] > optimized_centers[i, 0] and optimized_centers[i, 2] > optimized_centers[i, 1]:
                    optimized_centers[i, 2] = min(255, int(optimized_centers[i, 2] * 1.1))
                
                # Enhance greens in foliage
                if optimized_centers[i, 1] > optimized_centers[i, 0] and optimized_centers[i, 1] > optimized_centers[i, 2]:
                    optimized_centers[i, 1] = min(255, int(optimized_centers[i, 1] * 1.1))
        
        return optimized_centers
    def _optimize_pet_color_palette(self, centers, pet_type='generic'):
        """
        Optimize color palette specifically for pet images
        
        Parameters:
        - centers: Color centers from k-means clustering
        - pet_type: Type of pet ('cat', 'dog', 'generic')
        
        Returns:
        - Optimized centers for pet coloring
        """
        optimized_centers = centers.copy()
        
        # Convert to HSV for better color manipulation
        centers_hsv = cv2.cvtColor(centers.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        
        # Process each color in the palette
        for i in range(len(centers_hsv)):
            # Get HSV values
            h, s, v = centers_hsv[i]
            
            # Don't modify colors that are too dark or too light
            if 30 < v < 225:
                # Slightly increase saturation for more vibrant fur colors
                s = min(255, int(s * 1.15))
                centers_hsv[i, 1] = s
                
                # For dog-specific processing
                if pet_type == 'dog':
                    # Enhance common dog fur colors (browns, tans)
                    if 15 <= h <= 30:  # Brown/tan range
                        centers_hsv[i, 1] = min(255, int(s * 1.2))  # More saturated browns
                
                # For cat-specific processing
                elif pet_type == 'cat':
                    # Enhance common cat colors (oranges, grays)
                    if (15 <= h <= 25) or (h <= 20 and s <= 40):  # Orange or gray
                        centers_hsv[i, 1] = min(255, int(s * 1.2))  # More saturated orange/defined gray
        
        # Convert back to RGB
        optimized_centers_rgb = cv2.cvtColor(centers_hsv.reshape(-1, 1, 3), cv2.COLOR_HSV2RGB)
        optimized_centers = optimized_centers_rgb.reshape(-1, 3)
        
        return optimized_centers
    def _enhance_dark_areas(self, image, dark_threshold):
        """
        Enhanced version for better dark area handling with adaptive approach
        
        Parameters:
        - image: Input RGB image
        - dark_threshold: Base threshold for dark area detection
        
        Returns:
        - Enhanced image with better dark area details
        """
        print(f"Enhancing dark areas with adaptive approach (base threshold: {dark_threshold})")
        
        # Set a maximum computation time limit
        max_processing_time = 5  # seconds
        start_time = time.time()
        
        try:
            # Create a copy to preserve original
            image_enhanced = image.copy()
            
            # Convert to HSV for better dark area detection
            hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            v_channel = hsv_img[:,:,2]
            
            # Analyze histogram to find better threshold adaptively
            hist = cv2.calcHist([v_channel], [0], None, [256], [0, 256])
            hist_cumsum = np.cumsum(hist) / np.sum(hist)
            
            # Check if we're taking too long already
            if time.time() - start_time > max_processing_time:
                print("Dark area enhancement taking too long, using simplest approach")
                # Just use the original dark threshold
                dark_mask = v_channel < dark_threshold
                gradient_mask = dark_mask.astype(np.float32)
                gradient_mask_3ch = np.stack([gradient_mask] * 3, axis=2)
                enhanced_rgb = np.minimum(image_enhanced * 1.3, 255).astype(np.uint8)
                return image_enhanced * (1 - gradient_mask_3ch) + enhanced_rgb * gradient_mask_3ch
            
            # Find threshold that covers the darkest 15% of pixels
            # This is more adaptive than using a fixed threshold
            for i in range(256):
                if hist_cumsum[i] > 0.15:
                    adaptive_threshold = max(dark_threshold - 10, i + 5)
                    break
            else:
                adaptive_threshold = dark_threshold
            
            # Create an adaptive dark mask
            dark_mask = v_channel < adaptive_threshold
            
            # Calculate dark area percentage
            dark_pixel_percentage = np.sum(dark_mask) / dark_mask.size * 100
            print(f"Dark area detected: {dark_pixel_percentage:.1f}% (adaptive threshold: {adaptive_threshold})")
            
            # Check processing time again
            if time.time() - start_time > max_processing_time:
                print("Dark area enhancement taking too long, using simplified processing")
                gradient_mask = dark_mask.astype(np.float32)
                gradient_mask_3ch = np.stack([gradient_mask] * 3, axis=2)
                enhanced_rgb = np.minimum(image_enhanced * 1.3, 255).astype(np.uint8)
                return image_enhanced * (1 - gradient_mask_3ch) + enhanced_rgb * gradient_mask_3ch
            
            if np.any(dark_mask):
                # Create gradient mask for smooth transition
                gradient_mask = dark_mask.astype(np.float32)
                gradient_mask = cv2.GaussianBlur(gradient_mask, (15, 15), 0)
                
                # Check processing time again
                if time.time() - start_time > max_processing_time:
                    print("Dark area enhancement taking too long, using simplified processing")
                    gradient_mask_3ch = np.stack([gradient_mask] * 3, axis=2)
                    enhanced_rgb = np.minimum(image_enhanced * 1.3, 255).astype(np.uint8)
                    return image_enhanced * (1 - gradient_mask_3ch) + enhanced_rgb * gradient_mask_3ch
                
                # Choose CLAHE parameters based on image characteristics
                if dark_pixel_percentage > 40:  # Very dark image
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
                else:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                
                # Convert to LAB color space (better for enhancement)
                lab = cv2.cvtColor(image_enhanced, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # Check processing time again
                if time.time() - start_time > max_processing_time:
                    print("Dark area enhancement taking too long, skipping advanced processing")
                    return image_enhanced
                
                # Apply CLAHE to the L channel
                l_enhanced = clahe.apply(l)
                
                # Merge channels back
                lab_enhanced = cv2.merge((l_enhanced, a, b))
                enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
                
                # Prepare for blending
                gradient_mask_3ch = np.stack([gradient_mask] * 3, axis=2)
                
                # Blend enhanced and original using gradient mask for smooth transition
                image_enhanced = image_enhanced * (1 - gradient_mask_3ch) + enhanced_rgb * gradient_mask_3ch
                
                # Check processing time again
                if time.time() - start_time > max_processing_time:
                    print("Dark area enhancement taking too long, skipping gamma correction")
                    return image_enhanced
                
                # Additional gamma correction specifically for dark areas
                if dark_pixel_percentage > 20:
                    # Create adaptive gamma based on darkness
                    gamma = 0.7 if dark_pixel_percentage > 50 else 0.8
                    lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(0, 256)]).astype("uint8")
                    
                    # Apply gamma correction only to dark areas with smooth transition
                    for i in range(3):
                        # Only apply to darker parts of the image
                        temp = image_enhanced[:,:,i].copy()
                        temp = cv2.LUT(temp, lookup_table)
                        image_enhanced[:,:,i] = image_enhanced[:,:,i] * (1 - gradient_mask_3ch[:,:,i]) + \
                                                temp * gradient_mask_3ch[:,:,i]
                
                # Skip the edge enhancement if we're running out of time
                if time.time() - start_time > max_processing_time:
                    return image_enhanced
                
                # Skip edge enhancement and additional operations to save time
            
            return image_enhanced
        
        except Exception as e:
            print(f"Error in dark area enhancement: {e}")
            # If anything fails, return original image
            return image
    
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
    def merge_small_regions(self, label_image, centers, min_region_percent=0.5, 
                       color_similarity_threshold=30, feature_importance=None,
                       image_type='general', merge_regions_level='normal'):
        """
        Merge small regions into neighboring regions with similar colors,
        while preserving important features
        
        Parameters:
        - label_image: Image with region labels
        - centers: Color centers from KMeans
        - min_region_percent: Minimum region size as percentage of total pixels (0.1-5.0)
        - color_similarity_threshold: Maximum color distance for merging (0-255)
        - feature_importance: Map of important features to preserve
        - image_type: Type of image for specialized handling
        - merge_regions_level: Level of region merging ('none', 'low', 'normal', 'aggressive', 'smart')
        
        Returns:
        - Updated label_image with merged regions
        """
        h, w = label_image.shape
        total_pixels = h * w
        
        # Apply smart merging logic for pet images
        if merge_regions_level == 'smart' and image_type == 'pet':
            # Create a mask of important features for pets (eyes, nose, etc.)
            is_near_feature = np.zeros((h, w), dtype=bool)
            
            if feature_importance is not None:
                # Areas with importance > 0.6 are considered important features
                is_near_feature = feature_importance > 0.6
                
                # Dilate to get areas near features too
                is_near_feature = cv2.dilate(
                    is_near_feature.astype(np.uint8), 
                    np.ones((15, 15), np.uint8)
                ).astype(bool)
                
                # Adjust merging parameters based on feature proximity
                fur_min_region_percent = min_region_percent * 2.0  # More aggressive in fur areas
                feature_min_region_percent = min_region_percent * 0.5  # Less merging near features
                
                fur_threshold = color_similarity_threshold + 20  # More relaxed matching in fur
                feature_threshold = color_similarity_threshold - 10  # Stricter matching near features
                
                print("Using smart region merging for pet image - preserving facial features")
        
        # Calculate minimum region size in pixels
        min_pixels = int(total_pixels * min_region_percent / 100)
        print(f"Merging small regions with feature preservation (min size: {min_region_percent}%, threshold: {min_pixels} pixels)")
        
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
        
        # Calculate region importance if we have feature importance
        region_importance = {}
        is_important_region = {}
        
        if feature_importance is not None:
            for label in unique_labels:
                region_mask = (label_image == label)
                if np.any(region_mask):
                    # Calculate average feature importance in this region
                    avg_importance = np.mean(feature_importance[region_mask])
                    region_importance[label] = avg_importance
                    is_important_region[label] = avg_importance > 0.7
        
        # Process each small region
        for small_label in sorted_small_regions:
            # Skip if this region was already merged
            if small_label not in np.unique(merged_labels):
                continue
                
            # Create mask for this region
            small_mask = (merged_labels == small_label)
            
            # Check if this region has high importance
            is_high_importance = False
            if small_label in is_important_region:
                is_high_importance = is_important_region[small_label]
            
            # Check if this region is near important features
            near_feature = False
            if merge_regions_level == 'smart' and image_type == 'pet' and feature_importance is not None:
                # Check if this region overlaps with the near-feature mask
                near_feature = np.any(small_mask & is_near_feature)
            
            # Determine if this is a dark region
            is_dark_region = np.mean(centers[small_label]) < 60
            
            # Customize thresholds based on importance, darkness, and features
            current_threshold = color_similarity_threshold
            current_min_pixels = min_pixels
            
            # Apply smart merging logic
            if merge_regions_level == 'smart' and image_type == 'pet':
                if near_feature:
                    current_threshold = feature_threshold
                    print(f"Region {small_label} is near facial features - using stricter threshold")
                else:
                    current_threshold = fur_threshold
            
            if is_high_importance:
                # More strict threshold for important regions
                current_threshold *= 0.7
                print(f"Preserving important region with label {small_label} (stricter threshold)")
            
            if is_dark_region:
                # More strict threshold for dark regions too
                current_threshold *= 0.8
                print(f"Preserving dark region with label {small_label} (stricter threshold)")
            
            # Find boundaries by dilating the mask and subtracting original
            dilated = cv2.dilate(small_mask.astype(np.uint8), np.ones((3,3), np.uint8))
            boundary = dilated.astype(bool) & ~small_mask
            
            # Get neighboring labels
            neighbor_labels = np.unique(merged_labels[boundary])
            neighbor_labels = [n for n in neighbor_labels if n != small_label]
            
            if not neighbor_labels:
                continue
                
            # Find best neighbor based on color similarity and importance
            best_neighbor = None
            min_score = float('inf')
            
            for neighbor in neighbor_labels:
                # Calculate color difference
                color_diff = np.sum(np.abs(centers[small_label] - centers[neighbor]))
                
                # Base score is the color difference
                score = color_diff
                
                # If we have importance data, consider region importance
                if feature_importance is not None:
                    if neighbor in region_importance:
                        # Penalize merging with high-importance region if this region is low importance
                        if region_importance[neighbor] > 0.6 and region_importance.get(small_label, 0) < 0.4:
                            score += 30
                        
                        # Favor merging with similar-importance regions
                        importance_diff = abs(region_importance.get(small_label, 0) - region_importance[neighbor])
                        score += importance_diff * 20
                
                # Penalize merging dark and light regions
                neighbor_darkness = np.mean(centers[neighbor]) < 60
                if is_dark_region != neighbor_darkness:
                    score += 40
                
                # Update best neighbor if this one has a lower score
                if score < min_score:
                    min_score = score
                    best_neighbor = neighbor
            
            # Only merge if color difference is acceptable
            if min_score <= current_threshold:
                merged_labels[small_mask] = best_neighbor
                regions_merged += 1
            else:
                print(f"Preserved region {small_label} due to high color difference ({min_score:.1f} > {current_threshold:.1f})")
        
        print(f"Merged {regions_merged} regions")
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