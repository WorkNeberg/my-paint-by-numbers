import os
import cv2
import numpy as np
import time
import hashlib
import pickle
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

class PerformanceOptimizer:
    """
    Provides performance optimization for image processing operations
    including caching, parallel processing, and smart downsizing.
    """
    
    def __init__(self, cache_dir=None, max_workers=None):
        """
        Initialize the performance optimizer
        
        Args:
            cache_dir: Directory to use for caching. If None, uses a default
            max_workers: Maximum number of worker threads for parallel operations
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), '../cache')
            
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Use default number of workers based on CPU count if not specified
        self.max_workers = max_workers or min(os.cpu_count(), 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
    def compute_image_hash(self, image, settings=None):
        """
        Compute a hash for an image and its processing settings
        
        Args:
            image: Input image
            settings: Optional dictionary of settings that affect processing
            
        Returns:
            String hash that uniquely identifies this image+settings combination
        """
        # Create a reduced size copy for faster hashing
        small_img = cv2.resize(image, (100, 100))
        img_bytes = small_img.tobytes()
        
        # Add settings to the hash if provided
        if settings:
            # Convert settings dict to a sorted, stable string representation
            settings_str = str(sorted(settings.items()))
            img_bytes += settings_str.encode()
            
        # Create hash
        return hashlib.md5(img_bytes).hexdigest()
        
    def cache_result(self, key, data=None):
        """
        Store or retrieve data from cache
        
        Args:
            key: Cache key (string)
            data: Data to cache. If None, attempts to load from cache
            
        Returns:
            Cached data if loading, None if storing or cache miss
        """
        cache_path = os.path.join(self.cache_dir, f"{key}.pkl")
        
        if data is None:
            # Try to load from cache
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    print(f"Cache load error: {e}")
                    return None
            return None
        else:
            # Save to cache
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
                return None
            except Exception as e:
                print(f"Cache save error: {e}")
                return None
    
    def optimize_image_size(self, image, target_size=1200):
        """
        Resize image to optimal processing size while maintaining aspect ratio
        
        Args:
            image: Input image
            target_size: Target maximum dimension
            
        Returns:
            Resized image, scale factor
        """
        h, w = image.shape[:2]
        
        # If image is already smaller than target size, return original
        if max(h, w) <= target_size:
            return image, 1.0
            
        # Calculate scale factor
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return resized, scale
        
    def process_in_parallel(self, regions, process_func):
        """
        Process image regions in parallel
        
        Args:
            regions: List of image regions to process
            process_func: Function to apply to each region
            
        Returns:
            List of processed regions
        """
        # Submit all tasks to the thread pool
        futures = [self.executor.submit(process_func, region) for region in regions]
        
        # Wait for all tasks to complete and collect results
        results = [future.result() for future in futures]
        
        return results
        
    def multi_scale_process(self, image, process_func, scales=[0.5, 1.0]):
        """
        Process image at multiple scales and combine results
        
        Args:
            image: Input image
            process_func: Function that processes the image
            scales: List of scales to process at (1.0 = original size)
            
        Returns:
            Combined result
        """
        results = []
        
        for scale in scales:
            if scale == 1.0:
                scaled_img = image
            else:
                h, w = image.shape[:2]
                scaled_img = cv2.resize(image, (int(w*scale), int(h*scale)), 
                                       interpolation=cv2.INTER_AREA)
            
            # Process at this scale
            result = process_func(scaled_img)
            
            # Resize result back to original size if needed
            if scale != 1.0:
                h, w = image.shape[:2]
                result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
            
            results.append(result)
        
        # Combine results (implementation depends on the type of result)
        # For binary masks, use bitwise OR
        if isinstance(results[0], np.ndarray) and results[0].dtype == np.uint8:
            combined = results[0]
            for r in results[1:]:
                combined = cv2.bitwise_or(combined, r)
            return combined
        
        # For other cases, return the result from the highest resolution
        return results[-1]  # Assuming scales are in ascending order