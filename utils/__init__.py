# filepath: c:\PROJECTS\Paint by numbers\utils\__init__.py
"""
Utility functions for the Paint by Numbers application.
"""

import os
import logging

# Import utilities that we will use in the enhanced version
from .parameter_manager import ParameterManager
# PDF functionality to be reimplemented in Stage 7
# from .pdf_generator import PBNPdfGenerator
from .variant_manager import VariantManager

logger = logging.getLogger('pbn-app.utils')

# Basic file utility functions
def ensure_directory_exists(directory_path):
    """Create a directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        logger.info(f"Created directory: {directory_path}")
    return directory_path

def get_file_extension(filename):
    """Get the extension of a file without the dot"""
    return os.path.splitext(filename)[1][1:].lower()

# Basic image utility functions
def resize_image_to_fit(image, max_size):
    """Resize an image to fit within max_size while preserving aspect ratio"""
    import cv2
    import numpy as np
    
    height, width = image.shape[:2]
    
    if height <= max_size and width <= max_size:
        return image
    
    # Calculate the scaling factor
    scale = min(max_size / width, max_size / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize the image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized

def convert_to_rgb(image):
    """Convert BGR image to RGB"""
    import cv2
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

__all__ = [
    'ParameterManager',
    # 'PBNPdfGenerator',  # Will be reimplemented in Stage 7
    'VariantManager',
    'ensure_directory_exists',
    'get_file_extension',
    'resize_image_to_fit',
    'convert_to_rgb'
]