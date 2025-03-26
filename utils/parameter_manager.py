import json
import os
import logging

logger = logging.getLogger('pbn-app.parameter_manager')

class ParameterManager:
    """Manages loading and applying settings for image processing"""
    
    def __init__(self, config_path=None):
        """Initialize the parameter manager with a config file"""
        if config_path is None:
            # Default to the configs directory in the project
            base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
            config_path = os.path.join(base_dir, 'configs', 'image_settings.json')
            
        self.config = self._load_config(config_path)
        self.param_descriptions = {
            "colors": "Number of distinct colors in the output (5-30)",
            "edge_strength": "Strength of edge detection (0.5-3.0)",
            "edge_width": "Width of edges in pixels (1-5)",
            "simplification_level": "Level of image simplification (low, medium, high, very_high)",
            "edge_style": "Style of edges (standard, hierarchical)",
            "merge_regions_level": "Aggressiveness of region merging (normal, smart, aggressive)"
        }
        
    def _load_config(self, config_path):
        """Load and parse the JSON config file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Return a minimal default configuration
            return {
                "base": {
                    "colors": 15,
                    "simplification_level": "medium"
                },
                "image_types": {}
            }
    
    def get_settings(self, image_type, variant=None, custom_params=None):
        """
        Get settings by merging base, image type, variant, and custom parameters.
        
        Args:
            image_type: Type of image (pet, portrait, landscape)
            variant: Optional variant name (standard, detailed, simplified)
            custom_params: Optional dictionary of custom parameters
            
        Returns:
            Dictionary of merged settings
        """
        # Start with base settings
        settings = self.config["base"].copy()
        
        # Apply image type settings if available
        if image_type in self.config["image_types"]:
            image_config = self.config["image_types"][image_type]
            
            # Apply base settings for this image type
            if "base" in image_config:
                settings.update(image_config["base"])
            
            # If no variant specified, use the recommended one
            if variant is None and "recommended" in image_config:
                variant = image_config["recommended"]
                logger.info(f"Using recommended variant '{variant}' for {image_type}")
            
            # Apply variant settings if specified and available
            if variant and "variants" in image_config and variant in image_config["variants"]:
                settings.update(image_config["variants"][variant])
                logger.info(f"Applied variant '{variant}' settings for {image_type}")
        
        # Apply custom parameters last (highest priority)
        if custom_params:
            # Filter out None values
            filtered_params = {k: v for k, v in custom_params.items() if v is not None}
            settings.update(filtered_params)
            logger.info(f"Applied custom parameters: {filtered_params}")
        
        # Validate the settings
        self._validate_settings(settings)
        
        return settings
    
    def _validate_settings(self, settings):
        """Validate that settings are within acceptable ranges"""
        # Colors should be between 5 and 30
        if "colors" in settings:
            settings["colors"] = max(5, min(30, settings["colors"]))
        
        # Edge strength should be between 0.5 and 3.0
        if "edge_strength" in settings:
            settings["edge_strength"] = max(0.5, min(3.0, settings["edge_strength"]))
        
        # Edge width should be between 1 and 5
        if "edge_width" in settings:
            settings["edge_width"] = max(1, min(5, settings["edge_width"]))
        
        # Simplification level should be one of the valid options
        if "simplification_level" in settings:
            valid_levels = ["low", "medium", "high", "very_high"]
            if settings["simplification_level"] not in valid_levels:
                settings["simplification_level"] = "medium"
                
        # Edge style should be one of the valid options
        if "edge_style" in settings:
            valid_styles = ["standard", "hierarchical"]
            if settings["edge_style"] not in valid_styles:
                settings["edge_style"] = "standard"
                
        # Merge regions level should be one of the valid options
        if "merge_regions_level" in settings:
            valid_levels = ["normal", "smart", "aggressive"]
            if settings["merge_regions_level"] not in valid_levels:
                settings["merge_regions_level"] = "normal"
    
    def get_available_variants(self, image_type):
        """Get available variants for an image type"""
        if image_type in self.config["image_types"] and "variants" in self.config["image_types"][image_type]:
            variants = self.config["image_types"][image_type]["variants"]
            return {name: data.get("description", "") for name, data in variants.items()}
        return {}
    
    def get_parameter_description(self, param_name):
        """Get description for a parameter"""
        return self.param_descriptions.get(param_name, "No description available")