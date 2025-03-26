import os
import json
import logging
import copy
import time

logger = logging.getLogger('pbn-app.settings_manager')

class SettingsManager:
    """Manages settings and presets for the Paint by Numbers processor"""
    
    def __init__(self, config_path=None, metadata_path=None):
        """Initialize the settings manager with config path"""
        if config_path is None:
            # Default to the configs directory in the project
            base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
            config_path = os.path.join(base_dir, 'configs', 'image_settings.json')
            
        if metadata_path is None:
            # Default to the configs directory in the project
            base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
            metadata_path = os.path.join(base_dir, 'configs', 'parameter_metadata.json')
        
        self.config_path = config_path
        self.metadata_path = metadata_path
        self.settings = self._load_config()
        self.valid_values = self._initialize_validation_rules()
        self.parameter_metadata = self._load_parameter_metadata()
        self.parameter_dependencies = self._initialize_parameter_dependencies()
        self.presets_dir = os.path.join(os.path.dirname(config_path), 'presets')
        os.makedirs(self.presets_dir, exist_ok=True)
        
    def _load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Return minimal default configuration
            return {
                "base": {
                    "colors": 15,
                    "simplification_level": "medium"
                },
                "image_types": {}
            }
    
    def _load_parameter_metadata(self):
        """Load parameter metadata from JSON file"""
        try:
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded parameter metadata from {self.metadata_path}")
                return metadata
            else:
                # Create default metadata
                metadata = self._create_default_parameter_metadata()
                # Save it for future use
                with open(self.metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"Created default parameter metadata at {self.metadata_path}")
                return metadata
        except Exception as e:
            logger.error(f"Error loading parameter metadata: {e}")
            return self._create_default_parameter_metadata()
            
    def _create_default_parameter_metadata(self):
        """Create default parameter metadata if no file exists"""
        # Basic metadata structure with categories
        metadata = {
            "categories": {
                "basic": {
                    "name": "Basic Settings",
                    "description": "Core settings that affect the overall result",
                    "order": 1
                },
                "preprocessing": {
                    "name": "Preprocessing",
                    "description": "Settings that affect how the image is processed before segmentation",
                    "order": 2
                },
                "feature_detection": {
                    "name": "Feature Detection",
                    "description": "Settings related to detecting and preserving important image features",
                    "order": 3
                },
                "color_control": {
                    "name": "Color Control",
                    "description": "Settings that affect color selection and processing",
                    "order": 4
                },
                "number_placement": {
                    "name": "Number Placement",
                    "description": "Settings that control how numbers are placed in regions",
                    "order": 5
                }
            },
            "parameters": {
                # Basic parameters
                "colors": {
                    "name": "Number of Colors",
                    "description": "The number of distinct colors to use in the paint-by-numbers template",
                    "long_description": "This setting controls how many different colors will be in the final template. More colors generally means more detail but also makes the template more complex to paint.",
                    "category": "basic",
                    "ui_control": "slider",
                    "visual_impact": 5,
                    "processing_impact": 3,
                    "complexity_impact": 5,
                    "examples": {
                        "low": "5-10 colors: Simple, impressionist style with less detail",
                        "medium": "15-20 colors: Balanced detail and complexity",
                        "high": "25-30 colors: High detail, more complex to paint"
                    }
                },
                "edge_strength": {
                    "name": "Edge Strength",
                    "description": "How strongly edges are detected and preserved",
                    "long_description": "Controls the sensitivity of edge detection. Higher values make edges more prominent, while lower values create softer transitions between regions.",
                    "category": "basic",
                    "ui_control": "slider",
                    "visual_impact": 4,
                    "processing_impact": 2,
                    "complexity_impact": 3
                },
                "edge_width": {
                    "name": "Edge Width",
                    "description": "Width of edges in pixels",
                    "long_description": "Determines how thick the edges appear in the template. Thicker edges are easier to see but may obscure small details.",
                    "category": "basic",
                    "ui_control": "slider",
                    "visual_impact": 3,
                    "processing_impact": 1,
                    "complexity_impact": 2
                },
                "simplification_level": {
                    "name": "Simplification Level",
                    "description": "How much to simplify the image",
                    "long_description": "Controls the level of detail in the template. Higher values create fewer, larger regions that are easier to paint but have less detail.",
                    "category": "basic",
                    "ui_control": "dropdown",
                    "visual_impact": 5,
                    "processing_impact": 4,
                    "complexity_impact": 5,
                    "options": {
                        "low": "High Detail - Preserves more intricate features",
                        "medium": "Medium Detail - Balanced approach",
                        "high": "Low Detail - Simplifies the image significantly",
                        "very_high": "Minimal Detail - Maximum simplification"
                    }
                },
                
                # Additional parameters metadata would be added in the same format
                # Example for a preprocessing parameter:
                "preprocessing_mode": {
                    "name": "Preprocessing Mode",
                    "description": "The approach used to prepare the image",
                    "long_description": "Determines how the image is preprocessed before color segmentation. Different modes are optimized for different types of images.",
                    "category": "preprocessing",
                    "ui_control": "dropdown",
                    "visual_impact": 4,
                    "processing_impact": 3,
                    "complexity_impact": 3,
                    "options": {
                        "standard": "Standard processing suitable for most images",
                        "enhanced": "Enhanced processing with better detail preservation",
                        "artistic": "Artistic processing with stylized effects"
                    }
                }
            }
        }
        
        # Add minimal metadata for all validation rules
        for param_name, validation in self.valid_values.items():
            if param_name not in metadata["parameters"]:
                # Extract category from parameter name if possible
                category = "basic"
                for cat in ["preprocessing", "feature_detection", "color_control", "number_placement"]:
                    if any(part in param_name for part in cat.split("_")):
                        category = cat
                        break
                
                # Add basic metadata
                metadata["parameters"][param_name] = {
                    "name": param_name.replace("_", " ").title(),
                    "description": f"Controls the {param_name.replace('_', ' ')}",
                    "category": category,
                    "ui_control": "slider" if validation["type"] in ["int", "float"] else "dropdown",
                    "visual_impact": 3,
                    "processing_impact": 2,
                    "complexity_impact": 2
                }
                
                # Add options for enum types
                if validation["type"] == "enum" and "values" in validation:
                    metadata["parameters"][param_name]["options"] = {
                        val: val.replace("_", " ").title() for val in validation["values"]
                    }
        
        return metadata

    def _initialize_validation_rules(self):
        """Define validation rules for parameters"""
        return {
            'colors': {'type': 'int', 'min': 5, 'max': 30},
            'edge_strength': {'type': 'float', 'min': 0.5, 'max': 3.0},
            'edge_width': {'type': 'int', 'min': 1, 'max': 5},
            'simplification_level': {'type': 'enum', 'values': ['low', 'medium', 'high', 'very_high']},
            'edge_style': {'type': 'enum', 'values': ['standard', 'soft', 'bold', 'thin', 'hierarchical']},
            'merge_regions_level': {'type': 'enum', 'values': ['low', 'normal', 'aggressive']},
            
            # Preprocessing parameters
            'preprocessing_mode': {'type': 'enum', 'values': ['standard', 'enhanced', 'artistic']},
            'sharpen_level': {'type': 'float', 'min': 0.0, 'max': 2.0},
            'contrast_boost': {'type': 'float', 'min': 0.0, 'max': 2.0},
            'noise_reduction_level': {'type': 'float', 'min': 0.0, 'max': 1.0},
            'detail_preservation': {'type': 'enum', 'values': ['low', 'medium', 'high', 'very_high']},
            
            # Feature detection parameters
            'feature_detection_enabled': {'type': 'bool'},
            'eye_detection_sensitivity': {'type': 'enum', 'values': ['low', 'medium', 'high']},
            'face_detection_mode': {'type': 'enum', 'values': ['basic', 'enhanced', 'ML']},
            'feature_importance': {'type': 'float', 'min': 0.0, 'max': 2.0},
            'preserve_expressions': {'type': 'bool'},
            'feature_protection_radius': {'type': 'int', 'min': 0, 'max': 20},
            
            # Color control parameters
            'color_harmony': {'type': 'enum', 'values': ['none', 'complementary', 'analogous', 'triadic', 'monochromatic']},
            'color_saturation_boost': {'type': 'float', 'min': -0.5, 'max': 1.5},
            'dark_area_enhancement': {'type': 'float', 'min': 0.0, 'max': 2.0},
            'light_area_protection': {'type': 'float', 'min': 0.0, 'max': 2.0},
            'color_grouping_threshold': {'type': 'float', 'min': 0.0, 'max': 1.0},
            'highlight_preservation': {'type': 'enum', 'values': ['low', 'medium', 'high', 'very_high']},
            
            # Number placement parameters
            'number_size_strategy': {'type': 'enum', 'values': ['uniform', 'proportional', 'adaptive']},
            'number_placement': {'type': 'enum', 'values': ['center', 'weighted', 'avoid_features']},
            'number_contrast': {'type': 'enum', 'values': ['low', 'medium', 'high', 'very_high']},
            'number_legibility_priority': {'type': 'float', 'min': 0.0, 'max': 1.0},
            'min_number_size': {'type': 'int', 'min': 8, 'max': 16},
            'number_overlap_strategy': {'type': 'enum', 'values': ['shrink', 'move', 'prioritize']}
        }
    
    def _initialize_parameter_dependencies(self):
        """Define parameter dependencies and relationships"""
        return {
            # Feature detection dependencies
            "feature_detection_enabled": {
                "controls": ["eye_detection_sensitivity", "face_detection_mode", "feature_importance",
                            "preserve_expressions", "feature_protection_radius"],
                "affects": ["number_placement"]
            },
            
            # Color harmony affects saturation
            "color_harmony": {
                "affects": ["color_saturation_boost"],
                "adjustments": {
                    "none": {},
                    "complementary": {"color_saturation_boost": "+0.2"},
                    "analogous": {"color_saturation_boost": "+0.1"},
                    "triadic": {"color_saturation_boost": "+0.3"},
                    "monochromatic": {"color_saturation_boost": "-0.2"}
                }
            },
            
            # Simplification affects other parameters
            "simplification_level": {
                "affects": ["edge_strength", "merge_regions_level"],
                "adjustments": {
                    "low": {"edge_strength": "-0.3", "merge_regions_level": "low"},
                    "medium": {},
                    "high": {"edge_strength": "+0.3", "merge_regions_level": "normal"},
                    "very_high": {"edge_strength": "+0.6", "merge_regions_level": "aggressive"}
                }
            },
            
            # Number placement strategy affects other number settings
            "number_size_strategy": {
                "affects": ["min_number_size"],
                "adjustments": {
                    "uniform": {},
                    "proportional": {},
                    "adaptive": {"min_number_size": "-1"}
                }
            }
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
        settings = self._deep_copy_dict(self.settings.get("base", {}))
        
        # Apply image type settings if available
        if image_type in self.settings.get("image_types", {}):
            image_config = self.settings["image_types"][image_type]
            
            # Apply base settings for this image type
            if "base" in image_config:
                self._deep_update_dict(settings, image_config["base"])
            
            # If no variant specified, use the recommended one
            if variant is None and "recommended" in image_config:
                variant = image_config["recommended"]
                logger.info(f"Using recommended variant '{variant}' for {image_type}")
            
            # Apply variant settings if specified and available
            if variant and "variants" in image_config and variant in image_config["variants"]:
                self._deep_update_dict(settings, image_config["variants"][variant])
                logger.info(f"Applied variant '{variant}' settings for {image_type}")
        
        # Check if there's a cross-cutting style variant to apply
        style_variants = self.settings.get("variant_styles", {})
        if custom_params and "style" in custom_params and custom_params["style"] in style_variants:
            style = custom_params["style"]
            self._deep_update_dict(settings, style_variants[style])
            logger.info(f"Applied style variant '{style}'")
            
        # Apply custom parameters last (highest priority)
        if custom_params:
            # Filter out None values and special keys like 'style'
            filtered_params = {k: v for k, v in custom_params.items() 
                              if v is not None and k != 'style'}
            
            # Handle any relative modifications ("+5", "-3")
            for key, value in filtered_params.items():
                if isinstance(value, str) and (value.startswith('+') or value.startswith('-')):
                    try:
                        if key in settings:
                            base_value = settings[key]
                            if isinstance(base_value, (int, float)):
                                delta = int(value)
                                filtered_params[key] = base_value + delta
                    except ValueError:
                        pass
                        
            # Apply the filtered parameters
            self._deep_update_dict(settings, filtered_params)
            logger.info(f"Applied custom parameters: {filtered_params}")
        
        # Apply parameter dependencies
        self._apply_parameter_dependencies(settings)
        
        # Validate and normalize all settings
        self._validate_settings(settings)
        
        return settings
    
    def _apply_parameter_dependencies(self, settings):
        """Apply parameter dependencies and relationships"""
        # Check feature detection dependencies
        if "feature_detection_enabled" in settings and not settings["feature_detection_enabled"]:
            # If feature detection is disabled, disable related settings
            if "eye_detection_sensitivity" in settings:
                settings["eye_detection_sensitivity"] = "low"
            if "face_detection_mode" in settings:
                settings["face_detection_mode"] = "basic"
            if "feature_importance" in settings:
                settings["feature_importance"] = 0.0
            if "preserve_expressions" in settings:
                settings["preserve_expressions"] = False
            if "feature_protection_radius" in settings:
                settings["feature_protection_radius"] = 0
                
        # Apply other parameter dependencies
        for param, dependency in self.parameter_dependencies.items():
            if param in settings and "adjustments" in dependency:
                value = settings[param]
                if value in dependency["adjustments"]:
                    # Apply the adjustments for this parameter value
                    adjustments = dependency["adjustments"][value]
                    for adj_param, adj_value in adjustments.items():
                        if adj_param in settings:
                            # If it's a relative adjustment
                            if isinstance(adj_value, str) and (adj_value.startswith('+') or adj_value.startswith('-')):
                                try:
                                    base_value = settings[adj_param]
                                    if isinstance(base_value, (int, float)):
                                        delta = float(adj_value)
                                        settings[adj_param] = base_value + delta
                                except (ValueError, TypeError):
                                    pass
                            else:
                                # Absolute value
                                settings[adj_param] = adj_value
    
    def _validate_settings(self, settings):
        """Recursively validate and normalize settings"""
        for key, value in list(settings.items()):
            if isinstance(value, dict):
                self._validate_settings(value)
            elif key in self.valid_values:
                validation = self.valid_values[key]
                
                if validation['type'] == 'int':
                    try:
                        settings[key] = int(value)
                        if 'min' in validation:
                            settings[key] = max(validation['min'], settings[key])
                        if 'max' in validation:
                            settings[key] = min(validation['max'], settings[key])
                    except (ValueError, TypeError):
                        settings[key] = validation.get('default', 0)
                        
                elif validation['type'] == 'float':
                    try:
                        settings[key] = float(value)
                        if 'min' in validation:
                            settings[key] = max(validation['min'], settings[key])
                        if 'max' in validation:
                            settings[key] = min(validation['max'], settings[key])
                    except (ValueError, TypeError):
                        settings[key] = validation.get('default', 0.0)
                        
                elif validation['type'] == 'enum':
                    if value not in validation['values']:
                        settings[key] = validation['values'][0]
                        
                elif validation['type'] == 'bool':
                    if not isinstance(value, bool):
                        settings[key] = bool(value)
    
    def _deep_copy_dict(self, source):
        """Create a deep copy of a dictionary"""
        if not isinstance(source, dict):
            return source
        result = {}
        for key, value in source.items():
            if isinstance(value, dict):
                result[key] = self._deep_copy_dict(value)
            else:
                result[key] = value
        return result
    
    def _deep_update_dict(self, target, source):
        """Update target dict with source dict recursively"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update_dict(target[key], value)
            else:
                target[key] = value
    
    # ====== STAGE 2 ENHANCEMENTS: PARAMETER DOCUMENTATION ======
    
    def get_parameter_info(self, param_name):
        """
        Get detailed information about a parameter
        
        Args:
            param_name: The name of the parameter
            
        Returns:
            Dictionary with parameter information
        """
        # Get metadata for the parameter
        if param_name in self.parameter_metadata["parameters"]:
            return self.parameter_metadata["parameters"][param_name]
        else:
            # Create basic metadata for unknown parameters
            return {
                "name": param_name.replace("_", " ").title(),
                "description": f"Controls the {param_name.replace('_', ' ')}",
                "category": "basic",
                "visual_impact": 3,
                "processing_impact": 2,
                "complexity_impact": 2
            }
    
    def get_parameter_categories(self):
        """
        Get all parameter categories with their metadata
        
        Returns:
            Dictionary of categories
        """
        return self.parameter_metadata["categories"]
    
    def get_parameters_by_category(self, category=None):
        """
        Get parameters grouped by category or for a specific category
        
        Args:
            category: Optional category name to filter by
            
        Returns:
            Dictionary of parameters grouped by category
        """
        params = {}
        
        # If category specified, only get parameters for that category
        if category:
            params[category] = []
            for param_name, metadata in self.parameter_metadata["parameters"].items():
                if metadata.get("category") == category:
                    params[category].append({
                        "name": param_name,
                        "metadata": metadata
                    })
            return params
        
        # Group all parameters by category
        for param_name, metadata in self.parameter_metadata["parameters"].items():
            cat = metadata.get("category", "basic")
            if cat not in params:
                params[cat] = []
                
            params[cat].append({
                "name": param_name,
                "metadata": metadata
            })
            
        # Sort categories by their order
        sorted_params = {}
        categories = self.parameter_metadata["categories"]
        for cat in sorted(categories.keys(), key=lambda c: categories[c].get("order", 99)):
            if cat in params:
                sorted_params[cat] = params[cat]
        
        # Add any categories not in the original ordering
        for cat in params:
            if cat not in sorted_params:
                sorted_params[cat] = params[cat]
                
        return sorted_params
    
    def get_parameter_ui_info(self, param_name):
        """
        Get UI-specific information for a parameter
        
        Args:
            param_name: The name of the parameter
            
        Returns:
            Dictionary with UI information
        """
        # Get parameter metadata
        metadata = self.get_parameter_info(param_name)
        
        # Get validation rules
        validation = self.valid_values.get(param_name, {})
        
        # Extract UI-specific information
        ui_info = {
            "name": metadata.get("name", param_name.replace("_", " ").title()),
            "description": metadata.get("description", ""),
            "tooltip": metadata.get("long_description", metadata.get("description", "")),
            "control_type": metadata.get("ui_control", "text"),
        }
        
        # Add control-specific properties
        if validation.get("type") == "int" or validation.get("type") == "float":
            ui_info["min"] = validation.get("min")
            ui_info["max"] = validation.get("max")
            ui_info["step"] = 1 if validation.get("type") == "int" else 0.1
            
        elif validation.get("type") == "enum":
            # Use metadata options if available, otherwise use validation values
            if "options" in metadata:
                ui_info["options"] = metadata["options"]
            elif "values" in validation:
                ui_info["options"] = {v: v.replace("_", " ").title() for v in validation["values"]}
                
        elif validation.get("type") == "bool":
            ui_info["control_type"] = "checkbox"
            
        return ui_info
    
    # ====== STAGE 2 ENHANCEMENTS: PARAMETER IMPACT ANALYSIS ======
    
    def get_parameter_impact(self, param_name):
        """
        Get impact ratings for a parameter
        
        Args:
            param_name: The name of the parameter
            
        Returns:
            Dictionary with impact ratings
        """
        metadata = self.get_parameter_info(param_name)
        
        return {
            "visual_impact": metadata.get("visual_impact", 3),
            "processing_impact": metadata.get("processing_impact", 2),
            "complexity_impact": metadata.get("complexity_impact", 2)
        }
    
    def get_high_impact_parameters(self, impact_type="visual", threshold=4):
        """
        Get parameters with high impact of a specific type
        
        Args:
            impact_type: Type of impact to filter by ('visual', 'processing', 'complexity')
            threshold: Minimum impact value to include (1-5)
            
        Returns:
            List of high-impact parameters
        """
        impact_key = f"{impact_type}_impact"
        high_impact = []
        
        for param_name, metadata in self.parameter_metadata["parameters"].items():
            if metadata.get(impact_key, 0) >= threshold:
                high_impact.append({
                    "name": param_name,
                    "impact": metadata.get(impact_key, 0),
                    "metadata": metadata
                })
                
        # Sort by impact (highest first)
        high_impact.sort(key=lambda p: p["impact"], reverse=True)
        return high_impact
    
    def analyze_settings_complexity(self, settings):
        """
        Analyze the complexity of a settings combination
        
        Args:
            settings: Dictionary of settings to analyze
            
        Returns:
            Dictionary with complexity analysis
        """
        total_score = 0
        max_score = 0
        key_factors = []
        
        for param_name, value in settings.items():
            metadata = self.get_parameter_info(param_name)
            impact = metadata.get("complexity_impact", 0)
            max_score += impact * 5  # Maximum possible score
            
            # Calculate relative score based on parameter value
            if param_name == "colors":
                # More colors = higher complexity
                relative_value = (value - 5) / 25  # Scale 5-30 to 0-1
                score = impact * (1 + 4 * relative_value)  # Scale to 1-5 based on impact
                total_score += score
                
                if value > 20:
                    key_factors.append(f"High color count ({value})")
                    
            elif param_name == "simplification_level":
                # Map values to numerical scores
                level_scores = {"low": 5, "medium": 3, "high": 2, "very_high": 1}
                score = impact * level_scores.get(value, 3) / 5  # Normalize to impact
                total_score += score
                
                if value == "low":
                    key_factors.append("High detail level")
                    
            else:
                # Default scoring based on impact only
                total_score += impact
        
        # Calculate overall complexity percentage
        complexity_pct = (total_score / max_score) * 100 if max_score > 0 else 50
        
        # Determine complexity level
        if complexity_pct >= 80:
            complexity_level = "Very Complex"
        elif complexity_pct >= 60:
            complexity_level = "Complex"
        elif complexity_pct >= 40:
            complexity_level = "Moderate"
        elif complexity_pct >= 20:
            complexity_level = "Simple"
        else:
            complexity_level = "Very Simple"
            
        return {
            "complexity_score": round(complexity_pct, 1),
            "complexity_level": complexity_level,
            "key_factors": key_factors
        }
    
    # ====== STAGE 2 ENHANCEMENTS: PARAMETER DEPENDENCIES ======
    
    def check_parameter_conflicts(self, settings):
        """
        Check for conflicts between parameter values
        
        Args:
            settings: Dictionary of settings to check
            
        Returns:
            List of conflict warnings
        """
        warnings = []
        
        # Check for high complexity with high colors
        if settings.get("simplification_level") == "low" and settings.get("colors", 0) > 25:
            warnings.append({
                "level": "warning",
                "message": "High detail level with many colors will create a very complex template",
                "parameters": ["simplification_level", "colors"]
            })
            
        # Check for feature detection without eye detection
        if (settings.get("feature_detection_enabled") and 
            settings.get("eye_detection_sensitivity") == "low" and 
            settings.get("image_type") == "pet"):
            warnings.append({
                "level": "info",
                "message": "Consider increasing eye detection sensitivity for pet images",
                "parameters": ["eye_detection_sensitivity"]
            })
            
        # Check for ineffective parameters
        if not settings.get("feature_detection_enabled") and settings.get("feature_importance", 0) > 0:
            warnings.append({
                "level": "info",
                "message": "Feature importance has no effect when feature detection is disabled",
                "parameters": ["feature_importance", "feature_detection_enabled"]
            })
            
        return warnings
    
    def get_parameter_dependencies(self, param_name):
        """
        Get dependencies for a specific parameter
        
        Args:
            param_name: Name of the parameter
            
        Returns:
            Dictionary with parameter dependencies
        """
        if param_name in self.parameter_dependencies:
            return self.parameter_dependencies[param_name]
        return {}
    
    # ====== STAGE 2 ENHANCEMENTS: PARAMETER PRESETS ======
    
    def save_preset(self, preset_name, settings, description="", tags=None):
        """
        Save current settings as a named preset
        
        Args:
            preset_name: Name of the preset
            settings: Dictionary of settings to save
            description: Optional description of the preset
            tags: Optional list of tags for categorization
            
        Returns:
            Path to the saved preset file
        """
        if tags is None:
            tags = []
            
        # Create preset object
        preset = {
            "name": preset_name,
            "description": description,
            "tags": tags,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "settings": settings
        }
        
        # Create safe filename
        safe_name = preset_name.lower().replace(" ", "_")
        filename = f"{safe_name}_{int(time.time())}.json"
        preset_path = os.path.join(self.presets_dir, filename)
        
        # Save preset to file
        with open(preset_path, 'w') as f:
            json.dump(preset, f, indent=2)
            
        logger.info(f"Saved preset '{preset_name}' to {preset_path}")
        return preset_path
    
    def load_preset(self, preset_path):
        """
        Load settings from a preset file
        
        Args:
            preset_path: Path to the preset file
            
        Returns:
            Dictionary with preset data and settings
        """
        try:
            with open(preset_path, 'r') as f:
                preset = json.load(f)
                logger.info(f"Loaded preset from {preset_path}")
                return preset
        except Exception as e:
            logger.error(f"Error loading preset: {e}")
            return None
    
    def get_available_presets(self, tag=None):
        """
        Get list of available presets
        
        Args:
            tag: Optional tag to filter presets
            
        Returns:
            List of preset metadata
        """
        presets = []
        
        # Look for all JSON files in the presets directory
        if os.path.exists(self.presets_dir):
            for filename in os.listdir(self.presets_dir):
                if filename.endswith(".json"):
                    try:
                        preset_path = os.path.join(self.presets_dir, filename)
                        with open(preset_path, 'r') as f:
                            preset = json.load(f)
                            
                        # Filter by tag if specified
                        if tag is None or tag in preset.get("tags", []):
                            presets.append({
                                "name": preset.get("name", filename),
                                "description": preset.get("description", ""),
                                "tags": preset.get("tags", []),
                                "created": preset.get("created", ""),
                                "path": preset_path
                            })
                    except Exception as e:
                        logger.warning(f"Could not load preset {filename}: {e}")
        
        # Sort by newest first
        presets.sort(key=lambda p: p.get("created", ""), reverse=True)
        return presets
    
    def compare_settings(self, settings1, settings2):
        """
        Compare two settings objects and highlight differences
        
        Args:
            settings1: First settings dictionary
            settings2: Second settings dictionary
            
        Returns:
            Dictionary with differences
        """
        differences = {}
        
        # Collect all keys from both settings
        all_keys = set(settings1.keys()) | set(settings2.keys())
        
        for key in all_keys:
            # Handle nested dictionaries
            if (key in settings1 and key in settings2 and
                isinstance(settings1[key], dict) and isinstance(settings2[key], dict)):
                nested_diff = self.compare_settings(settings1[key], settings2[key])
                if nested_diff:
                    differences[key] = nested_diff
            # Handle different values or missing keys
            elif key not in settings1:
                differences[key] = {
                    "only_in_second": settings2[key]
                }
            elif key not in settings2:
                differences[key] = {
                    "only_in_first": settings1[key]
                }
            elif settings1[key] != settings2[key]:
                differences[key] = {
                    "first": settings1[key],
                    "second": settings2[key]
                }
                
                # Add metadata for this parameter if available
                if key in self.parameter_metadata["parameters"]:
                    differences[key]["metadata"] = self.parameter_metadata["parameters"][key]
        
        return differences
    
    # Legacy methods for backward compatibility
    
    def get_available_variants(self, image_type):
        """Get available variants for an image type"""
        if image_type in self.settings.get("image_types", {}):
            image_config = self.settings["image_types"][image_type]
            if "variants" in image_config:
                return {name: data.get("description", "") for name, data in image_config["variants"].items()}
        return {}
    
    def get_style_variants(self):
        """Get available style variants"""
        return {name: data.get("description", "") for name, data in self.settings.get("variant_styles", {}).items()}
    
    def get_recommended_variant(self, image_type):
        """Get recommended variant for an image type"""
        if image_type in self.settings.get("image_types", {}):
            image_config = self.settings["image_types"][image_type]
            if "recommended" in image_config:
                return image_config["recommended"]
        return "standard"