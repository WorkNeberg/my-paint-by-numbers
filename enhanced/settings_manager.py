class SettingsManager:
    def __init__(self):
        self.presets = self._initialize_presets()
        
    def _initialize_presets(self):
        """Initialize all presets for different image types and complexity levels"""
        presets = {
            # PORTRAIT PRESETS - Focus on facial features
            'portrait': {
                'low': {  # Simplest, fewest regions
                    'num_colors': 8,
                    'simplification_level': 'high',
                    'edge_strength': 0.7,
                    'edge_width': 1,
                    'enhance_dark_areas': True,
                    'dark_threshold': 50,
                    'edge_style': 'soft',
                    'merge_regions_level': 'aggressive',
                    'min_region_percent': 0.8,
                    'color_similarity_threshold': 50,
                    'preserve_features': True,
                    'feature_detail_boost': 1.2
                },
                'medium': {  # Balanced detail
                    'num_colors': 12,
                    'simplification_level': 'medium',
                    'edge_strength': 0.8,
                    'edge_width': 1,
                    'enhance_dark_areas': True,
                    'dark_threshold': 50,
                    'edge_style': 'soft',
                    'merge_regions_level': 'normal',
                    'min_region_percent': 0.5,
                    'color_similarity_threshold': 40,
                    'preserve_features': True,
                    'feature_detail_boost': 1.5
                },
                'high': {  # Maximum detail
                    'num_colors': 15,
                    'simplification_level': 'low',
                    'edge_strength': 0.9,
                    'edge_width': 1,
                    'enhance_dark_areas': True,
                    'dark_threshold': 50,
                    'edge_style': 'soft',
                    'merge_regions_level': 'low',
                    'min_region_percent': 0.3,
                    'color_similarity_threshold': 30,
                    'preserve_features': True,
                    'feature_detail_boost': 2.0
                },
                'davinci': {  # Leonardo da Vinci style - subtle shading
                    'num_colors': 20,
                    'simplification_level': 'low',
                    'edge_strength': 0.6,
                    'edge_width': 1,
                    'enhance_dark_areas': True,
                    'dark_threshold': 60,
                    'edge_style': 'thin',
                    'merge_regions_level': 'normal',
                    'min_region_percent': 0.4,
                    'color_similarity_threshold': 25,
                    'preserve_features': True,
                    'feature_detail_boost': 2.5,
                    'style_name': 'Leonardo da Vinci'
                }
            },
            
            # PET PRESETS - Handle fur and facial features
            'pet': {
                'low': {
                    'num_colors': 10,
                    'simplification_level': 'high',
                    'edge_strength': 0.7,
                    'edge_width': 1,
                    'enhance_dark_areas': True,
                    'dark_threshold': 60,
                    'edge_style': 'soft',
                    'merge_regions_level': 'aggressive',
                    'min_region_percent': 1.0,
                    'color_similarity_threshold': 60,
                    'preserve_features': True,
                    'feature_detail_boost': 1.5
                },
                'medium': {
                    'num_colors': 12,
                    'simplification_level': 'medium',
                    'edge_strength': 0.7,
                    'edge_width': 1,
                    'enhance_dark_areas': True,
                    'dark_threshold': 60,
                    'edge_style': 'soft',
                    'merge_regions_level': 'aggressive',
                    'min_region_percent': 0.7,
                    'color_similarity_threshold': 50,
                    'preserve_features': True,
                    'feature_detail_boost': 1.8
                },
                'high': {
                    'num_colors': 15,
                    'simplification_level': 'medium',
                    'edge_strength': 0.8,
                    'edge_width': 1,
                    'enhance_dark_areas': True,
                    'dark_threshold': 60,
                    'edge_style': 'soft',
                    'merge_regions_level': 'normal',
                    'min_region_percent': 0.5,
                    'color_similarity_threshold': 40,
                    'preserve_features': True,
                    'feature_detail_boost': 2.0
                },
                'detailed_fur': {  # Special preset for fluffy pets
                    'num_colors': 18,
                    'simplification_level': 'medium',
                    'edge_strength': 0.7,
                    'edge_width': 1,
                    'enhance_dark_areas': True,
                    'dark_threshold': 50,
                    'edge_style': 'soft',
                    'merge_regions_level': 'low',
                    'min_region_percent': 0.3,
                    'color_similarity_threshold': 25,
                    'preserve_features': True,
                    'feature_detail_boost': 2.0,
                    'style_name': 'Detailed Fur'
                }
            },
            
            # LANDSCAPE PRESETS
            'landscape': {
                'low': {
                    'num_colors': 12,
                    'simplification_level': 'high',
                    'edge_strength': 1.0,
                    'edge_width': 2,
                    'enhance_dark_areas': False,
                    'dark_threshold': 50,
                    'edge_style': 'normal',
                    'merge_regions_level': 'aggressive',
                    'min_region_percent': 1.0,
                    'color_similarity_threshold': 60,
                    'preserve_features': False,
                    'feature_detail_boost': 1.0
                },
                'medium': {
                    'num_colors': 18,
                    'simplification_level': 'medium',
                    'edge_strength': 1.0,
                    'edge_width': 1,
                    'enhance_dark_areas': False,
                    'dark_threshold': 50,
                    'edge_style': 'normal',
                    'merge_regions_level': 'normal',
                    'min_region_percent': 0.5,
                    'color_similarity_threshold': 40,
                    'preserve_features': False,
                    'feature_detail_boost': 1.0
                },
                'high': {
                    'num_colors': 24,
                    'simplification_level': 'low',
                    'edge_strength': 1.0,
                    'edge_width': 1,
                    'enhance_dark_areas': False,
                    'dark_threshold': 50,
                    'edge_style': 'normal',
                    'merge_regions_level': 'low',
                    'min_region_percent': 0.3,
                    'color_similarity_threshold': 30,
                    'preserve_features': False,
                    'feature_detail_boost': 1.0
                },
                'impressionist': {  # Monet-like style
                    'num_colors': 20,
                    'simplification_level': 'medium',
                    'edge_strength': 0.8,
                    'edge_width': 1,
                    'enhance_dark_areas': False,
                    'dark_threshold': 50,
                    'edge_style': 'soft',
                    'merge_regions_level': 'normal',
                    'min_region_percent': 0.4,
                    'color_similarity_threshold': 35,
                    'preserve_features': False,
                    'feature_detail_boost': 1.0,
                    'style_name': 'Impressionist'
                }
            },
            
            # CARTOON PRESETS
            'cartoon': {
                'low': {
                    'num_colors': 8,
                    'simplification_level': 'high',
                    'edge_strength': 1.2,
                    'edge_width': 2,
                    'enhance_dark_areas': False,
                    'dark_threshold': 50,
                    'edge_style': 'normal',
                    'merge_regions_level': 'aggressive',
                    'min_region_percent': 1.0,
                    'color_similarity_threshold': 60,
                    'preserve_features': False,
                    'feature_detail_boost': 1.0
                },
                'medium': {
                    'num_colors': 10,
                    'simplification_level': 'high',
                    'edge_strength': 1.2,
                    'edge_width': 2,
                    'enhance_dark_areas': False,
                    'dark_threshold': 50,
                    'edge_style': 'normal',
                    'merge_regions_level': 'normal',
                    'min_region_percent': 0.7,
                    'color_similarity_threshold': 50,
                    'preserve_features': False,
                    'feature_detail_boost': 1.0
                },
                'high': {
                    'num_colors': 12,
                    'simplification_level': 'medium',
                    'edge_strength': 1.2,
                    'edge_width': 2,
                    'enhance_dark_areas': False,
                    'dark_threshold': 50,
                    'edge_style': 'normal',
                    'merge_regions_level': 'normal',
                    'min_region_percent': 0.5,
                    'color_similarity_threshold': 40,
                    'preserve_features': False,
                    'feature_detail_boost': 1.0
                },
                'comic': {  # Comic book style
                    'num_colors': 8,
                    'simplification_level': 'high',
                    'edge_strength': 1.5,
                    'edge_width': 2,
                    'enhance_dark_areas': False,
                    'dark_threshold': 50,
                    'edge_style': 'normal',
                    'merge_regions_level': 'aggressive',
                    'min_region_percent': 0.8,
                    'color_similarity_threshold': 50,
                    'preserve_features': False,
                    'feature_detail_boost': 1.0,
                    'style_name': 'Comic Book'
                }
            },
            
            # STILL LIFE PRESETS
            'still_life': {
                'low': {
                    'num_colors': 12,
                    'simplification_level': 'high',
                    'edge_strength': 1.0,
                    'edge_width': 1,
                    'enhance_dark_areas': True,
                    'dark_threshold': 50,
                    'edge_style': 'normal',
                    'merge_regions_level': 'normal',
                    'min_region_percent': 0.7,
                    'color_similarity_threshold': 50,
                    'preserve_features': True,
                    'feature_detail_boost': 1.2
                },
                'medium': {
                    'num_colors': 16,
                    'simplification_level': 'medium',
                    'edge_strength': 1.0,
                    'edge_width': 1,
                    'enhance_dark_areas': True,
                    'dark_threshold': 50,
                    'edge_style': 'normal',
                    'merge_regions_level': 'normal',
                    'min_region_percent': 0.5,
                    'color_similarity_threshold': 40,
                    'preserve_features': True,
                    'feature_detail_boost': 1.5
                },
                'high': {
                    'num_colors': 20,
                    'simplification_level': 'low',
                    'edge_strength': 1.0,
                    'edge_width': 1,
                    'enhance_dark_areas': True,
                    'dark_threshold': 50,
                    'edge_style': 'normal',
                    'merge_regions_level': 'low',
                    'min_region_percent': 0.3,
                    'color_similarity_threshold': 30,
                    'preserve_features': True,
                    'feature_detail_boost': 1.8
                },
                'old_master': {  # Dutch Golden Age still life style
                    'num_colors': 18,
                    'simplification_level': 'low',
                    'edge_strength': 0.9,
                    'edge_width': 1,
                    'enhance_dark_areas': True,
                    'dark_threshold': 60,
                    'edge_style': 'soft',
                    'merge_regions_level': 'low',
                    'min_region_percent': 0.4,
                    'color_similarity_threshold': 35,
                    'preserve_features': True,
                    'feature_detail_boost': 2.0,
                    'style_name': 'Dutch Master'
                }
            },
            
            # DEFAULT/GENERAL PRESET
            'general': {
                'low': {
                    'num_colors': 10,
                    'simplification_level': 'high',
                    'edge_strength': 1.0,
                    'edge_width': 1,
                    'enhance_dark_areas': False,
                    'dark_threshold': 50,
                    'edge_style': 'normal',
                    'merge_regions_level': 'aggressive',
                    'min_region_percent': 0.8,
                    'color_similarity_threshold': 50,
                    'preserve_features': False,
                    'feature_'
                    'feature_detail_boost': 1.0
                },
                'medium': {
                    'num_colors': 15,
                    'simplification_level': 'medium',
                    'edge_strength': 1.0,
                    'edge_width': 1,
                    'enhance_dark_areas': False,
                    'dark_threshold': 50,
                    'edge_style': 'normal',
                    'merge_regions_level': 'normal',
                    'min_region_percent': 0.5,
                    'color_similarity_threshold': 40,
                    'preserve_features': False,
                    'feature_detail_boost': 1.0
                },
                'high': {
                    'num_colors': 20,
                    'simplification_level': 'low',
                    'edge_strength': 1.0,
                    'edge_width': 1,
                    'enhance_dark_areas': False,
                    'dark_threshold': 50,
                    'edge_style': 'normal',
                    'merge_regions_level': 'low',
                    'min_region_percent': 0.3,
                    'color_similarity_threshold': 30,
                    'preserve_features': False,
                    'feature_detail_boost': 1.0
                },
                'classic': {
                    'num_colors': 15,
                    'simplification_level': 'medium',
                    'edge_strength': 1.0,
                    'edge_width': 1,
                    'enhance_dark_areas': False,
                    'dark_threshold': 50,
                    'edge_style': 'normal',
                    'merge_regions_level': 'normal',
                    'min_region_percent': 0.5,
                    'color_similarity_threshold': 40,
                    'preserve_features': False,
                    'feature_detail_boost': 1.0,
                    'style_name': 'Classic'
                }
            }
        }
        
        return presets
        
    def get_settings(self, image_type, complexity_level='medium', style=None):
        """
        Get settings for a specific image type and complexity level
        
        Parameters:
        - image_type: Type of image ('portrait', 'pet', etc.)
        - complexity_level: Level of complexity ('low', 'medium', 'high')
        - style: Optional specific style (e.g., 'davinci', 'impressionist')
        
        Returns:
        - settings: Dictionary of processing settings
        """
        # Default to general if image type not found
        if image_type not in self.presets:
            image_type = 'general'
            
        presets_for_type = self.presets[image_type]
        
        # If style is specified and exists, use it
        if style and style in presets_for_type:
            return presets_for_type[style]
            
        # Otherwise use the complexity level if it exists
        if complexity_level in presets_for_type:
            return presets_for_type[complexity_level]
            
        # Fall back to medium if specified level doesn't exist
        return presets_for_type.get('medium', presets_for_type.get(list(presets_for_type.keys())[0]))
        
    def get_available_styles(self, image_type=None):
        """
        Get all available styles for a specific image type or all image types
        
        Parameters:
        - image_type: Optional type of image to limit results
        
        Returns:
        - styles: Dictionary of available styles by image type
        """
        styles = {}
        
        if image_type and image_type in self.presets:
            # Get styles for specific image type
            for style_name, settings in self.presets[image_type].items():
                if 'style_name' in settings:
                    styles[style_name] = {
                        'name': settings['style_name'],
                        'image_type': image_type
                    }
        else:
            # Get all styles across all image types
            for img_type, presets in self.presets.items():
                styles[img_type] = {}
                for style_name, settings in presets.items():
                    if 'style_name' in settings:
                        styles[img_type][style_name] = settings['style_name']
                        
        return styles