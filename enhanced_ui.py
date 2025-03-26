from werkzeug.utils import secure_filename
import os
import time
import uuid
import cv2
import numpy as np
from enhanced.integration import enhanced_integration
import logging
import json
# Add this at the top with the other imports
from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for, send_from_directory, current_app

# Setup logging
logger = logging.getLogger('pbn-app.enhanced_ui')

# Create blueprint
enhanced_bp = Blueprint('enhanced', __name__, url_prefix='/enhanced')

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
PREVIEW_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'previews')
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREVIEW_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Add this helper function at the top of your file with other imports
import numpy as np

def ensure_correct_format(image):
    """Ensure image is in correct format for OpenCV operations"""
    if image is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Check data type and convert if necessary
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            # Normalize float values to 0-255 range
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        else:
            # For other types, just convert
            image = image.astype(np.uint8)
    
    return image

# Routes
@enhanced_bp.route('/')
def index():
    """Enhanced main page"""
    return render_template('enhanced/index.html')

@enhanced_bp.route('/upload', methods=['POST'])
def upload():
    """Handle image upload"""
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if file and allowed_file(file.filename):
        # Create unique filename
        timestamp = int(time.time())
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Store in session for later use
        session['uploaded_image'] = {
            'path': file_path,
            'filename': filename,
            'timestamp': timestamp
        }
        
        # Detect image type
        image_type = enhanced_integration.image_type_detector.detect_image_type(file_path)
        
        # Return success with image path and type
        return jsonify({
            'success': True,
            'image': f"/uploads/{filename}",
            'image_type': image_type
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@enhanced_bp.route('/editor')
def editor():
    """Enhanced editor page"""
    # Check if we have an uploaded image
    if 'uploaded_image' not in session:
        return redirect(url_for('enhanced.index'))
        
    # Get image data
    image_data = session['uploaded_image']
    
    # Get parameter metadata
    parameter_metadata = enhanced_integration.get_parameter_metadata()
    
    # Get available presets
    presets = enhanced_integration.get_available_presets()
    
    return render_template('enhanced/editor.html', 
                          image=image_data,
                          parameter_metadata=parameter_metadata,
                          presets=presets)

@enhanced_bp.route('/api/preview', methods=['POST'])
def generate_preview():
    """API endpoint to generate previews with current settings"""
    # Get image path from session
    if 'uploaded_image' not in session:
        return jsonify({'error': 'No image uploaded'}), 400
        
    image_data = session['uploaded_image']
    image_path = image_data['path']
    
    # Get settings from request
    data = request.json
    settings = data.get('settings', {})
    preview_type = data.get('preview_type', 'template')
    
    # Generate preview
    preview_result = enhanced_integration.generate_preview(image_path, settings, preview_type)
    
    if preview_result is None:
        return jsonify({'error': 'Failed to generate preview'}), 500
        
    # Save preview image
    preview_filename = f"preview_{image_data['timestamp']}_{preview_type}.jpg"
    preview_path = os.path.join(PREVIEW_FOLDER, preview_filename)
    
    # Add safety check before color conversion
    if preview_result and 'preview' in preview_result and preview_result['preview'] is not None:
        # Fix for image type conversion
        preview_img = preview_result['preview']
        
        # Convert image to appropriate format before color conversion
        if preview_img.dtype != np.uint8:
            if np.issubdtype(preview_img.dtype, np.integer):
                # For integer types (including CV_32S)
                preview_img = np.clip(preview_img, 0, 255).astype(np.uint8)
            elif np.issubdtype(preview_img.dtype, np.floating):
                # For float types
                preview_img = np.clip(preview_img * 255, 0, 255).astype(np.uint8)
        
        # Make sure it's 3-channel for color conversion
        if len(preview_img.shape) == 2:
            # Convert grayscale to RGB
            preview_image = cv2.cvtColor(preview_img, cv2.COLOR_GRAY2BGR)
        else:
            # Convert RGB to BGR
            preview_image = cv2.cvtColor(preview_img, cv2.COLOR_RGB2BGR)
    else:
        # Create a blank image if no preview is available
        preview_image = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(preview_image, "No preview available", (50, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite(preview_path, preview_image)
    
    # Return preview info
    return jsonify({
        'success': True,
        'preview_url': f"/static/previews/{preview_filename}",
        'preview_type': preview_type
    })

@enhanced_bp.route('/api/process', methods=['POST'])
def process_image():
    try:
        # Get settings from request
        settings = request.json or {}
        
        # Extract stay_on_page parameter
        stay_on_page = False
        if 'stay_on_page' in settings:
            stay_on_page = settings.pop('stay_on_page', False)
        
        # Fix edge parameters if needed
        if 'edge_width' in settings and int(settings['edge_width']) > 5:
            settings['edge_width'] = 3
        
        # Get image path from session
        if 'uploaded_image' not in session:
            return jsonify({'error': 'No image uploaded'}), 400
            
        image_path = session['uploaded_image']['path']
        timestamp = os.path.basename(image_path).split('_')[0]
        
        # Process the image
        result = enhanced_integration.process_image(image_path, settings)
        
        # Generate previews for all stages if staying on page
        if stay_on_page:
            preview_types = ['preprocessed', 'segments', 'edges', 'template']
            for preview_type in preview_types:
                try:
                    # Generate each preview type
                    preview_result = enhanced_integration.generate_preview(image_path, settings, preview_type)
                    if preview_result and 'preview' in preview_result:
                        # Get the preview image
                        preview_img = preview_result['preview']
                        
                        # Convert to proper format for saving
                        if preview_img.dtype != np.uint8:
                            preview_img = (preview_img * 255).astype(np.uint8)
                        
                        # Make sure it's BGR for OpenCV saving
                        if len(preview_img.shape) == 2:  # Grayscale
                            preview_img = cv2.cvtColor(preview_img, cv2.COLOR_GRAY2BGR)
                        elif preview_img.shape[2] == 3:  # RGB
                            preview_img = cv2.cvtColor(preview_img, cv2.COLOR_RGB2BGR)
                        
                        # Create previews directory if it doesn't exist
                        preview_dir = os.path.join(current_app.static_folder, 'previews')
                        os.makedirs(preview_dir, exist_ok=True)
                        
                        # Save preview image
                        preview_path = os.path.join(preview_dir, f"preview_{timestamp}_{preview_type}.jpg")
                        cv2.imwrite(preview_path, preview_img)
                except Exception as e:
                    logger.error(f"Error generating {preview_type} preview: {str(e)}")
        
        # Store settings in session for later use
        session['processing_settings'] = settings
        
        # Return based on stay_on_page parameter
        if stay_on_page:
            return jsonify({
                'success': True,
                'message': 'Processing complete'
            })
        else:
            # Original behavior - redirect
            return jsonify({
                'success': True,
                'redirect': '/enhanced/results'
            })
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        return jsonify({
            'error': 'Processing error',
            'details': str(e)
        }), 500

@enhanced_bp.route('/api/analyze', methods=['POST'])
def analyze_settings():
    """API endpoint to analyze the complexity of current settings"""
    # Get settings from request
    data = request.json
    settings = data.get('settings', {})
    
    # Analyze settings
    analysis = enhanced_integration.analyze_settings(settings)
    
    return jsonify({
        'success': True,
        'analysis': analysis
    })

@enhanced_bp.route('/api/presets/save', methods=['POST'])
def save_preset():
    """API endpoint to save current settings as a preset"""
    data = request.json
    name = data.get('name', '')
    description = data.get('description', '')
    settings = data.get('settings', {})
    
    if not name:
        return jsonify({'error': 'Preset name is required'}), 400
        
    try:
        preset_path = enhanced_integration.save_preset(name, settings, description)
        return jsonify({
            'success': True,
            'message': f"Preset '{name}' saved successfully",
            'preset_path': preset_path
        })
    except Exception as e:
        logger.error(f"Error saving preset: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@enhanced_bp.route('/api/presets/load/<preset_name>', methods=['GET'])
def load_preset(preset_name):
    """API endpoint to load a preset"""
    try:
        settings = enhanced_integration.load_preset(preset_name)
        return jsonify({
            'success': True,
            'settings': settings
        })
    except Exception as e:
        logger.error(f"Error loading preset: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
        
@enhanced_bp.route('/results')
def results():
    """Display the results page for the processed image"""
    if 'processing_results' not in session:
        return redirect(url_for('enhanced_bp.home'))
        
    raw_results = session['processing_results']
    
    # Structure results correctly for template
    results = {
        'files': {
            'template': raw_results.get('template', ''),
            'template_with_numbers': raw_results.get('template_with_numbers', ''),
            'color_reference': raw_results.get('color_reference', '')
        },
        'timestamp': raw_results.get('timestamp', int(time.time())),
        'colors': raw_results.get('colors', 0),
        'processing_time': raw_results.get('processing_time', 0.0),
        'settings': raw_results.get('settings', {})
    }
    
    return render_template('enhanced/results.html', results=results)

# Serve files from upload folder
@enhanced_bp.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serve files from the upload directory"""
    return send_from_directory(UPLOAD_FOLDER, filename)
    
# Serve files from output folder
@enhanced_bp.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

# API endpoint to get parameter metadata
@enhanced_bp.route('/api/parameters', methods=['GET'])
def get_parameters():
    """API endpoint to get parameter metadata"""
    try:
        metadata = enhanced_integration.get_parameter_metadata()
        return jsonify({
            'success': True,
            'metadata': metadata
        })
    except Exception as e:
        logger.error(f"Error getting parameter metadata: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@enhanced_bp.route('/api/process_all_previews', methods=['POST'])
def process_all_previews():
    """Process and generate all preview types at once"""
    if 'uploaded_image' not in session:
        return jsonify({'error': 'No image to process'}), 400
    
    try:
        # Get image path and settings
        image_path = session['uploaded_image']['path']
        timestamp = session['uploaded_image']['timestamp']
        
        # Get settings from request
        data = request.json or {}
        settings = data.get('settings', {})
        
        # Fix edge parameters to use more reasonable values
        if 'edge_width' in settings and settings['edge_width'] > 5:
            settings['edge_width'] = 3  # More reasonable value
            
        if 'edge_strength' in settings and settings['edge_strength'] > 5:
            settings['edge_strength'] = 2.0  # More reasonable value
        
        # Generate all preview types
        preview_types = ['preprocessed', 'segments', 'edges', 'template']
        
        for preview_type in preview_types:
            # Generate this preview
            preview_result = enhanced_integration.generate_preview(image_path, settings, preview_type)
            
            if preview_result and 'preview' in preview_result:
                # Ensure correct format
                preview_img = preview_result['preview']
                
                # Convert to proper format for saving
                if preview_img.dtype != np.uint8:
                    preview_img = (preview_img * 255).astype(np.uint8)
                
                # Make sure it's BGR for OpenCV saving
                if len(preview_img.shape) == 2:  # Grayscale
                    preview_img = cv2.cvtColor(preview_img, cv2.COLOR_GRAY2BGR)
                elif preview_img.shape[2] == 3:  # RGB
                    preview_img = cv2.cvtColor(preview_img, cv2.COLOR_RGB2BGR)
                
                # Save preview image
                preview_path = os.path.join(current_app.static_folder, 'previews')
                os.makedirs(preview_path, exist_ok=True)
                preview_file = os.path.join(preview_path, f"preview_{timestamp}_{preview_type}.jpg")
                cv2.imwrite(preview_file, preview_img)
        
        # Keep the same settings for the final process
        session['processing_settings'] = settings
        
        return jsonify({
            'success': True,
            'message': 'All previews generated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error generating previews: {e}", exc_info=True)
        return jsonify({
            'error': 'Preview generation error',
            'details': str(e)
        }), 500