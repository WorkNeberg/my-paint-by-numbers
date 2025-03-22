from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from core.paint_by_numbers import PaintByNumbersGenerator
from utils.pdf_generator import PBNPdfGenerator
from enhanced.integration import enhance_paint_by_numbers_generator
from enhanced.image_type_detector import ImageTypeDetector
from enhanced.enhanced_processor import EnhancedProcessor
import os
import time
import uuid
import logging
import cv2
import numpy as np
from datetime import datetime

# Third-party imports
import matplotlib.pyplot as plt

# Project imports
# from enhanced.pet_pattern_preserver import PetPatternPreserver  # Already implemented

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('pbn-app')

# Get the absolute path of the current directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
logger.info(f"Base directory: {BASE_DIR}")

# Initialize Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

# File upload settings
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
VARIANTS_FOLDER = os.path.join(BASE_DIR, 'static', 'variants')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Create required directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(VARIANTS_FOLDER, exist_ok=True)

# Set upload folder in Flask config
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

app.config['OUTPUT_FOLDER'] = os.path.join(BASE_DIR, 'output')
app.config['PREVIEW_FOLDER'] = os.path.join(BASE_DIR, 'preview')

# Create the folders if they don't exist
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['PREVIEW_FOLDER'], exist_ok=True)

# Initialize the generators
pbn_generator = PaintByNumbersGenerator()
pdf_generator = PBNPdfGenerator()
enhanced_generator = enhance_paint_by_numbers_generator(pbn_generator)
enhanced_processor = EnhancedProcessor()
image_type_detector = ImageTypeDetector()

# Helper function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function for flashing messages to the user
def flash_message(message, category='info'):
    if 'messages' not in session:
        session['messages'] = []
    session['messages'].append((message, category))

@app.route('/')
def index():
    messages = session.pop('messages', [])
    return render_template('index.html', messages=messages)

@app.route('/results')
def results():
    logger.info("Results route called")
    logger.info(f"Session keys: {list(session.keys())}")
    
    if 'result' not in session:
        logger.info("No result in session, redirecting to index")
        return redirect(url_for('index'))
        
    result = session['result']
    logger.info(f"Result found with keys: {list(result.keys())}")
    return render_template('results.html', result=result)

@app.route('/convert', methods=['POST'])
def convert():
    """Process uploaded image and convert to paint by numbers"""
    # Add this line at the very beginning of the function
    start_time = time.time()
    
    logger.info("========= CONVERT ROUTE STARTED =========")
    
    try:
        # Get form data with logging
        logger.info("Step 1: Getting form data")
        processing_mode = request.form.get('processing_mode', 'auto')
        preset_type = request.form.get('preset', '')
        complexity = request.form.get('complexity', 'medium')
        num_colors = int(request.form.get('colors', 15))
        
        logger.info(f"Form data: mode={processing_mode}, preset={preset_type}, complexity={complexity}, colors={num_colors}")
        
        # Check for file with logging
        logger.info("Step 2: Checking file upload")
        if 'image' not in request.files:  # Using 'image' instead of 'file'
            logger.warning("No file in request.files - redirecting")
            flash("No file selected", "error")
            return redirect(url_for('index'))
            
        file = request.files['image']  # Using 'image' instead of 'file'
        logger.info(f"File object: {file}, filename: {file.filename}")
        
        if file.filename == '':
            logger.warning("Empty filename - redirecting")
            flash("No file selected", "error")
            return redirect(url_for('index'))
        
        # Check file type with logging
        logger.info(f"Step 3: Validating file type")
        if not allowed_file(file.filename):
            logger.warning(f"Invalid file type: {file.filename} - redirecting")
            flash("File type not supported. Please upload a JPG, JPEG, or PNG file.", "error")
            return redirect(url_for('index'))
        
        # Save file with logging    
        logger.info("Step 4: Saving uploaded file")
        unique_id = str(int(time.time()))
        filename = f"{unique_id}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        logger.info(f"Saving to path: {filepath}")
        try:
            file.save(filepath)
            logger.info(f"File saved successfully. Size: {os.path.getsize(filepath)} bytes")
        except Exception as e:
            logger.error(f"File save error: {str(e)}")
            flash("Error saving uploaded file", "error")
            return redirect(url_for('index'))
        
        # Determine image type with logging
        logger.info("Step 5: Determining image type")
        if processing_mode == 'preset' and preset_type == 'pet':
            image_type = 'pet'
            logger.info("Pet preset selected - setting image_type to pet")
        else:
            image_type = image_type_detector.detect_image_type(filepath)
            logger.info(f"Auto-detected image type: {image_type}")
        
        # Create settings with logging
        logger.info("Step 6: Creating settings dictionary")
        settings = {
            'complexity': complexity,
            'colors': num_colors,
            'image_type': image_type,
            'preset_type': preset_type
        }
        logger.info(f"Settings: {settings}")
        
        # In the convert route, modify the settings for pet images

        if settings['image_type'] == 'pet':
            # Settings tuned for better pet recognition
            settings.update({
                'edge_strength': 1.2,  # Stronger edges for key features
                'edge_width': 1,       # Keep edges thin
                'simplification_level': 'medium',  # Changed from low
                'edge_style': 'hierarchical',  # New style - implement below
                'merge_regions_level': 'smart'  # New level - implement below
            })
        
        # Update pet settings:

        if settings['image_type'] == 'pet':
            settings.update({
                'colors': min(9, settings['colors']),  # Limit to 9 colors max for clearer numbering
                'edge_strength': 1.5,  # Stronger outlines
                'edge_width': 2,       # Thicker outlines
                'simplification_level': 'high',  # More simplification to reduce "wool"
                'edge_style': 'hierarchical',  # Important outlines darker
                'merge_regions_level': 'aggressive'  # Reduce small regions
            })
        
        # Simplify pet settings
        if settings['image_type'] == 'pet':
            settings.update({
                'colors': 9,  # Hard limit to 9 colors for pets
                'simplification_level': 'high',  # Reduce detail in fur
                'merge_regions_level': 'aggressive'  # Merge similar regions
            })
        
        # Modify the pet settings in app.py:

        if settings['image_type'] == 'pet':
            settings.update({
                'colors': 7,  # Even fewer colors for pets (7 is ideal for painting)
                'simplification_level': 'very_high',  # Maximum simplification 
                'edge_strength': 2.0,  # Stronger pet outline
                'edge_width': 3,  # Thicker outline
                'edge_style': 'hierarchical'
            })
        
        # Process image with logging
        logger.info("Step 7: Starting image processing")
        try:
            result = enhanced_processor.process_image(filepath, settings)
            logger.info("Image processing completed successfully")
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}", exc_info=True)
            flash("Error processing image", "error")
            return redirect(url_for('index'))
        
        # Save results with logging
        logger.info("Step 8: Saving output files")
        try:
            # Create output dirs if needed
            os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
            os.makedirs(app.config['PREVIEW_FOLDER'], exist_ok=True)
            
            # Generate filenames
            output_filename = f"{unique_id}_output.png"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            
            preview_filename = f"{unique_id}_preview.png"
            preview_path = os.path.join(app.config['PREVIEW_FOLDER'], preview_filename)
            
            # Save files
            logger.info(f"Saving output to {output_path}")
            cv2.imwrite(output_path, cv2.cvtColor(result['segmented_image'], cv2.COLOR_RGB2BGR))
            
            logger.info(f"Saving preview to {preview_path}")
            cv2.imwrite(preview_path, cv2.cvtColor(result['preview_image'], cv2.COLOR_RGB2BGR))
            
            logger.info("Files saved successfully")
            
            # Save to session
            session['result'] = {
                'output_url': f"/output/{output_filename}",
                'preview_url': f"/preview/{preview_filename}",
                'colors': result['colors'],
                'processing_time': f"{time.time() - start_time:.1f}",
                'paintability': result.get('paintability', 75)
            }
            logger.info("Session updated with results")
            
            # Add this code after saving the preview image in the convert function

            # Generate paint-by-numbers template with numbers
            try:
                logger.info("Generating paint-by-numbers template with numbers")
                
                # Create template output path
                template_filename = f"{unique_id}_numbered.png"
                template_path = os.path.join(app.config['OUTPUT_FOLDER'], template_filename)
                
                # Generate the numbered template
                numbered_image = pbn_generator.generate_numbered_template(
                    result['segmented_image'], 
                    result['colors'],
                    complexity=settings.get('complexity', 'medium')
                )
                
                # Save the numbered template
                cv2.imwrite(template_path, cv2.cvtColor(numbered_image, cv2.COLOR_RGB2BGR))
                logger.info(f"Numbered template saved to {template_path}")
                
                # Add numbered template URL to result
                session['result']['numbered_template_url'] = f"/output/{template_filename}"
                
                # Generate PDF from the template file
                pdf_filename = f"{unique_id}_template.pdf"
                pdf_path = os.path.join(app.config['OUTPUT_FOLDER'], pdf_filename)
                
                # Replace your PDF generation code with this simpler version

                # Generate the PDF without the unnumbered option
                try:
                    logger.info(f"Generating PDF: {pdf_path}")
                    pdf_generator.generate_pdf(
                        template_path,  # Numbered template path
                        result['colors'],
                        output_path=pdf_path,
                        page_size=request.form.get('page_size', 'a4'),
                        include_alternate_styles=True,
                        include_unnumbered=False  # Set to False to avoid the error
                    )
                    
                    logger.info(f"PDF saved to {pdf_path}")
                    
                    # Add PDF URL to result
                    session['result']['pdf_url'] = f"/output/{pdf_filename}"
                    
                except Exception as e:
                    logger.error(f"Error generating PDF: {str(e)}", exc_info=True)
                    # Continue without PDF if generation fails

            except Exception as e:
                logger.error(f"Error generating template/PDF: {str(e)}", exc_info=True)
                # Continue without template/PDF if generation fails

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}", exc_info=True)
            flash("Error saving processed image", "error")
            return redirect(url_for('index'))
        
        logger.info("========= CONVERT ROUTE COMPLETED =========")
        return redirect(url_for('results'))
        
    except Exception as e:
        logger.error(f"Unhandled exception in convert route: {str(e)}", exc_info=True)
        flash("An unexpected error occurred", "error")
        return redirect(url_for('index'))

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

@app.route('/api/process_image', methods=['POST'])
def api_process_image():
    # Check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['image']
    
    # If user does not select file, browser submits an empty file without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    # Check if the file is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file format. Allowed formats: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
    # Save the file
    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    safe_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
    file.save(filepath)
    
    # Get parameters from request
    data = request.form.to_dict()
    
    # Generate unique output directory
    output_dir = os.path.join(OUTPUT_FOLDER, f"pbn_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the image
    try:
        # Detect image type
        image_type = image_type_detector.detect_image_type(filepath)
        
        # Set processing parameters
        params = {
            'complexity': data.get('complexity', 'medium'),
            'colors': int(data.get('colors', 15)),
            'style': data.get('style', 'classic'),
            'image_type': image_type
        }
        
        # Add pet-specific processing if it's a pet image
        if image_type == 'pet':
            params['preserve_pet_patterns'] = True
            
        # Process with enhanced generator
        result = enhanced_processor.process_image(filepath, params)
        
        # Return API response
        return jsonify({
            'success': True,
            'result': {
                'image': result.get('output_url', ''),
                'pdf': result.get('pdf_url', ''),
                'preview': result.get('preview_url', ''),
                'colors': result.get('colors', []),
            }
        })
        
    except Exception as e:
        logger.error(f"API Error processing image: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process_image():
    try:
        # Get uploaded file
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        safe_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(filepath)
        
        # Get form parameters
        complexity = request.form.get('complexity', 'medium')
        colors = int(request.form.get('colors', 15))
        style = request.form.get('style', 'classic')
        
        # Detect image type
        image_type = image_type_detector.detect_image_type(filepath)
        
        # Set up processing parameters
        settings = {
            'complexity': complexity,
            'colors': colors,
            'style': style,
            'image_type': image_type
        }
        
        # Add pet-specific processing if it's a pet image
        if image_type == 'pet':
            settings['preserve_pet_patterns'] = True
        
        # Process the image
        result = enhanced_processor.process_image(filepath, settings)
        
        # Return response
        return jsonify({
            'success': True,
            'result_url': result.get('output_url'),
            'preview_url': result.get('preview_url'),
            'download_url': result.get('download_url')
        })
        
    except Exception as e:
        logger.error(f"Error in process_image: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Function that should be defined outside the route
def test_image_with_multiple_styles(image_path, output_dir):
    """
    Generate multiple processing variants of the same image using different settings.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save the variant results
        
    Returns:
        Dictionary with paths to generated variants and metadata
    """
    print(f"Generating variants for {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to RGB for processing
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Define variants to generate
    variants = [
        {
            "name": "Standard Processing",
            "settings": {
                "complexity": "medium",
                "colors": 15
            }
        },
        {
            "name": "Enhanced Detail",
            "settings": {
                "complexity": "high",
                "colors": 20,
                "enhanced": True
            }
        }
    ]
    
    # Detect if image is a pet
    image_type = image_type_detector.detect_image_type(image_path)
    is_pet = (image_type == 'pet')
    
    # Add pet-specific variants if it's a pet image
    if is_pet:
        variants.append({
            "name": "Pet Standard",
            "settings": {
                "complexity": "medium",
                "colors": 15,
                "image_type": "pet"
            }
        })
        variants.append({
            "name": "Pet with Pattern Preservation",
            "settings": {
                "complexity": "medium",
                "colors": 15,
                "image_type": "pet",
                "preserve_pet_patterns": True
            }
        })
    
    # Process each variant
    results = []
    
    for variant in variants:
        print(f"Processing variant: {variant['name']}")
        start_time = time.time()
        
        settings = variant["settings"]
        output_path = os.path.join(output_dir, f"{variant['name'].replace(' ', '_').lower()}.png")
        
        # Process the image
        result = enhanced_processor.process_image(image.copy(), settings)
        
        # Handle different result formats
        output_image = None
        
        # Try to get processed image from result
        if 'processed_image' in result:
            output_image = result['processed_image']
        elif 'output_image' in result:
            output_image = result['output_image']
        elif 'output_url' in result:
            # If result contains URL, try to load the image
            try:
                output_path_from_url = result['output_url']
                if os.path.exists(output_path_from_url):
                    output_image = cv2.imread(output_path_from_url)
                    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.error(f"Error loading output image from URL: {e}")
        
        # Save the output image
        if output_image is not None:
            cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
        else:
            logger.warning(f"No output image found for variant: {variant['name']}")
            # Create a placeholder image
            placeholder = np.ones((400, 400, 3), dtype=np.uint8) * 200  # Light gray
            cv2.putText(placeholder, f"No output for {variant['name']}", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.imwrite(output_path, placeholder)
        
        processing_time = time.time() - start_time
        
        # Store result info
        results.append({
            "name": variant["name"],
            "path": output_path.replace('\\', '/'),  # Ensure proper path format for web
            "processing_time": f"{processing_time:.2f}s",
            "settings": settings
        })
    
    # Also save the original image for comparison
    original_path = os.path.join(output_dir, "original.png")
    cv2.imwrite(original_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    return {
        "variants": results,
        "original": original_path.replace('\\', '/'),  # Ensure proper path format for web
    }

# Your route handler - now using the function defined above
@app.route('/variants', methods=['POST'])
def generate_variants():
    try:
        # Get uploaded file
        if 'image' not in request.files:
            flash_message('No file uploaded', 'error')
            return redirect(url_for('index'))
            
        file = request.files['image']
        if file.filename == '':
            flash_message('No file selected', 'error')
            return redirect(url_for('index'))
            
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        temp_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_{filename}")
        file.save(temp_path)
        
        # Create unique output directory
        output_dir = os.path.join(VARIANTS_FOLDER, f'comparison_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        
        # Use the function defined above instead of importing
        result = test_image_with_multiple_styles(temp_path, output_dir)
        
        # Prepare data for rendering the comparison template
        return render_template('variants.html', 
                               variants=result["variants"],
                               original=result["original"],
                               timestamp=timestamp)
                              
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Error in generate_variants: {e}", exc_info=True)
        flash_message(f'Error generating variants: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/debug_form', methods=['POST'])
def debug_form():
    """Debug route to see what's in the form"""
    logger.info("Form data:")
    for key, value in request.form.items():
        logger.info(f"  {key}: {value}")
    return jsonify(dict(request.form))

@app.route('/test_pet_processing')
def test_pet_processing():
    """Test route that processes a sample image as a pet"""
    logger.info("===== TEST PET PROCESSING ROUTE =====")
    
    # Use a sample image from the static folder
    sample_path = os.path.join(app.static_folder, 'sample_cat.jpg')
    if not os.path.exists(sample_path):
        return "Sample image not found"
    
    # Set up pet processing settings
    settings = {
        'complexity': 'medium',
        'colors': 15,
        'image_type': 'pet',
        'preset_type': 'pet'
    }
    
    try:
        # Process the image
        logger.info("Calling process_image with pet settings")
        result = enhanced_processor.process_image(sample_path, settings)
        logger.info("Processing complete")
        
        # Return success
        return "Pet processing successful! Check logs for details."
    except Exception as e:
        logger.error(f"Error in test processing: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

@app.route('/output/<filename>')
def serve_output(filename):
    """Serve files from the output directory"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/preview/<filename>')
def serve_preview(filename):
    """Serve files from the preview directory"""
    return send_from_directory(app.config['PREVIEW_FOLDER'], filename)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}", exc_info=True)
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)