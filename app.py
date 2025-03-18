import os
import uuid
import time
import json
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, session
from werkzeug.utils import secure_filename
from paint_by_numbers import PaintByNumbersGenerator
from pdf_generator import PBNPdfGenerator
from integration import enhance_paint_by_numbers_generator

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
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Create required directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Set upload folder in Flask config
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize the generators
pbn_generator = PaintByNumbersGenerator()
pdf_generator = PBNPdfGenerator()
enhanced_generator = enhance_paint_by_numbers_generator(pbn_generator)

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
    """Render the main page"""
    logger.info("Rendering index.html template")
    return render_template('index.html')

@app.route('/results')
def results():
    """Show results page"""
    # Get parameters from query string
    template_url = request.args.get('template_url')
    chart_url = request.args.get('chart_url')
    preview_url = request.args.get('preview_url')
    pdf_url = request.args.get('pdf_url')
    paintability = request.args.get('paintability')
    
    if paintability:
        try:
            paintability = int(paintability)
        except:
            paintability = None
    
    # Get additional URLs if available
    minimal_url = request.args.get('minimal_url')
    detailed_url = request.args.get('detailed_url')
    
    # Image analysis
    analysis = None
    image_type = request.args.get('image_type')
    dark_percentage = request.args.get('dark_percentage')
    edge_complexity = request.args.get('edge_complexity')
    
    if image_type:
        analysis = {
            'image_type': image_type,
            'dark_percentage': dark_percentage or 0,
            'edge_complexity': edge_complexity or 'medium'
        }
    
    # Validate required URLs
    if not template_url or not preview_url:
        return redirect(url_for('index'))
    
    return render_template('results.html',
                          template_url=template_url,
                          chart_url=chart_url or template_url,
                          preview_url=preview_url,
                          pdf_url=pdf_url or template_url,
                          paintability=paintability,
                          minimal_url=minimal_url,
                          detailed_url=detailed_url,
                          analysis=analysis)

@app.route('/convert', methods=['POST'])
def convert():
    """Process image and create paint-by-numbers template"""
    # Check if file was uploaded
    if 'image' not in request.files:
        logger.warning("No file part in convert request")
        return redirect(url_for('index'))
    
    file = request.files['image']
    if file.filename == '':
        logger.warning("No selected file in convert request")
        return redirect(url_for('index'))
    
    if not allowed_file(file.filename):
        logger.warning(f"Invalid file format: {file.filename}")
        flash_message("Invalid file format. Please use JPG, PNG or WEBP.", "error")
        return redirect(url_for('index'))
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        logger.warning(f"File too large: {file_size} bytes")
        flash_message("File too large. Maximum size is 10MB.", "error")
        return redirect(url_for('index'))
    
    # Generate unique ID for this upload
    unique_id = str(uuid.uuid4())
    timestamp = int(time.time())
    
    # Get processing mode
    processing_mode = request.form.get('processing_mode', 'auto')
    logger.info(f"Processing mode: {processing_mode}")

    # Determine parameters based on mode
    parameters = {
        'template_style': request.form.get('template_style', 'classic')
    }
    
    if processing_mode == 'auto':
        # Auto-detect mode - minimal parameters
        parameters['auto_detect'] = True
        
    elif processing_mode == 'preset':
        # Preset mode - use a predefined preset
        preset_name = request.form.get('preset', 'portrait')
        parameters['auto_detect'] = False
        
        # Add preset-specific parameters
        if preset_name == 'portrait':
            parameters.update({
                'num_colors': 12,
                'simplification_level': 'low',
                'edge_strength': 0.8,
                'edge_width': 1,
                'enhance_dark_areas': True,
                'dark_threshold': 50,
                'edge_style': 'soft',
                'image_type': 'portrait',
                'merge_regions_level': 'normal'
            })
        elif preset_name == 'pet':
            parameters.update({
                'num_colors': 10,
                'simplification_level': 'medium',
                'edge_strength': 0.7,
                'edge_width': 1,
                'enhance_dark_areas': True,
                'dark_threshold': 60,
                'edge_style': 'soft',
                'image_type': 'pet',
                'merge_regions_level': 'aggressive'
            })
        elif preset_name == 'landscape':
            parameters.update({
                'num_colors': 18,
                'simplification_level': 'medium',
                'edge_strength': 1.0,
                'edge_width': 2,
                'enhance_dark_areas': False,
                'edge_style': 'normal',
                'image_type': 'landscape',
                'merge_regions_level': 'low'
            })
        elif preset_name == 'cartoon':
            parameters.update({
                'num_colors': 8,
                'simplification_level': 'high',
                'edge_strength': 1.2,
                'edge_width': 2,
                'enhance_dark_areas': False,
                'edge_style': 'normal',
                'image_type': 'cartoon',
                'merge_regions_level': 'normal'
            })
    elif processing_mode == 'custom':
        # Custom mode - use user-specified parameters
        parameters.update({
            'auto_detect': False,
            'num_colors': int(request.form.get('num_colors', 15)),
            'simplification_level': request.form.get('simplification_level', 'medium'),
            'enhance_dark_areas': 'enhance_dark' in request.form
        })
        
    # Add common parameters
    page_size = request.form.get('page_size', 'a4')
    include_unnumbered = 'include_unnumbered' in request.form
    include_alternate_styles = 'include_alternate_styles' in request.form
    
    # Save uploaded file with explicit flushing and verification
    file_ext = os.path.splitext(file.filename)[1]
    upload_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}{file_ext}")
    
    try:
        file.save(upload_path)
        file.close()  # Ensure file is closed
        
        # Verify the file was saved correctly
        if not os.path.exists(upload_path):
            logger.error(f"Failed to save uploaded file at {upload_path}")
            flash_message("Failed to upload file. Please try again.", "error")
            return redirect(url_for('index'))
        
        file_size = os.path.getsize(upload_path)
        logger.info(f"Saved file: {upload_path}, Size: {file_size} bytes")
        
        if file_size == 0:
            logger.error("Uploaded file is empty")
            flash_message("The uploaded file is empty. Please try again.", "error")
            return redirect(url_for('index'))
            
        # Process the image with smart analysis and optimal parameters
        logger.info("Starting image processing with smart analysis...")
        result = enhanced_generator.process_with_enhancements(
            upload_path, 
            output_dir=OUTPUT_FOLDER,
            **parameters
        )
        
        # Generate PDF
        logger.info("Generating PDF...")
        pdf_path = os.path.join(OUTPUT_FOLDER, f"{unique_id}_{timestamp}_pbn.pdf")
        pdf_generator.generate_pdf(
            result['template'], 
            result['region_data'], 
            pdf_path, 
            page_size=page_size, 
            include_unnumbered=include_unnumbered,
            include_alternate_styles=include_alternate_styles
        )
        
        # Create URLs for front-end
        template_url = f"/download/{os.path.basename(result['template'])}"
        chart_url = f"/download/{os.path.basename(result['chart'])}"
        preview_url = f"/download/{os.path.basename(result['preview'])}"
        pdf_url = f"/download/{os.path.basename(pdf_path)}"
        
        # Prepare redirect to results page with parameters
        redirect_params = {
            'template_url': template_url,
            'chart_url': chart_url,
            'preview_url': preview_url,
            'pdf_url': pdf_url
        }
        
        # Add paintability score if available
        if 'paintability' in result:
            redirect_params['paintability'] = result['paintability']
        
        # Add analysis information if available
        if 'image_type' in result:
            redirect_params['image_type'] = result['image_type']
            redirect_params['dark_percentage'] = result.get('dark_percentage', 0)
            redirect_params['edge_complexity'] = result.get('edge_complexity', 'medium')
        
        # Add URLs for alternate styles if they exist
        for key in ['minimal', 'detailed']:
            if key in result:
                redirect_params[f'{key}_url'] = f"/download/{os.path.basename(result[key])}"
        
        logger.info("Processing completed successfully")
        
        # Redirect to results page with all parameters
        return redirect(url_for('results', **redirect_params))
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        flash_message(f"Error processing image: {str(e)}", "error")
        return redirect(url_for('index'))
    finally:
        # Clean up the uploaded file
        try:
            if os.path.exists(upload_path):
                os.remove(upload_path)
                logger.info(f"Removed temporary file: {upload_path}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary file: {str(e)}")

@app.route('/download/<filename>')
def download_file(filename):
    """Serve generated files for download"""
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/api/process_image', methods=['POST'])
def api_process_image():
    """API endpoint for image processing"""
    if 'image' not in request.files:
        logger.warning("No image file provided in API request")
        return jsonify({
            'success': False,
            'error': 'No image file provided'
        }), 400
        
    # Get parameters from request
    try:
        params = json.loads(request.form.get('params', '{}'))
    except:
        params = {}
    
    # Process similar to /convert route
    file = request.files['image']
    logger.info(f"API processing image: {file.filename}")
    
    # Generate unique ID
    unique_id = str(uuid.uuid4())
    timestamp = int(time.time())
    
    # Save uploaded file
    file_ext = os.path.splitext(file.filename)[1]
    upload_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}{file_ext}")
    
    try:
        file.save(upload_path)
        logger.info(f"Saved API uploaded file to {upload_path}")
        
        # Process with auto-detect by default
        parameters = {
            'auto_detect': True,
            'template_style': params.get('template_style', 'classic')
        }
        
        # Override with any provided parameters
        for key in ['num_colors', 'simplification_level', 'edge_strength',
                   'edge_width', 'enhance_dark_areas', 'dark_threshold']:
            if key in params:
                parameters[key] = params[key]
        
        # Process the image
        logger.info(f"API processing with parameters: {parameters}")
        result = enhanced_generator.process_with_enhancements(
            upload_path, 
            output_dir=OUTPUT_FOLDER,
            **parameters
        )
        
        # Generate PDF
        pdf_path = os.path.join(OUTPUT_FOLDER, f"{unique_id}_{timestamp}_pbn.pdf")
        pdf_generator.generate_pdf(
            result['template'], 
            result['region_data'], 
            pdf_path, 
            page_size=params.get('page_size', 'a4'), 
            include_unnumbered=params.get('include_unnumbered', False),
            include_alternate_styles=params.get('include_alternate_styles', False)
        )
        
        # Create response with file URLs
        response = {
            'success': True,
            'files': {
                'template': f"/download/{os.path.basename(result['template'])}",
                'chart': f"/download/{os.path.basename(result['chart'])}",
                'preview': f"/download/{os.path.basename(result['preview'])}",
                'pdf': f"/download/{os.path.basename(pdf_path)}"
            },
            'paintability': result.get('paintability', 50)
        }
        
        logger.info("API processing completed successfully")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"API processing error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    finally:
        # Clean up
        if os.path.exists(upload_path):
            try:
                os.remove(upload_path)
                logger.info(f"Removed API temporary file: {upload_path}")
            except:
                pass

@app.route('/process', methods=['POST'])
def process_image():
    try:
        # Get uploaded file
        if 'image' not in request.files:
            flash_message('No file uploaded', 'error')
            return redirect(url_for('index'))
            
        file = request.files['image']
        if file.filename == '':
            flash_message('No file selected', 'error')
            return redirect(url_for('index'))
            
        # Get form parameters
        preset = request.form.get('preset', None)
        complexity = request.form.get('complexity', 'medium')
        
        # Save uploaded file temporarily
        temp_path = os.path.join('uploads', secure_filename(file.filename))
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        file.save(temp_path)
        
        # Process with enhanced generator
        results = enhanced_generator.process_with_enhancements(
            temp_path,
            preset_style=preset,
            simplification_level=complexity,
            output_dir='static/outputs',
            auto_detect_type=(preset is None)  # Only auto-detect if no preset selected
        )
        
        # Extract needed paths for display
        result_data = {
            'preview': os.path.basename(results['preview']),
            'template': os.path.basename(results['template']),
            'processed': os.path.basename(results['processed']),
            'chart': os.path.basename(results['chart']),
            'pdf': os.path.basename(results['pdf']),
            'paintability': results['paintability'],
            'image_type': results['image_type']
        }
        
        return render_template('results.html', results=result_data)
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        flash_message(f"Error processing image: {str(e)}", 'error')
        return redirect(url_for('index'))

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
        temp_path = os.path.join('uploads', secure_filename(file.filename))
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        file.save(temp_path)
        
        # Create unique output directory
        timestamp = int(time.time())
        output_dir = os.path.join('static', 'variants', f'comparison_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        
        # Import the test function
        from test_variants import test_image_with_multiple_styles
        
        # Generate variants
        results = test_image_with_multiple_styles(temp_path, output_dir)
        
        # Prepare results for display
        comparison_data = {
            'processed_comparison': os.path.join('variants', f'comparison_{timestamp}', 'processed_comparison.png'),
            'template_comparison': os.path.join('variants', f'comparison_{timestamp}', 'template_comparison.png'),
            'pdf_report': os.path.join('variants', f'comparison_{timestamp}', 'variants_comparison.pdf')
        }
        
        return render_template('variants.html', results=comparison_data)
        
    except Exception as e:
        logger.error(f"Error generating variants: {str(e)}")
        flash_message(f"Error generating variants: {str(e)}", 'error')
        return redirect(url_for('index'))   

@app.errorhandler(404)
def page_not_found(e):
    """404 error handler"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """500 error handler"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    logger.info("Starting Paint by Numbers Generator app...")
    app.run(debug=True, host='0.0.0.0', port=5000)