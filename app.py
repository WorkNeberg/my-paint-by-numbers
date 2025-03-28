import os
import logging
from flask import Flask, render_template, redirect, url_for, send_from_directory
from enhanced_ui import enhanced_bp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'paint_by_numbers.log'))
    ]
)

# Create logger
logger = logging.getLogger('pbn-app')

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

# Register the enhanced UI blueprint
app.register_blueprint(enhanced_bp)

# Redirect root to enhanced UI
@app.route('/')
def index():
    return redirect(url_for('enhanced.index'))

# Add this route at the app level to handle all uploaded files
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Serve files from the upload directory"""
    upload_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    return send_from_directory(upload_folder, filename)

@app.route('/static/previews/<path:filename>')
def serve_preview(filename):
    """Serve files from the previews directory"""
    preview_dir = os.path.join(app.static_folder, 'previews')
    return send_from_directory(preview_dir, filename)

# Add this route to serve files from the output directory

@app.route('/output/<path:filename>')
def serve_output(filename):
    """Serve files from the output directory"""
    output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    return send_from_directory(output_folder, filename)

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('enhanced/error.html', error_code=404, error_message="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return render_template('enhanced/error.html', error_code=500, error_message="Server error"), 500

# Run the app
if __name__ == '__main__':
    # Create required directories if they don't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), 'uploads'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'static', 'previews'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'output'), exist_ok=True)
    
    # Create configs directory for settings and presets
    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    presets_dir = os.path.join(configs_dir, 'presets')
    os.makedirs(configs_dir, exist_ok=True)
    os.makedirs(presets_dir, exist_ok=True)
    
    # Run the app
    app.run(debug=True)