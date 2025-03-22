# This file shows how to integrate the enhanced processing into your existing system
import cv2
import numpy as np
import matplotlib.pyplot as plt
from paint_by_numbers import PaintByNumbersGenerator
from enhanced_processor import EnhancedProcessor
from image_type_detector import ImageTypeDetector
from settings_manager import SettingsManager
import os

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape, A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import inch
import time

def enhance_paint_by_numbers_generator(pbn_generator):
    """Enhance an existing PaintByNumbersGenerator with our new features"""
    # Add new components to the generator
    pbn_generator.enhanced_processor = EnhancedProcessor()
    pbn_generator.type_detector = ImageTypeDetector()
    pbn_generator.settings_manager = SettingsManager()
    
    # Add new method for advanced processing
    def process_with_enhancements(self, image_path, preset_style=None, complexity_level='medium', 
                                output_dir='output', auto_detect_type=True):
        """
        Process image with advanced enhancements and specialized handling
        
        Parameters:
        - image_path: Path to input image
        - preset_style: Optional specific style preset to use
        - complexity_level: Level of complexity ('low', 'medium', 'high')
        - output_dir: Directory to save output
        - auto_detect_type: Whether to auto-detect the image type
        
        Returns:
        - results: Dictionary with paths to output files
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filenames
        basename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(basename)[0]
        timestamp = int(time.time())
        
        # Preprocess the image
        image = self.preprocess_image(image_path)
        
        # Detect image type if needed
        if auto_detect_type:
            image_type, confidence = self.type_detector.detect_type_simple(image)
            print(f"Auto-detected image type: {image_type} (confidence: {confidence:.2f})")
        else:
            # Default to general type
            image_type = 'general'
        
        # Get appropriate settings
        settings = self.settings_manager.get_settings(image_type, complexity_level, preset_style)
        print(f"Using settings for {image_type}, complexity: {complexity_level}, style: {preset_style}")
        
        # Process the image with enhanced processing
        result = self.enhanced_processor.process_with_feature_preservation(image, settings, image_type)
        
        # Create the output files using your existing methods
        vectorized = result['vectorized']
        label_image = result['label_image']
        edges = result['edges']
        centers = result['centers']
        region_data = result['region_data']
        paintability = result['paintability']
        
        # Enrich region data
        enriched_data = self.enrich_region_data(region_data, centers)
        
        # Use existing methods to create templates with the processed results
        template_path = os.path.join(output_dir, f"{name_without_ext}_{timestamp}_template.png")
        minimal_path = os.path.join(output_dir, f"{name_without_ext}_{timestamp}_minimal.png")
        detailed_path = os.path.join(output_dir, f"{name_without_ext}_{timestamp}_detailed.png")
        chart_path = os.path.join(output_dir, f"{name_without_ext}_{timestamp}_chart.png")
        preview_path = os.path.join(output_dir, f"{name_without_ext}_{timestamp}_preview.png")
        processed_path = os.path.join(output_dir, f"{name_without_ext}_{timestamp}_processed.png")
        
        # Create and save templates
        edge_style = settings.get('edge_style', 'normal')
        classic_template = self.create_template(
            vectorized.shape, edges, enriched_data, label_image, 
            style="classic", edge_style=edge_style
        )
        minimal_template = self.create_template(
            vectorized.shape, edges, enriched_data, label_image, 
            style="minimal", edge_style=edge_style
        )
        detailed_template = self.create_template(
            vectorized.shape, edges, enriched_data, label_image, 
            style="detailed", edge_style=edge_style
        )
        
        # Save outputs
        import matplotlib.pyplot as plt
        plt.imsave(template_path, classic_template)
        plt.imsave(minimal_path, minimal_template)
        plt.imsave(detailed_path, detailed_template)
        plt.imsave(processed_path, vectorized)
        
        # Create color chart
        color_chart = self.create_color_chart(enriched_data)
        plt.imsave(chart_path, color_chart)
        
        # Create badge
        badge = self.create_paintability_badge(paintability)
        
        # Create preview
        h, w = classic_template.shape[:2]
        badge_h, badge_w = badge.shape[:2]
        
        preview = np.ones((h, w*2 + badge_w + 20, 3), dtype=np.uint8) * 255
        preview[:, :w] = classic_template
        preview[:, w+10:w*2+10] = vectorized
        
        # Add the badge at the right side
        y_offset = max(0, (h - badge_h) // 2)
        badge_h_actual = min(badge_h, h - y_offset)
        
        if badge_h_actual > 0 and badge_w > 0:
            preview[y_offset:y_offset+badge_h_actual, w*2+20:w*2+20+badge_w] = badge[:badge_h_actual]
        
        plt.imsave(preview_path, preview)
        
        # Create PDF output
        pdf_path = os.path.join(output_dir, f"{name_without_ext}_{timestamp}_complete.pdf")
        self.create_pdf_output(
            pdf_path,
            classic_template,
            vectorized,
            color_chart,
            enriched_data,
            settings,
            image_type
        )
        
        return {
            'template': template_path,
            'minimal': minimal_path,
            'detailed': detailed_path,
            'chart': chart_path,
            'preview': preview_path,
            'processed': processed_path,
            'pdf': pdf_path,
            'region_data': enriched_data,
            'paintability': paintability,
            'image_type': image_type,
            'settings': settings
        }
    
    # Add PDF creation method
    def create_pdf_output(self, pdf_path, template, processed, color_chart, region_data, settings, image_type):
        """Create a comprehensive PDF output with template, color chart and instructions"""
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
        from reportlab.lib.units import inch
        import io
        import matplotlib.pyplot as plt
        
        # Convert images to format usable by reportlab
        def image_to_reportlab(img):
            img_bytes = io.BytesIO()
            plt.imsave(img_bytes, img, format='png')
            img_bytes.seek(0)
            return ImageReader(img_bytes)
        
        template_img = image_to_reportlab(template)
        processed_img = image_to_reportlab(processed)
        chart_img = image_to_reportlab(color_chart)
        
        # Create PDF
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
        
        # Add title
        c.setFont("Helvetica-Bold", 18)
        title = f"Paint by Numbers - {image_type.capitalize()}"
        c.drawCentredString(width/2, height - 30, title)
        
        # Add template on first page
        img_width = width - 100
        img_height = img_width * template.shape[0] / template.shape[1]
        c.drawImage(template_img, 50, height - 60 - img_height, width=img_width, height=img_height)
        
        # Add settings info
        c.setFont("Helvetica", 10)
        y_pos = height - 80 - img_height
        c.drawString(50, y_pos, f"Number of colors: {settings.get('num_colors', 15)}")
        c.drawString(50, y_pos - 15, f"Style: {settings.get('style_name', 'Standard')}")
        c.drawString(50, y_pos - 30, f"Edge style: {settings.get('edge_style', 'normal')}")
        
        # Add instructions
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_pos - 60, "Instructions:")
        c.setFont("Helvetica", 10)
        instructions = [
            "1. Match the color numbers on the template to the color chart",
            "2. Paint each region with its corresponding color",
            "3. Start from the top of the image and work your way down",
            "4. Allow each section to dry before painting adjacent areas"
        ]
        for i, line in enumerate(instructions):
            c.drawString(50, y_pos - 80 - i*15, line)
        
        c.showPage()
        
        # Second page: color chart and reference image
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 30, "Color Reference Chart")
        
        # Add reference image
        ref_width = width / 3
        ref_height = ref_width * processed.shape[0] / processed.shape[1]
        c.drawImage(processed_img, width - ref_width - 50, height - 50 - ref_height, 
                   width=ref_width, height=ref_height)
        
        # Add color chart
        chart_width = width / 2
        chart_height = chart_width * color_chart.shape[0] / color_chart.shape[1]
        c.drawImage(chart_img, 50, height - 60 - chart_height, width=chart_width, height=chart_height)
        
        c.save()
    
    # Add the methods to the generator
    setattr(pbn_generator.__class__, 'process_with_enhancements', process_with_enhancements)
    setattr(pbn_generator.__class__, 'create_pdf_output', create_pdf_output)
    
    return pbn_generator

# Example usage
def test_enhanced_generator():
    # Create the original generator
    pbn_generator = PaintByNumbersGenerator()
    
    # Enhance it with our new features
    enhanced_generator = enhance_paint_by_numbers_generator(pbn_generator)
    
    # Use the enhanced generator
    image_path = "example.jpg"
    results = enhanced_generator.process_with_enhancements(
        image_path, 
        preset_style="davinci",  # For portraits
        complexity_level="medium",
        auto_detect_type=True
    )
    
    print(f"Generated files: {results}")
    
    # Create variants for comparison
    image = enhanced_generator.preprocess_image(image_path)
    variants = enhanced_generator.enhanced_processor.create_comparison_variants(image)
    
    print(f"Created {len(variants)} comparison variants")