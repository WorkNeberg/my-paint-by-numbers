import os
import numpy as np
from reportlab.lib.pagesizes import A4, LETTER, A3
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import landscape
from reportlab.platypus import PageBreak
from PIL import Image as PILImage
import matplotlib.pyplot as plt

class PBNPdfGenerator:
    def __init__(self):
        pass
        
    def generate_pdf(self, template_path, region_data, output_path, page_size='a4', 
                    include_unnumbered=False, include_alternate_styles=False):
        """Generate a PDF with the paint-by-numbers template and color chart"""
        print(f"Generating PDF: {output_path}")
        
        # Set page size
        if page_size.lower() == 'letter':
            page_dims = LETTER
            margin = 0.5 * inch
        elif page_size.lower() == 'a3':
            page_dims = A3
            margin = 1.0 * cm
        else:  # Default to A4
            page_dims = A4
            margin = 1.0 * cm
            
        # Create the PDF document
        doc = SimpleDocTemplate(output_path, 
                               pagesize=page_dims,
                               leftMargin=margin,
                               rightMargin=margin,
                               topMargin=margin,
                               bottomMargin=margin)
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            fontSize=16
        )
        
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Heading2'],
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            fontSize=14
        )
        
        normal_style = ParagraphStyle(
            'Normal',
            parent=styles['Normal'],
            alignment=TA_LEFT,
            fontSize=10
        )
        
        # Create content
        content = []
        
        # Title
        title = Paragraph("Paint by Numbers Template", title_style)
        content.append(title)
        content.append(Spacer(1, 0.5*cm))
        
        # Determine if we have alternative templates
        base_dir = os.path.dirname(template_path)
        base_name = os.path.basename(template_path)
        name_parts = base_name.split('_')
        timestamp = name_parts[-2] if len(name_parts) >= 2 else ""
        
        # Check if alternate styles exist
        minimal_path = os.path.join(base_dir, f"{name_parts[0]}_{timestamp}_minimal.png") 
        detailed_path = os.path.join(base_dir, f"{name_parts[0]}_{timestamp}_detailed.png")
        
        # Get chart path
        chart_path = os.path.join(base_dir, f"{name_parts[0]}_{timestamp}_chart.png")
        
        # Create full page template image
        self._add_full_page_image(content, template_path, "Standard Template", page_dims, title_style)
        
        # Add color chart
        if os.path.exists(chart_path):
            # Add a new page for the color chart
            content.append(PageBreak())
            content.append(Paragraph("Color Reference Chart", title_style))
            content.append(Spacer(1, 0.5*cm))
            
            # Create color reference table
            data = [["Color #", "Color Name", "Color"]]
            
            for region in region_data:
                color_cell = f'<font color="{region["color_hex"]}" size="14">■■■</font>'
                data.append([str(region["id"]), region["color_name"], Paragraph(color_cell, normal_style)])
                
            # Create table
            table = Table(data, colWidths=[1*cm, 3*cm, 1.5*cm])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            content.append(table)
            
            # Also add the color chart image
            content.append(Spacer(1, 1*cm))
            img_width, img_height = PILImage.open(chart_path).size
            scale_factor = min((page_dims[0] - 2*margin) / img_width, (page_dims[1] - 5*cm) / img_height)
            display_width = img_width * scale_factor
            display_height = img_height * scale_factor
            img = Image(chart_path, width=display_width, height=display_height)
            content.append(img)
        
        # Add alternative styles if requested
        if include_alternate_styles:
            # Check if minimal style exists
            if os.path.exists(minimal_path):
                content.append(PageBreak())  # Start new page
                self._add_full_page_image(content, minimal_path, "Minimal Style (Numbers Outside)", page_dims, title_style)
                
            # Check if detailed style exists
            if os.path.exists(detailed_path):
                content.append(PageBreak())  # Start new page
                self._add_full_page_image(content, detailed_path, "Detailed Style (All Numbers Inside)", page_dims, title_style)
        
        # Add unnumbered template if requested
        if include_unnumbered:
            # Generate an unnumbered version from the template by removing numbers
            unnumbered_path = os.path.join(base_dir, f"{name_parts[0]}_{timestamp}_unnumbered.png")
            
            # Try to load the template
            try:
                template_img = plt.imread(template_path)
                
                # Create a mask for black pixels (edges)
                # This is a simple approach - assumes edges are pure black
                edges_mask = np.all(template_img < 20, axis=2)
                
                # Create white background
                unnumbered = np.ones_like(template_img) * 255
                
                # Copy only the edges
                for i in range(3):
                    unnumbered[:,:,i] = np.where(edges_mask, 0, unnumbered[:,:,i])
                    
                plt.imsave(unnumbered_path, unnumbered)
                
                # Add to the PDF
                content.append(PageBreak())  # Start new page
                self._add_full_page_image(content, unnumbered_path, "Unnumbered Template", page_dims, title_style)
                
                # Clean up
                try:
                    if os.path.exists(unnumbered_path):
                        os.remove(unnumbered_path)
                except:
                    pass
            except Exception as e:
                print(f"Error creating unnumbered template: {str(e)}")
        
        # Build the PDF
        doc.build(content)
        print(f"PDF generated: {output_path}")
        
        return output_path
    
    def _add_full_page_image(self, content, image_path, title, page_dims, title_style):
        """Add an image that fills the full page with title"""
        # Add title
        content.append(Paragraph(title, title_style))
        content.append(Spacer(1, 0.5*cm))
        
        # Get image dimensions
        img_width, img_height = PILImage.open(image_path).size
        
        # Calculate max available space for the image
        available_height = page_dims[1] - 3*cm  # Leave room for title and margins
        available_width = page_dims[0] - 2*cm  # Leave margins
        
        # Determine orientation
        image_ratio = img_width / img_height
        page_ratio = available_width / available_height
        
        if image_ratio > page_ratio:
            # Image is wider than page proportion - fit to width
            display_width = available_width
            display_height = display_width / image_ratio
        else:
            # Image is taller than page proportion - fit to height
            display_height = available_height
            display_width = display_height * image_ratio
        
        # Add image
        img = Image(image_path, width=display_width, height=display_height)
        content.append(img)