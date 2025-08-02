"""
Image Extraction Utilities for Document Processing

This module provides utilities for extracting and rendering actual images from PDFs
using PyMuPDF (fitz) library.
"""

import fitz  # PyMuPDF
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import io
import base64
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class PDFImageExtractor:
    """Utility class for extracting images from PDF documents."""
    
    def __init__(self, output_dir: str = "output/extracted_images"):
        """Initialize the image extractor.
        
        Args:
            output_dir: Directory to save extracted images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract all images from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing image information
        """
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)
            images = []
            
            logger.info(f"Processing PDF: {pdf_path}")
            logger.info(f"Total pages: {len(doc)}")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get image list for this page
                image_list = page.get_images()
                
                logger.info(f"Page {page_num + 1}: Found {len(image_list)} images")
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]  # Cross-reference number
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Convert to PIL Image
                        img_data = pix.tobytes("png")
                        pil_image = Image.open(io.BytesIO(img_data))
                        
                        # Get image metadata
                        img_info = {
                            "page_number": page_num + 1,
                            "image_index": img_index,
                            "xref": xref,
                            "width": pil_image.width,
                            "height": pil_image.height,
                            "format": pil_image.format,
                            "mode": pil_image.mode,
                            "size_bytes": len(img_data),
                            "bbox": img[1],  # Bounding box
                            "rotation": img[2],  # Rotation
                            "colorspace": img[3],  # Colorspace
                            "bits_per_component": img[4],  # Bits per component
                            "image_data": img_data,  # Raw image data
                            "pil_image": pil_image  # PIL Image object
                        }
                        
                        images.append(img_info)
                        
                        # Clean up
                        pix = None
                        
                    except Exception as e:
                        logger.warning(f"Error extracting image {img_index} from page {page_num + 1}: {e}")
                        continue
            
            doc.close()
            logger.info(f"Successfully extracted {len(images)} images from {pdf_path}")
            return images
            
        except Exception as e:
            logger.error(f"Error extracting images from {pdf_path}: {e}")
            return []
    
    def save_extracted_images(self, images: List[Dict[str, Any]], pdf_name: str) -> List[str]:
        """Save extracted images to files.
        
        Args:
            images: List of image dictionaries
            pdf_name: Name of the source PDF (for naming files)
            
        Returns:
            List of saved image file paths
        """
        saved_paths = []
        
        # Create subdirectory for this PDF
        pdf_dir = self.output_dir / pdf_name.replace('.pdf', '')
        pdf_dir.mkdir(exist_ok=True)
        
        for i, img_info in enumerate(images):
            try:
                # Create filename
                filename = f"page_{img_info['page_number']:03d}_img_{img_info['image_index']:03d}.png"
                filepath = pdf_dir / filename
                
                # Save image
                pil_image = img_info['pil_image']
                pil_image.save(filepath, 'PNG')
                
                # Update image info with file path
                img_info['saved_path'] = str(filepath)
                saved_paths.append(str(filepath))
                
                logger.info(f"Saved image: {filepath}")
                
            except Exception as e:
                logger.error(f"Error saving image {i}: {e}")
                continue
        
        return saved_paths
    
    def create_image_gallery(self, images: List[Dict[str, Any]], pdf_name: str) -> str:
        """Create an HTML gallery of extracted images.
        
        Args:
            images: List of image dictionaries
            pdf_name: Name of the source PDF
            
        Returns:
            Path to the generated HTML gallery
        """
        try:
            # Create HTML content
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Gallery - {pdf_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #333;
            color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            padding: 20px;
        }}
        .image-card {{
            background: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .image-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }}
        .image-container {{
            text-align: center;
            margin-bottom: 10px;
        }}
        .image-container img {{
            max-width: 100%;
            max-height: 300px;
            border-radius: 5px;
            border: 1px solid #ddd;
        }}
        .image-info {{
            font-size: 12px;
            color: #666;
            margin-top: 10px;
        }}
        .stats {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .stats h3 {{
            margin-top: 0;
            color: #333;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .stat-item {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📸 Image Gallery</h1>
        <p>Extracted from: {pdf_name}</p>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="stats">
        <h3>📊 Extraction Statistics</h3>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value">{len(images)}</div>
                <div class="stat-label">Total Images</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{len(set(img['page_number'] for img in images))}</div>
                <div class="stat-label">Pages with Images</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{sum(img['size_bytes'] for img in images) / 1024:.1f} KB</div>
                <div class="stat-label">Total Size</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{len([img for img in images if img['width'] > img['height']])}</div>
                <div class="stat-label">Landscape Images</div>
            </div>
        </div>
    </div>
    
    <div class="gallery">
"""
            
            # Add each image to the gallery
            for i, img_info in enumerate(images):
                # Convert image data to base64 for embedding
                img_data = img_info['image_data']
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                
                html_content += f"""
        <div class="image-card">
            <div class="image-container">
                <img src="data:image/png;base64,{img_base64}" 
                     alt="Image {i+1}" 
                     title="Page {img_info['page_number']}, Image {img_info['image_index']}">
            </div>
            <div class="image-info">
                <strong>Page {img_info['page_number']}, Image {img_info['image_index']}</strong><br>
                Size: {img_info['width']} × {img_info['height']} pixels<br>
                Format: {img_info['format']} | Mode: {img_info['mode']}<br>
                File Size: {img_info['size_bytes'] / 1024:.1f} KB<br>
                Rotation: {img_info['rotation']}°<br>
                Colorspace: {img_info['colorspace']}
            </div>
        </div>
"""
            
            html_content += """
    </div>
</body>
</html>
"""
            
            # Save HTML file
            gallery_path = self.output_dir / f"{pdf_name.replace('.pdf', '')}_gallery.html"
            with open(gallery_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Created image gallery: {gallery_path}")
            return str(gallery_path)
            
        except Exception as e:
            logger.error(f"Error creating image gallery: {e}")
            return ""
    
    def create_image_summary_report(self, images: List[Dict[str, Any]], pdf_name: str) -> str:
        """Create a summary report of extracted images.
        
        Args:
            images: List of image dictionaries
            pdf_name: Name of the source PDF
            
        Returns:
            Path to the generated report
        """
        try:
            # Calculate statistics
            total_images = len(images)
            total_size = sum(img['size_bytes'] for img in images)
            pages_with_images = len(set(img['page_number'] for img in images))
            landscape_images = len([img for img in images if img['width'] > img['height']])
            portrait_images = len([img for img in images if img['width'] <= img['height']])
            
            # Group by page
            images_by_page = {}
            for img in images:
                page = img['page_number']
                if page not in images_by_page:
                    images_by_page[page] = []
                images_by_page[page].append(img)
            
            # Create report content
            report_content = f"""# Image Extraction Report

## 📄 Source Document
- **PDF Name**: {pdf_name}
- **Extraction Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Summary Statistics
- **Total Images Extracted**: {total_images}
- **Pages with Images**: {pages_with_images}
- **Total Size**: {total_size / 1024:.1f} KB
- **Average Size per Image**: {total_size / total_images / 1024:.1f} KB
- **Landscape Images**: {landscape_images}
- **Portrait Images**: {portrait_images}

## 📋 Image Details by Page

"""
            
            for page_num in sorted(images_by_page.keys()):
                page_images = images_by_page[page_num]
                report_content += f"### Page {page_num}\n"
                report_content += f"- **Images on this page**: {len(page_images)}\n"
                
                for i, img in enumerate(page_images):
                    report_content += f"- **Image {i+1}**: {img['width']} × {img['height']} pixels, {img['size_bytes'] / 1024:.1f} KB\n"
                
                report_content += "\n"
            
            # Save report
            report_path = self.output_dir / f"{pdf_name.replace('.pdf', '')}_report.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"Created image report: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error creating image report: {e}")
            return ""
    
    def extract_and_render_images(self, pdf_path: str) -> Dict[str, Any]:
        """Complete workflow to extract and render images from a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extraction results
        """
        try:
            logger.info(f"Starting image extraction from: {pdf_path}")
            
            # Extract images
            images = self.extract_images_from_pdf(pdf_path)
            
            if not images:
                logger.warning(f"No images found in {pdf_path}")
                return {
                    "pdf_path": pdf_path,
                    "total_images": 0,
                    "saved_paths": [],
                    "gallery_path": "",
                    "report_path": ""
                }
            
            # Get PDF name for file naming
            pdf_name = Path(pdf_path).name
            
            # Save images
            saved_paths = self.save_extracted_images(images, pdf_name)
            
            # Create gallery
            gallery_path = self.create_image_gallery(images, pdf_name)
            
            # Create report
            report_path = self.create_image_summary_report(images, pdf_name)
            
            # Calculate statistics
            total_size = sum(img['size_bytes'] for img in images)
            pages_with_images = len(set(img['page_number'] for img in images))
            
            results = {
                "pdf_path": pdf_path,
                "total_images": len(images),
                "total_size_bytes": total_size,
                "total_size_kb": total_size / 1024,
                "pages_with_images": pages_with_images,
                "saved_paths": saved_paths,
                "gallery_path": gallery_path,
                "report_path": report_path,
                "images": images
            }
            
            logger.info(f"Image extraction completed successfully!")
            logger.info(f"  - Total images: {len(images)}")
            logger.info(f"  - Total size: {total_size / 1024:.1f} KB")
            logger.info(f"  - Pages with images: {pages_with_images}")
            logger.info(f"  - Gallery: {gallery_path}")
            logger.info(f"  - Report: {report_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in image extraction workflow: {e}")
            return {
                "pdf_path": pdf_path,
                "error": str(e),
                "total_images": 0,
                "saved_paths": [],
                "gallery_path": "",
                "report_path": ""
            }


def extract_images_from_pdf_with_most_images(analysis_file: str, storage_dir: str = "storage/TogafD") -> Dict[str, Any]:
    """Extract images from the PDF with the most images.
    
    Args:
        analysis_file: Path to the analysis JSON file
        storage_dir: Directory containing the PDF files
        
    Returns:
        Dictionary containing extraction results
    """
    try:
        # Load analysis data
        with open(analysis_file, 'r') as f:
            analysis_data = json.load(f)
        
        # Find PDF with most images
        images_by_document = analysis_data.get('images_by_document', {})
        
        if not images_by_document:
            logger.error("No image data found in analysis file")
            return {}
        
        # Find document with most images
        max_images = 0
        pdf_with_most_images = None
        
        for doc_id, image_count in images_by_document.items():
            if image_count > max_images:
                max_images = image_count
                pdf_with_most_images = doc_id
        
        if not pdf_with_most_images:
            logger.error("No PDF with images found")
            return {}
        
        # Extract PDF filename from document ID
        # The document ID format is: pdf_togaf-standard-introduction-and-core-concepts:latest:01-doc:chap03_1753225643.893056
        # We need to extract: togaf-standard-introduction-and-core-concepts:latest:01-doc:chap03.pdf
        
        # Remove the 'pdf_' prefix and the timestamp suffix
        if pdf_with_most_images.startswith('pdf_'):
            # Remove 'pdf_' prefix
            doc_id_without_prefix = pdf_with_most_images[4:]  # Remove 'pdf_'
            
            # Find the last underscore (before the timestamp)
            last_underscore_index = doc_id_without_prefix.rfind('_')
            if last_underscore_index != -1:
                # Remove the timestamp suffix
                pdf_filename = doc_id_without_prefix[:last_underscore_index] + ".pdf"
            else:
                pdf_filename = doc_id_without_prefix + ".pdf"
        else:
            pdf_filename = pdf_with_most_images + ".pdf"
        
        pdf_path = Path(storage_dir) / pdf_filename
        
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return {}
        
        logger.info(f"Found PDF with most images: {pdf_filename} ({max_images} images)")
        
        # Extract images
        extractor = PDFImageExtractor()
        results = extractor.extract_and_render_images(str(pdf_path))
        
        return results
        
    except Exception as e:
        logger.error(f"Error extracting images from PDF with most images: {e}")
        return {}


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        analysis_file = sys.argv[1]
        results = extract_images_from_pdf_with_most_images(analysis_file)
        print("Image extraction results:")
        print(json.dumps(results, indent=2, default=str))
    else:
        print("Usage: python image_extraction_utils.py <analysis_file.json>") 