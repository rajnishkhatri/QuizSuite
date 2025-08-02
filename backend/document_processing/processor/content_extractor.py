"""
Content Extractor Module

This module provides comprehensive content extraction capabilities for the document processing pipeline.
It handles extraction of images, figures, tables, and code chunks from PDF documents.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)


class ContentExtractor:
    """
    Comprehensive content extractor for PDF documents.
    
    Extracts images, figures, tables, and code chunks from PDF documents
    and provides structured output for the document processing pipeline.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the content extractor.
        
        Args:
            output_dir: Directory to save extracted content (optional)
        """
        self.output_dir = output_dir
        self.extracted_content = {
            'images': [],
            'tables': [],
            'code_chunks': [],
            'figures': []  # Figures are a subset of images with specific characteristics
        }
        
        # Code patterns for detection
        self.code_patterns = [
            r'```[\s\S]*?```',  # Markdown code blocks
            r'`[^`]+`',  # Inline code
            r'^\s*[A-Za-z_][A-Za-z0-9_]*\s*[:=]\s*[^;]+;?$',  # Variable assignments
            r'^\s*(if|for|while|def|class|import|from)\s+',  # Code keywords
            r'^\s*[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*{?$',  # Function definitions
            r'^\s*[A-Za-z_][A-Za-z0-9_]*\s*\.\s*[A-Za-z_][A-Za-z0-9_]*\s*\(',  # Method calls
            r'^\s*<!--.*-->$',  # HTML comments
            r'^\s*//.*$',  # Single line comments
            r'^\s*/\*.*\*/$',  # Multi-line comments
            r'^\s*#.*$',  # Python/bash comments
        ]
    
    def extract_all_content(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract all content types from a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing all extracted content with metadata
        """
        logger.info(f"Starting comprehensive content extraction from: {pdf_path}")
        
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            logger.info(f"Processing PDF with {total_pages} pages")
            
            # Initialize results
            results = {
                'pdf_path': pdf_path,
                'total_pages': total_pages,
                'images': [],
                'tables': [],
                'code_chunks': [],
                'figures': [],
                'metadata': {
                    'total_images': 0,
                    'total_tables': 0,
                    'total_code_chunks': 0,
                    'total_figures': 0,
                    'pages_with_images': 0,
                    'pages_with_tables': 0,
                    'pages_with_code': 0,
                    'pages_with_figures': 0
                }
            }
            
            # Process each page
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                page_results = self._extract_from_page(page, page_num + 1)
                
                # Aggregate results
                results['images'].extend(page_results['images'])
                results['tables'].extend(page_results['tables'])
                results['code_chunks'].extend(page_results['code_chunks'])
                results['figures'].extend(page_results['figures'])
                
                # Update metadata
                if page_results['images']:
                    results['metadata']['pages_with_images'] += 1
                if page_results['tables']:
                    results['metadata']['pages_with_tables'] += 1
                if page_results['code_chunks']:
                    results['metadata']['pages_with_code'] += 1
                if page_results['figures']:
                    results['metadata']['pages_with_figures'] += 1
            
            # Update total counts
            results['metadata']['total_images'] = len(results['images'])
            results['metadata']['total_tables'] = len(results['tables'])
            results['metadata']['total_code_chunks'] = len(results['code_chunks'])
            results['metadata']['total_figures'] = len(results['figures'])
            
            doc.close()
            
            logger.info(f"Content extraction completed successfully")
            logger.info(f"  - Images: {results['metadata']['total_images']}")
            logger.info(f"  - Tables: {results['metadata']['total_tables']}")
            logger.info(f"  - Code chunks: {results['metadata']['total_code_chunks']}")
            logger.info(f"  - Figures: {results['metadata']['total_figures']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during content extraction: {e}")
            raise
    
    def _extract_from_page(self, page: fitz.Page, page_num: int) -> Dict[str, List]:
        """
        Extract all content types from a single page.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number
            
        Returns:
            Dictionary containing extracted content from the page
        """
        page_results = {
            'images': [],
            'tables': [],
            'code_chunks': [],
            'figures': []
        }
        
        # Extract images
        images = self._extract_images_from_page(page, page_num)
        page_results['images'] = images
        
        # Extract tables
        tables = self._extract_tables_from_page(page, page_num)
        page_results['tables'] = tables
        
        # Extract code chunks
        code_chunks = self._extract_code_from_page(page, page_num)
        page_results['code_chunks'] = code_chunks
        
        # Identify figures (subset of images with specific characteristics)
        figures = self._identify_figures(images, page_num)
        page_results['figures'] = figures
        
        return page_results
    
    def _extract_images_from_page(self, page: fitz.Page, page_num: int) -> List[Dict]:
        """
        Extract images from a page.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number
            
        Returns:
            List of image dictionaries with metadata
        """
        images = []
        
        try:
            # Get image list from page
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    pix = fitz.Pixmap(page.parent, xref)
                    
                    # Create image metadata
                    image_info = {
                        'page_number': page_num,
                        'image_index': img_index + 1,
                        'width': pix.width,
                        'height': pix.height,
                        'colorspace': pix.colorspace.name if pix.colorspace else 'unknown',
                        'size_bytes': len(pix.tobytes()),
                        'bbox': img[2],  # Bounding box
                        'type': 'image',
                        'extracted_text': '',
                        'ocr_success': False
                    }
                    
                    # Try OCR if image is large enough
                    if pix.width > 100 and pix.height > 100:
                        try:
                            # Convert to PIL Image for OCR
                            img_data = pix.tobytes("png")
                            pil_image = Image.open(io.BytesIO(img_data))
                            
                            # Perform OCR
                            text = pytesseract.image_to_string(pil_image)
                            if text.strip():
                                image_info['extracted_text'] = text.strip()
                                image_info['ocr_success'] = True
                                image_info['text_lines'] = len(text.strip().split('\n'))
                                image_info['text_characters'] = len(text.strip())
                        except Exception as ocr_error:
                            logger.debug(f"OCR failed for image {img_index} on page {page_num}: {ocr_error}")
                    
                    images.append(image_info)
                    
                except Exception as img_error:
                    logger.warning(f"Error processing image {img_index} on page {page_num}: {img_error}")
                    continue
            
            if images:
                logger.info(f"Page {page_num}: Extracted {len(images)} images")
            else:
                logger.debug(f"Page {page_num}: No images found")
                
        except Exception as e:
            logger.error(f"Error extracting images from page {page_num}: {e}")
        
        return images
    
    def _extract_tables_from_page(self, page: fitz.Page, page_num: int) -> List[Dict]:
        """
        Extract tables from a page.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number
            
        Returns:
            List of table dictionaries with metadata
        """
        tables = []
        
        try:
            # Use PyMuPDF table finder
            table_finder = page.find_tables()
            
            if table_finder and table_finder.tables:
                for table_index, table in enumerate(table_finder.tables):
                    try:
                        # Extract table data
                        table_data = []
                        for row in table.extract():
                            table_data.append([cell.strip() if cell else "" for cell in row])
                        
                        # Create table metadata
                        table_info = {
                            'page_number': page_num,
                            'table_index': table_index + 1,
                            'rows': len(table_data),
                            'columns': len(table_data[0]) if table_data else 0,
                            'data': table_data,
                            'bbox': table.bbox,
                            'type': 'table',
                            'total_cells': sum(len(row) for row in table_data),
                            'non_empty_cells': sum(1 for row in table_data for cell in row if cell.strip())
                        }
                        
                        tables.append(table_info)
                        
                    except Exception as table_error:
                        logger.warning(f"Error processing table {table_index} on page {page_num}: {table_error}")
                        continue
                
                logger.info(f"Page {page_num}: Extracted {len(tables)} tables")
            else:
                logger.debug(f"Page {page_num}: No tables found")
                
        except Exception as e:
            logger.error(f"Error extracting tables from page {page_num}: {e}")
        
        return tables
    
    def _extract_code_from_page(self, page: fitz.Page, page_num: int) -> List[Dict]:
        """
        Extract code chunks from a page.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number
            
        Returns:
            List of code chunk dictionaries with metadata
        """
        code_chunks = []
        
        try:
            # Extract text from page
            text = page.get_text()
            
            # Look for code patterns
            found_code = []
            
            # Check for markdown code blocks
            markdown_blocks = re.findall(r'```[\s\S]*?```', text)
            for i, block in enumerate(markdown_blocks):
                found_code.append({
                    'type': 'markdown_code_block',
                    'content': block,
                    'index': i + 1
                })
            
            # Check for inline code
            inline_codes = re.findall(r'`([^`]+)`', text)
            for i, code in enumerate(inline_codes):
                found_code.append({
                    'type': 'inline_code',
                    'content': code,
                    'index': i + 1
                })
            
            # Check for code-like patterns
            lines = text.split('\n')
            code_lines = []
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if line:
                    # Check various code patterns
                    for pattern in self.code_patterns[2:]:  # Skip markdown patterns
                        if re.match(pattern, line):
                            code_lines.append({
                                'line_number': line_num,
                                'content': line,
                                'pattern_matched': pattern
                            })
                            break
            
            # Create code chunk objects
            for code_item in found_code:
                code_chunk = {
                    'page_number': page_num,
                    'chunk_index': len(code_chunks) + 1,
                    'type': code_item['type'],
                    'content': code_item['content'],
                    'lines': len(code_item['content'].split('\n')),
                    'characters': len(code_item['content']),
                    'bbox': None,  # Code chunks don't have specific bounding boxes
                    'code_type': 'structured'
                }
                code_chunks.append(code_chunk)
            
            # Add code-like lines as a single chunk
            if code_lines:
                code_chunk = {
                    'page_number': page_num,
                    'chunk_index': len(code_chunks) + 1,
                    'type': 'code_like_lines',
                    'content': '\n'.join([line['content'] for line in code_lines]),
                    'lines': len(code_lines),
                    'characters': sum(len(line['content']) for line in code_lines),
                    'bbox': None,
                    'code_type': 'pattern_matched',
                    'line_details': code_lines
                }
                code_chunks.append(code_chunk)
            
            if code_chunks:
                logger.info(f"Page {page_num}: Extracted {len(code_chunks)} code chunks")
            else:
                logger.debug(f"Page {page_num}: No code chunks found")
                
        except Exception as e:
            logger.error(f"Error extracting code from page {page_num}: {e}")
        
        return code_chunks
    
    def _identify_figures(self, images: List[Dict], page_num: int) -> List[Dict]:
        """
        Identify figures from images based on characteristics.
        
        Args:
            images: List of image dictionaries
            page_num: Page number
            
        Returns:
            List of figure dictionaries
        """
        figures = []
        
        for image in images:
            # Apply figure identification logic
            is_figure = self._is_figure(image)
            
            if is_figure:
                figure_info = {
                    **image,
                    'type': 'figure',
                    'figure_type': self._classify_figure(image),
                    'page_number': page_num
                }
                figures.append(figure_info)
        
        if figures:
            logger.info(f"Page {page_num}: Identified {len(figures)} figures")
        
        return figures
    
    def _is_figure(self, image: Dict) -> bool:
        """
        Determine if an image is a figure based on characteristics.
        
        Args:
            image: Image dictionary
            
        Returns:
            True if the image is classified as a figure
        """
        # Figure identification criteria
        width = image.get('width', 0)
        height = image.get('height', 0)
        has_text = image.get('ocr_success', False)
        
        # Figures typically have:
        # 1. Reasonable size (not too small, not too large)
        # 2. Often contain text (diagrams, charts, etc.)
        # 3. Specific aspect ratios
        
        if width < 50 or height < 50:
            return False  # Too small
        
        if width > 2000 or height > 2000:
            return False  # Too large (likely a full-page image)
        
        # Check aspect ratio (figures often have specific ratios)
        aspect_ratio = width / height if height > 0 else 0
        if 0.5 <= aspect_ratio <= 2.0:
            return True
        
        # If it has extracted text, it's likely a figure
        if has_text:
            return True
        
        return False
    
    def _classify_figure(self, image: Dict) -> str:
        """
        Classify the type of figure.
        
        Args:
            image: Image dictionary
            
        Returns:
            Figure type classification
        """
        width = image.get('width', 0)
        height = image.get('height', 0)
        has_text = image.get('ocr_success', False)
        aspect_ratio = width / height if height > 0 else 0
        
        if has_text:
            return 'text_diagram'
        elif aspect_ratio > 1.5:
            return 'wide_diagram'
        elif aspect_ratio < 0.7:
            return 'tall_diagram'
        else:
            return 'standard_figure'
    
    def save_extracted_content(self, results: Dict[str, Any], output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Save extracted content to files.
        
        Args:
            results: Extraction results dictionary
            output_dir: Output directory (uses self.output_dir if not provided)
            
        Returns:
            Dictionary with file paths and metadata
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        if output_dir is None:
            logger.warning("No output directory specified, skipping file save")
            return results
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {
            'images': [],
            'tables': [],
            'code_chunks': [],
            'figures': []
        }
        
        # Save images
        for i, image in enumerate(results['images']):
            try:
                # Create image metadata file
                image_file = output_dir / f"image_{i+1:03d}_page_{image['page_number']:03d}.json"
                with open(image_file, 'w') as f:
                    json.dump(image, f, indent=2)
                saved_files['images'].append(str(image_file))
            except Exception as e:
                logger.error(f"Error saving image {i+1}: {e}")
        
        # Save tables
        for i, table in enumerate(results['tables']):
            try:
                # Save as CSV
                csv_file = output_dir / f"table_{i+1:03d}_page_{table['page_number']:03d}.csv"
                df = pd.DataFrame(table['data'])
                df.to_csv(csv_file, index=False, header=False)
                saved_files['tables'].append(str(csv_file))
                
                # Save metadata
                meta_file = output_dir / f"table_{i+1:03d}_page_{table['page_number']:03d}_meta.json"
                table_meta = {k: v for k, v in table.items() if k != 'data'}
                with open(meta_file, 'w') as f:
                    json.dump(table_meta, f, indent=2)
                    
            except Exception as e:
                logger.error(f"Error saving table {i+1}: {e}")
        
        # Save code chunks
        for i, code in enumerate(results['code_chunks']):
            try:
                code_file = output_dir / f"code_{i+1:03d}_page_{code['page_number']:03d}.txt"
                with open(code_file, 'w', encoding='utf-8') as f:
                    f.write(code['content'])
                saved_files['code_chunks'].append(str(code_file))
            except Exception as e:
                logger.error(f"Error saving code chunk {i+1}: {e}")
        
        # Save figures
        for i, figure in enumerate(results['figures']):
            try:
                figure_file = output_dir / f"figure_{i+1:03d}_page_{figure['page_number']:03d}.json"
                with open(figure_file, 'w') as f:
                    json.dump(figure, f, indent=2)
                saved_files['figures'].append(str(figure_file))
            except Exception as e:
                logger.error(f"Error saving figure {i+1}: {e}")
        
        results['saved_files'] = saved_files
        logger.info(f"Saved extracted content to {output_dir}")
        
        return results
    
    def get_extraction_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of the extraction results.
        
        Args:
            results: Extraction results dictionary
            
        Returns:
            Summary dictionary
        """
        metadata = results.get('metadata', {})
        
        summary = {
            'total_pages': results.get('total_pages', 0),
            'extraction_stats': {
                'images': {
                    'total': metadata.get('total_images', 0),
                    'pages_with_images': metadata.get('pages_with_images', 0),
                    'with_ocr_text': sum(1 for img in results.get('images', []) if img.get('ocr_success', False))
                },
                'tables': {
                    'total': metadata.get('total_tables', 0),
                    'pages_with_tables': metadata.get('pages_with_tables', 0),
                    'total_cells': sum(tbl.get('total_cells', 0) for tbl in results.get('tables', []))
                },
                'code_chunks': {
                    'total': metadata.get('total_code_chunks', 0),
                    'pages_with_code': metadata.get('pages_with_code', 0),
                    'by_type': self._count_code_types(results.get('code_chunks', []))
                },
                'figures': {
                    'total': metadata.get('total_figures', 0),
                    'pages_with_figures': metadata.get('pages_with_figures', 0),
                    'by_type': self._count_figure_types(results.get('figures', []))
                }
            },
            'success_rate': {
                'images': metadata.get('total_images', 0) / max(results.get('total_pages', 1), 1),
                'tables': metadata.get('total_tables', 0) / max(results.get('total_pages', 1), 1),
                'code_chunks': metadata.get('total_code_chunks', 0) / max(results.get('total_pages', 1), 1),
                'figures': metadata.get('total_figures', 0) / max(results.get('total_pages', 1), 1)
            }
        }
        
        return summary
    
    def _count_code_types(self, code_chunks: List[Dict]) -> Dict[str, int]:
        """Count code chunks by type."""
        type_counts = {}
        for chunk in code_chunks:
            chunk_type = chunk.get('type', 'unknown')
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        return type_counts
    
    def _count_figure_types(self, figures: List[Dict]) -> Dict[str, int]:
        """Count figures by type."""
        type_counts = {}
        for figure in figures:
            fig_type = figure.get('figure_type', 'unknown')
            type_counts[fig_type] = type_counts.get(fig_type, 0) + 1
        return type_counts


# Import required modules
import io
import json 