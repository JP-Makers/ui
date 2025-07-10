#!/usr/bin/env python3
"""
High-accuracy PDF text, heading, and table extraction script.
Extracts content from PDFs and outputs structured JSON format.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import pdfplumber
import pandas as pd

from utils.text_analyzer import TextAnalyzer
from utils.table_processor import TableProcessor
from utils.json_formatter import JSONFormatter
from config.extraction_config import ExtractionConfig


class PDFExtractor:
    """Main PDF extraction class with advanced parsing capabilities."""
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.text_analyzer = TextAnalyzer(config)
        self.table_processor = TableProcessor(config)
        self.json_formatter = JSONFormatter(config)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('pdf_extraction.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def extract_pdf_content(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract all content from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing extracted content
        """
        try:
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            self.logger.info(f"Starting extraction of: {pdf_path}")
            
            extracted_content = {
                "document_info": {
                    "filename": pdf_file.name,
                    "path": str(pdf_file.absolute()),
                    "total_pages": 0
                },
                "text_content": [],
                "headings": [],
                "tables": [],
                "extraction_metadata": {
                    "extraction_method": "pdfplumber",
                    "config_version": self.config.version
                }
            }
            
            with pdfplumber.open(pdf_path) as pdf:
                extracted_content["document_info"]["total_pages"] = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages, 1):
                    self.logger.info(f"Processing page {page_num} of {len(pdf.pages)}")
                    
                    # Extract page content
                    page_content = self._extract_page_content(page, page_num)
                    
                    # Merge content
                    extracted_content["text_content"].extend(page_content["text"])
                    extracted_content["headings"].extend(page_content["headings"])
                    extracted_content["tables"].extend(page_content["tables"])
            
            self.logger.info(f"Extraction completed successfully. Found {len(extracted_content['text_content'])} text blocks, "
                           f"{len(extracted_content['headings'])} headings, and {len(extracted_content['tables'])} tables.")
            
            return extracted_content
            
        except Exception as e:
            self.logger.error(f"Error extracting PDF content: {str(e)}")
            raise
    
    def _extract_page_content(self, page: pdfplumber.page.Page, page_num: int) -> Dict[str, List]:
        """Extract content from a single page."""
        page_content = {
            "text": [],
            "headings": [],
            "tables": []
        }
        
        try:
            # Extract tables first to avoid including table text in regular text
            tables = self.table_processor.extract_tables(page, page_num)
            page_content["tables"] = tables
            
            # Get table bounding boxes to exclude from text extraction
            table_bboxes = self.table_processor.get_table_bboxes(page)
            
            # Extract text and analyze for headings
            chars = page.chars
            if chars:
                text_blocks, headings = self.text_analyzer.analyze_text(chars, page_num, table_bboxes)
                page_content["text"] = text_blocks
                page_content["headings"] = headings
            
        except Exception as e:
            self.logger.error(f"Error processing page {page_num}: {str(e)}")
            # Continue with empty content for this page
        
        return page_content
    
    def save_to_json(self, content: Dict[str, Any], output_path: str) -> None:
        """Save extracted content to JSON file."""
        try:
            formatted_content = self.json_formatter.format_content(content)
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(formatted_content, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Content saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving JSON: {str(e)}")
            raise
    
    def print_summary(self, content: Dict[str, Any]) -> None:
        """Print extraction summary to console."""
        print("\n" + "="*60)
        print("PDF EXTRACTION SUMMARY")
        print("="*60)
        print(f"Document: {content['document_info']['filename']}")
        print(f"Total Pages: {content['document_info']['total_pages']}")
        print(f"Text Blocks: {len(content['text_content'])}")
        print(f"Headings: {len(content['headings'])}")
        print(f"Tables: {len(content['tables'])}")
        
        if content['headings']:
            print("\nHeadings Found:")
            for i, heading in enumerate(content['headings'][:10], 1):  # Show first 10
                print(f"  {i}. {heading['text'][:60]}{'...' if len(heading['text']) > 60 else ''}")
            if len(content['headings']) > 10:
                print(f"  ... and {len(content['headings']) - 10} more")
        
        if content['tables']:
            print("\nTables Found:")
            for i, table in enumerate(content['tables'], 1):
                print(f"  Table {i}: {table['rows']} rows × {table['columns']} columns (Page {table['page']})")
        
        print("="*60)


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Extract text, headings, and tables from PDF files with high accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pdf_extractor.py document.pdf
  python pdf_extractor.py document.pdf -o extracted_content.json
  python pdf_extractor.py document.pdf --log-level DEBUG
        """
    )
    
    parser.add_argument(
        'pdf_path',
        help='Path to the PDF file to extract content from'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output JSON file path (default: <pdf_name>_extracted.json)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set the logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--config',
        help='Path to custom configuration file'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = ExtractionConfig()
        config.log_level = args.log_level
        
        if args.config:
            config.load_from_file(args.config)
        
        # Initialize extractor
        extractor = PDFExtractor(config)
        
        # Extract content
        print(f"Extracting content from: {args.pdf_path}")
        content = extractor.extract_pdf_content(args.pdf_path)
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            pdf_name = Path(args.pdf_path).stem
            output_path = f"{pdf_name}_extracted.json"
        
        # Save to JSON
        extractor.save_to_json(content, output_path)
        
        # Print summary
        extractor.print_summary(content)
        
        print(f"\nExtraction completed successfully!")
        print(f"Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
