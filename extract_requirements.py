#!/usr/bin/env python3
"""
Enhanced PDF Requirements Extraction Script
Extracts requirements from PDF files with improved accuracy and multiple parsing strategies
"""

import pdfplumber
import PyPDF2
import fitz  # PyMuPDF
import csv
import re
import sys
import os
import argparse
from typing import List, Dict, Set, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFRequirementsExtractor:
    """Enhanced PDF requirements extractor with multiple parsing strategies"""
    
    def __init__(self):
        self.req_pattern = re.compile(r'REQ_[A-Z0-9_]+', re.IGNORECASE)
        self.stats = {
            'pages_processed': 0,
            'tables_found': 0,
            'text_blocks_found': 0,
            'requirements_found': 0,
            'parsing_errors': 0
        }
        
    def extract_requirements(self, pdf_path: str) -> List[Dict[str, str]]:
        """
        Extract requirements from PDF using multiple strategies
        Returns list of requirements with Request ID and Description
        """
        logger.info(f"Starting requirement extraction from {pdf_path}")
        
        # Reset stats
        self.stats = {key: 0 for key in self.stats.keys()}
        
        all_requirements = []
        
        # Get total pages first (for accurate statistics)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
            self.stats['pages_processed'] = total_pages
        except Exception as e:
            logger.warning(f"Could not get page count: {e}")
        
        # Strategy 1: pdfplumber (best for tables and structured text)
        try:
            plumber_reqs = self._extract_with_pdfplumber(pdf_path)
            all_requirements.extend(plumber_reqs)
            logger.info(f"pdfplumber found {len(plumber_reqs)} requirements")
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            
        # Strategy 2: PyMuPDF (good for complex layouts)
        try:
            pymupdf_reqs = self._extract_with_pymupdf(pdf_path)
            all_requirements.extend(pymupdf_reqs)
            logger.info(f"PyMuPDF found {len(pymupdf_reqs)} requirements")
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            
        # Strategy 3: PyPDF2 (fallback)
        try:
            pypdf2_reqs = self._extract_with_pypdf2(pdf_path)
            all_requirements.extend(pypdf2_reqs)
            logger.info(f"PyPDF2 found {len(pypdf2_reqs)} requirements")
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
        
        # Deduplicate and clean requirements
        unique_requirements = self._deduplicate_requirements(all_requirements)
        
        self.stats['requirements_found'] = len(unique_requirements)
        logger.info(f"Total unique requirements found: {len(unique_requirements)}")
        
        return unique_requirements
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> List[Dict[str, str]]:
        """Extract requirements using pdfplumber"""
        requirements = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                
                # Extract from tables first
                tables = page.extract_tables()
                self.stats['tables_found'] += len(tables)
                
                for table in tables:
                    if not table:
                        continue
                    
                    # Look for requirements in table cells
                    for row in table:
                        for cell in row:
                            if cell and 'REQ_' in cell:
                                req_matches = self.req_pattern.findall(cell)
                                for req_id in req_matches:
                                    description = self._extract_description_from_cell(cell, req_id)
                                    if description:
                                        requirements.append({
                                            'Request ID': req_id,
                                            'Description': self._clean_text(description)
                                        })
                
                # Extract from text blocks
                text = page.extract_text()
                if text:
                    text_requirements = self._extract_from_text_blocks(text)
                    requirements.extend(text_requirements)
        
        return requirements
    
    def _extract_with_pymupdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """Extract requirements using PyMuPDF"""
        requirements = []
        
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text blocks
            text_blocks = page.get_text("dict")
            
            for block in text_blocks.get('blocks', []):
                if 'lines' not in block:
                    continue
                
                # Extract text from block
                block_text = ''
                for line in block['lines']:
                    for span in line.get('spans', []):
                        block_text += span.get('text', '') + ' '
                
                if self.req_pattern.search(block_text):
                    req_matches = self.req_pattern.findall(block_text)
                    for req_id in req_matches:
                        description = self._extract_description_from_block(block_text, req_id)
                        if description:
                            requirements.append({
                                'Request ID': req_id,
                                'Description': self._clean_text(description)
                            })
            
            # Extract from tables
            try:
                tables = page.find_tables()
                self.stats['tables_found'] += len(tables)
                
                for table in tables:
                    table_data = table.extract()
                    
                    for row in table_data:
                        for cell in row:
                            if cell and 'REQ_' in cell:
                                req_matches = self.req_pattern.findall(cell)
                                for req_id in req_matches:
                                    description = self._extract_description_from_cell(cell, req_id)
                                    if description:
                                        requirements.append({
                                            'Request ID': req_id,
                                            'Description': self._clean_text(description)
                                        })
            except Exception as e:
                logger.warning(f"PyMuPDF table extraction failed for page {page_num}: {e}")
        
        doc.close()
        return requirements
    
    def _extract_with_pypdf2(self, pdf_path: str) -> List[Dict[str, str]]:
        """Extract requirements using PyPDF2"""
        requirements = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                
                try:
                    text = page.extract_text()
                    if text:
                        page_requirements = self._extract_from_text_blocks(text)
                        requirements.extend(page_requirements)
                except Exception as e:
                    logger.warning(f"PyPDF2 page {page_num} extraction failed: {e}")
                    self.stats['parsing_errors'] += 1
        
        return requirements
    
    def _extract_from_text_blocks(self, text: str) -> List[Dict[str, str]]:
        """Extract requirements from text blocks"""
        requirements = []
        
        # Clean text first
        text = self._clean_block_text(text)
        lines = text.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for requirement pattern
            if self.req_pattern.search(line):
                req_id = self._extract_req_id(line)
                if req_id:
                    description_lines = []
                    
                    # Get description from same line
                    remaining_text = line.replace(req_id, '').strip()
                    if remaining_text and not remaining_text.startswith(':'):
                        remaining_text = re.sub(r'^[:\-\s]+', '', remaining_text)
                        if remaining_text:
                            description_lines.append(remaining_text)
                    
                    # Look for continuation lines
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j].strip()
                        
                        if not next_line:
                            j += 1
                            continue
                        
                        # Stop if we hit another requirement or section header
                        if (self.req_pattern.search(next_line) or 
                            self._is_section_header(next_line) or
                            self._is_metadata_line(next_line)):
                            break
                        
                        description_lines.append(next_line)
                        j += 1
                    
                    if description_lines:
                        description = ' '.join(description_lines)
                        if len(description.strip()) > 5:  # Minimum description length
                            requirements.append({
                                'Request ID': req_id,
                                'Description': self._clean_text(description)
                            })
                    
                    i = j
                    continue
            
            i += 1
        
        return requirements
    
    def _clean_block_text(self, text: str) -> str:
        """Clean text block and remove artifacts"""
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove page numbers
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\/]', ' ', text)
        
        return text.strip()
    
    def _extract_req_id(self, text: str) -> Optional[str]:
        """Extract requirement ID from text"""
        match = self.req_pattern.search(text)
        return match.group(0) if match else None
    
    def _extract_description_from_cell(self, cell_text: str, req_id: str) -> str:
        """Extract description from table cell"""
        # Remove the requirement ID from the cell text
        description = cell_text.replace(req_id, '').strip()
        
        # Remove leading punctuation
        description = re.sub(r'^[:\-\s]+', '', description)
        
        return description
    
    def _extract_description_from_block(self, block_text: str, req_id: str) -> str:
        """Extract description from text block"""
        # Find the requirement ID and get text after it
        req_index = block_text.find(req_id)
        if req_index == -1:
            return ""
        
        # Get text after the requirement ID
        description = block_text[req_index + len(req_id):].strip()
        
        # Remove leading punctuation
        description = re.sub(r'^[:\-\s]+', '', description)
        
        # Take the first meaningful chunk but don't cut off too early
        sentences = re.split(r'[.!?]\s+', description)
        if sentences and len(sentences[0]) > 20:
            return sentences[0] + '.'
        
        # If first sentence is too short, try to get more context
        if len(sentences) > 1 and len(sentences[0]) < 20:
            combined = sentences[0] + '. ' + sentences[1]
            if len(combined) < 200:
                return combined + '.'
        
        return description
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\/]', ' ', text)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _is_section_header(self, line: str) -> bool:
        """Check if line is a section header"""
        line = line.strip()
        if not line:
            return False
        
        # Check for common section header patterns
        if line.isupper() and len(line) > 3:
            return True
        
        if re.match(r'^\d+(\.\d+)*\s+[A-Z]', line):
            return True
        
        return False
    
    def _is_metadata_line(self, line: str) -> bool:
        """Check if line is metadata"""
        line = line.strip()
        if not line or line.startswith("REQ_"):
            return False
        
        # Check for key:value patterns
        if ':' in line:
            key = line.split(':', 1)[0].strip()
            return ' ' not in key and len(key) <= 20
        
        return False
    
    def _deduplicate_requirements(self, requirements: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove duplicate requirements"""
        seen = set()
        unique_requirements = []
        
        for req in requirements:
            req_id = req.get('Request ID', '')
            if req_id and req_id not in seen:
                seen.add(req_id)
                unique_requirements.append(req)
        
        return unique_requirements
    
    def get_stats(self) -> Dict[str, int]:
        """Get extraction statistics"""
        return self.stats.copy()

def main():
    """Main function to run the extraction"""
    parser = argparse.ArgumentParser(description='Extract requirements from PDF files')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('-o', '--output', default='requirements.csv', help='Output CSV file path')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if input file exists
    if not os.path.exists(args.pdf_path):
        logger.error(f"PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    # Initialize extractor
    extractor = PDFRequirementsExtractor()
    
    try:
        # Extract requirements
        requirements = extractor.extract_requirements(args.pdf_path)
        
        # Write to CSV
        with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Request ID', 'Description']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(requirements)
        
        # Print statistics
        stats = extractor.get_stats()
        logger.info(f"Extraction completed successfully!")
        logger.info(f"Pages processed: {stats['pages_processed']}")
        logger.info(f"Tables found: {stats['tables_found']}")
        logger.info(f"Requirements found: {stats['requirements_found']}")
        logger.info(f"Parsing errors: {stats['parsing_errors']}")
        logger.info(f"Output saved to: {args.output}")
        
        print(f"✅ CSV created at '{args.output}' with {len(requirements)} requirements.")
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()