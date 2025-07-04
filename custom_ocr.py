import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import string
import os
from typing import List, Tuple, Dict, Optional, Union, Any

class CustomOCR:
    """
    A complete OCR library built from scratch in Python.
    No external OCR dependencies - implements all algorithms internally.
    """
    
    def __init__(self):
        """Initialize the OCR engine with all components"""
        self.template_size = (24, 32)  # width, height
        self.templates = {}
        self.min_text_area = 100
        self.max_text_area = 50000
        self.min_char_width = 5
        self.min_char_height = 10
        
        # Generate character templates
        self._generate_character_templates()
    
    def extract_text_from_image(self, image_path: str, 
                               noise_reduction: bool = True,
                               edge_enhancement: bool = True,
                               confidence_threshold: float = 0.5,
                               output_format: str = 'text') -> Union[str, Dict]:
        """
        Extract text from an image file
        
        Args:
            image_path: Path to the image file
            noise_reduction: Apply noise reduction preprocessing
            edge_enhancement: Apply edge enhancement
            confidence_threshold: Minimum confidence for character recognition
            output_format: 'text', 'json', or 'both'
            
        Returns:
            String (text format) or Dictionary (json format) or both
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            result = self._process_image(image, noise_reduction, edge_enhancement, confidence_threshold)
            return self._format_output(result, output_format)
            
        except Exception as e:
            error_result = {
                'text': '',
                'confidence': 0.0,
                'error': str(e),
                'characters_detected': 0
            }
            return self._format_output(error_result, output_format)
    
    def extract_text_from_array(self, image_array: np.ndarray,
                               noise_reduction: bool = True,
                               edge_enhancement: bool = True,
                               confidence_threshold: float = 0.5,
                               output_format: str = 'text') -> Union[str, Dict]:
        """
        Extract text from a numpy image array
        
        Args:
            image_array: Numpy array representing the image
            noise_reduction: Apply noise reduction preprocessing
            edge_enhancement: Apply edge enhancement
            confidence_threshold: Minimum confidence for character recognition
            output_format: 'text', 'json', or 'both'
            
        Returns:
            String (text format) or Dictionary (json format) or both
        """
        result = self._process_image(image_array, noise_reduction, edge_enhancement, confidence_threshold)
        return self._format_output(result, output_format)
    
    def extract_text_from_pdf(self, pdf_path: str,
                             noise_reduction: bool = True,
                             edge_enhancement: bool = True,
                             confidence_threshold: float = 0.5,
                             output_format: str = 'text') -> Union[str, Dict]:
        """
        Extract text from a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            noise_reduction: Apply noise reduction preprocessing
            edge_enhancement: Apply edge enhancement
            confidence_threshold: Minimum confidence for character recognition
            output_format: 'text', 'json', or 'both'
            
        Returns:
            String (text format) or Dictionary (json format) or both
        """
        try:
            import fitz  # PyMuPDF
            
            # Open PDF
            pdf_document = fitz.open(pdf_path)
            all_text = ""
            page_results = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                
                # Convert page to image
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)  # type: ignore
                
                # Convert to numpy array
                img_data = pix.tobytes("ppm")
                import io
                from PIL import Image as PILImage
                img = PILImage.open(io.BytesIO(img_data))
                img_array = np.array(img)
                
                # Process the page
                result = self._process_image(img_array, noise_reduction, edge_enhancement, confidence_threshold)
                
                if result.get('text'):
                    all_text += f"Page {page_num + 1}:\n{result.get('text', '')}\n\n"
                
                page_results.append({
                    'page': page_num + 1,
                    'text': result.get('text', ''),
                    'confidence': result.get('confidence', 0.0)
                })
            
            pdf_document.close()
            
            # Combine results
            combined_result = {
                'text': all_text.strip(),
                'pages': page_results,
                'total_pages': len(pdf_document)
            }
            
            return self._format_output(combined_result, output_format)
            
        except Exception as e:
            error_result = {
                'text': '',
                'error': str(e),
                'pages': []
            }
            return self._format_output(error_result, output_format)
    
    def _process_image(self, image: np.ndarray, noise_reduction: bool, 
                      edge_enhancement: bool, confidence_threshold: float) -> Dict:
        """Core image processing and text extraction"""
        try:
            # Preprocess image
            processed_img = self._preprocess_image(image, noise_reduction, edge_enhancement)
            
            # Detect text regions
            text_regions = self._detect_text_regions(processed_img)
            
            # Extract text from each region
            extracted_text = ""
            total_confidence = 0
            total_chars = 0
            char_details = []
            
            for region in text_regions:
                region_text, region_confidence, region_chars = self._extract_text_from_region(
                    processed_img, region, confidence_threshold
                )
                
                if region_text:
                    extracted_text += region_text + " "
                    total_confidence += region_confidence * len(region_text)
                    total_chars += len(region_text)
                    char_details.extend(region_chars)
            
            # Calculate overall confidence
            avg_confidence = total_confidence / max(total_chars, 1)
            
            return {
                'text': extracted_text.strip(),
                'confidence': avg_confidence,
                'characters_detected': total_chars,
                'text_regions_found': len(text_regions),
                'character_details': char_details
            }
            
        except Exception as e:
            return {
                'text': '',
                'confidence': 0.0,
                'error': str(e),
                'characters_detected': 0
            }
    
    def _preprocess_image(self, image: np.ndarray, noise_reduction: bool, edge_enhancement: bool) -> np.ndarray:
        """Apply preprocessing steps to improve OCR accuracy"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply noise reduction
        if noise_reduction:
            gray = cv2.medianBlur(gray, 3)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Normalize contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Apply edge enhancement
        if edge_enhancement:
            blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
            gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        
        # Binarize the image
        _, otsu_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive_binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
        )
        
        # Combine both thresholding methods
        binary = cv2.bitwise_and(otsu_binary, adaptive_binary)
        
        return binary
    
    def _detect_text_regions(self, binary_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect potential text regions in the image"""
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image, connectivity=8
        )
        
        text_regions = []
        
        # Analyze each component
        for i in range(1, num_labels):  # Skip background
            x, y, w, h, area = stats[i]
            
            # Filter components based on size and aspect ratio
            if self._is_potential_text_region(w, h, area):
                text_regions.append((x, y, w, h))
        
        # Merge nearby regions
        merged_regions = self._merge_nearby_regions(text_regions)
        
        # Sort regions by vertical position (top to bottom)
        merged_regions.sort(key=lambda r: r[1])
        
        return merged_regions
    
    def _is_potential_text_region(self, width: int, height: int, area: int) -> bool:
        """Check if a region could contain text"""
        # Filter by area
        if area < self.min_text_area or area > self.max_text_area:
            return False
        
        # Filter by dimensions
        if width < self.min_char_width or height < self.min_char_height:
            return False
        
        # Filter by aspect ratio
        aspect_ratio = width / height
        if aspect_ratio < 0.1 or aspect_ratio > 20:
            return False
        
        return True
    
    def _merge_nearby_regions(self, regions: List[Tuple[int, int, int, int]], 
                             merge_threshold: int = 20) -> List[Tuple[int, int, int, int]]:
        """Merge nearby regions that likely belong to the same text line"""
        if not regions:
            return []
        
        merged = []
        current_group = [regions[0]]
        
        for i in range(1, len(regions)):
            x, y, w, h = regions[i]
            should_merge = False
            
            for group_region in current_group:
                gx, gy, gw, gh = group_region
                
                # Check vertical overlap and horizontal proximity
                vertical_overlap = max(0, min(y + h, gy + gh) - max(y, gy))
                horizontal_distance = abs(x - (gx + gw))
                
                if vertical_overlap > min(h, gh) * 0.5 and horizontal_distance < merge_threshold:
                    should_merge = True
                    break
            
            if should_merge:
                current_group.append(regions[i])
            else:
                if current_group:
                    merged_region = self._merge_region_group(current_group)
                    merged.append(merged_region)
                current_group = [regions[i]]
        
        if current_group:
            merged_region = self._merge_region_group(current_group)
            merged.append(merged_region)
        
        return merged
    
    def _merge_region_group(self, region_group: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        """Merge a group of regions into a single bounding box"""
        if not region_group:
            return (0, 0, 0, 0)
        
        min_x = min(r[0] for r in region_group)
        min_y = min(r[1] for r in region_group)
        max_x = max(r[0] + r[2] for r in region_group)
        max_y = max(r[1] + r[3] for r in region_group)
        
        return (min_x, min_y, max_x - min_x, max_y - min_y)
    
    def _extract_text_from_region(self, image: np.ndarray, region: Tuple[int, int, int, int], 
                                 confidence_threshold: float) -> Tuple[str, float, List[Dict]]:
        """Extract text from a single text region"""
        x, y, w, h = region
        
        # Extract region from image
        region_img = image[y:y+h, x:x+w]
        
        # Segment characters in the region
        char_segments = self._segment_characters(region_img)
        
        recognized_text = ""
        total_confidence = 0
        char_details = []
        
        for char_segment in char_segments:
            char_x, char_y, char_w, char_h = char_segment
            
            # Extract character image
            char_img = region_img[char_y:char_y+char_h, char_x:char_x+char_w]
            
            # Recognize character
            char, confidence = self._recognize_character(char_img)
            
            if confidence >= confidence_threshold:
                recognized_text += char
                char_details.append({
                    'character': char,
                    'confidence': confidence,
                    'position': (x + char_x, y + char_y),
                    'bbox': (char_x, char_y, char_w, char_h)
                })
                total_confidence += confidence
        
        avg_confidence = total_confidence / max(len(recognized_text), 1)
        
        return recognized_text, avg_confidence, char_details
    
    def _segment_characters(self, text_region_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Segment individual characters within a text region"""
        if text_region_image.size == 0:
            return []
        
        # Find connected components for character segmentation
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            text_region_image, connectivity=8
        )
        
        char_segments = []
        
        for i in range(1, num_labels):  # Skip background
            x, y, w, h, area = stats[i]
            
            # Filter potential characters
            if self._is_potential_character(w, h, area):
                char_segments.append((x, y, w, h))
        
        # Sort characters by horizontal position (left to right)
        char_segments.sort(key=lambda c: c[0])
        
        return char_segments
    
    def _is_potential_character(self, width: int, height: int, area: int) -> bool:
        """Check if a component could be a character"""
        # Size requirements
        if width < self.min_char_width or height < self.min_char_height:
            return False
        
        if width > 100 or height > 150:
            return False
        
        # Aspect ratio check
        aspect_ratio = width / height
        if aspect_ratio > 3.0:
            return False
        
        # Minimum area
        if area < 20:
            return False
        
        return True
    
    def _generate_character_templates(self):
        """Generate templates for character recognition"""
        characters = string.ascii_uppercase + string.ascii_lowercase + string.digits + " .,!?-"
        
        for char in characters:
            template = self._create_character_template(char)
            if template is not None:
                self.templates[char] = template
    
    def _create_character_template(self, char: str) -> Optional[np.ndarray]:
        """Create a template image for a character"""
        try:
            # Create a white image
            img = Image.new('L', self.template_size, color=255)
            draw = ImageDraw.Draw(img)
            
            # Calculate text position to center it
            bbox = draw.textbbox((0, 0), char)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (self.template_size[0] - text_width) // 2
            y = (self.template_size[1] - text_height) // 2 - 2
            
            # Draw the character in black
            draw.text((x, y), char, fill=0)
            
            # Convert to numpy array and binarize
            template_array = np.array(img)
            _, binary_template = cv2.threshold(template_array, 127, 255, cv2.THRESH_BINARY)
            
            # Invert so text is white on black background
            binary_template = 255 - binary_template
            
            return binary_template
            
        except:
            # Fallback: create simple geometric templates
            return self._create_simple_template(char)
    
    def _create_simple_template(self, char: str) -> np.ndarray:
        """Create simple geometric templates for basic characters"""
        template = np.zeros(self.template_size[::-1], dtype=np.uint8)  # height, width
        h, w = template.shape
        
        # Define simple patterns for some characters
        if char == '0' or char == 'O':
            cv2.ellipse(template, (w//2, h//2), (w//3, h//2-2), 0, 0, 360, (255,), 2)
        elif char == '1' or char == 'I':
            cv2.line(template, (w//2, 2), (w//2, h-2), (255,), 2)
        elif char == '-':
            cv2.line(template, (w//4, h//2), (3*w//4, h//2), (255,), 2)
        elif char == '.':
            cv2.circle(template, (w//2, h-4), 2, (255,), -1)
        elif char == ' ':
            # Space character - empty template
            pass
        else:
            # For other characters, create a simple rectangular pattern
            cv2.rectangle(template, (w//4, h//4), (3*w//4, 3*h//4), (255,), 1)
        
        return template
    
    def _recognize_character(self, char_image: np.ndarray) -> Tuple[str, float]:
        """Recognize a character using template matching"""
        if char_image.size == 0:
            return ' ', 0.0
        
        # Preprocess the character image
        processed_char = self._preprocess_character(char_image)
        
        best_match = ' '
        best_score = 0.0
        
        # Try template matching for each character
        for char, template in self.templates.items():
            score = self._match_template(processed_char, template)
            
            if score > best_score:
                best_score = score
                best_match = char
        
        # Calculate confidence
        confidence = self._calculate_confidence(best_score, processed_char)
        
        return best_match, confidence
    
    def _preprocess_character(self, char_image: np.ndarray) -> np.ndarray:
        """Preprocess character image for template matching"""
        # Ensure grayscale
        if len(char_image.shape) == 3:
            char_image = cv2.cvtColor(char_image, cv2.COLOR_RGB2GRAY)
        
        # Binarize if not already binary
        if char_image.max() > 1:
            _, char_image = cv2.threshold(char_image, 127, 255, cv2.THRESH_BINARY)
        
        # Remove noise
        char_image = cv2.medianBlur(char_image, 3)
        
        # Resize to template size
        char_resized = cv2.resize(char_image, self.template_size, interpolation=cv2.INTER_AREA)
        
        # Ensure text is white on black background
        if np.mean(char_resized) > 127:
            char_resized = 255 - char_resized
        
        return char_resized
    
    def _match_template(self, char_image: np.ndarray, template: np.ndarray) -> float:
        """Perform template matching between character and template"""
        try:
            # Ensure both images are the same size
            if char_image.shape != template.shape:
                char_image = cv2.resize(char_image, template.shape[::-1])
            
            # Normalize images
            char_norm = char_image.astype(np.float32) / 255.0
            template_norm = template.astype(np.float32) / 255.0
            
            # Calculate normalized cross-correlation
            correlation = cv2.matchTemplate(char_norm, template_norm, cv2.TM_CCORR_NORMED)
            
            # Get maximum correlation value
            _, max_val, _, _ = cv2.minMaxLoc(correlation)
            
            return max_val
            
        except:
            return 0.0
    
    def _calculate_confidence(self, raw_score: float, char_image: np.ndarray) -> float:
        """Calculate confidence score based on matching score and character quality"""
        confidence = raw_score
        
        # Adjust based on character image quality
        if char_image.size > 0:
            # Check foreground pixel ratio
            foreground_ratio = np.sum(char_image > 127) / char_image.size
            
            # Penalize very sparse or very dense characters
            if foreground_ratio < 0.1 or foreground_ratio > 0.8:
                confidence *= 0.7
            
            # Check aspect ratio
            h, w = char_image.shape
            aspect_ratio = w / h
            if aspect_ratio < 0.2 or aspect_ratio > 3.0:
                confidence *= 0.8
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _format_output(self, result: Dict, output_format: str) -> Union[str, Dict]:
        """Format the output according to the specified format"""
        import json
        
        if output_format == 'text':
            return result.get('text', '')
        elif output_format == 'json':
            if 'pages' in result:
                # PDF format - return only text content
                return {
                    'text': result.get('text', ''),
                    'pages': [{'page': p['page'], 'text': p['text']} for p in result.get('pages', [])]
                }
            else:
                # Single image format - return only text content
                return {'text': result.get('text', '')}
        elif output_format == 'both':
            text_output = result.get('text', '')
            if 'pages' in result:
                json_output = {
                    'text': result.get('text', ''),
                    'pages': [{'page': p['page'], 'text': p['text']} for p in result.get('pages', [])]
                }
            else:
                json_output = {'text': result.get('text', '')}
            
            return {
                'text_format': text_output,
                'json_format': json_output
            }
        else:
            return result.get('text', '')
    
    def get_template_count(self) -> int:
        """Get the number of character templates loaded"""
        return len(self.templates)
    
    def get_supported_characters(self) -> List[str]:
        """Get list of characters that can be recognized"""
        return list(self.templates.keys())


# Example usage
if __name__ == "__main__":
    # Simple usage example
    ocr = CustomOCR()
    print(f"OCR initialized with {ocr.get_template_count()} character templates")
    print("Ready to process images and PDFs")
    print("Usage:")
    print("  text = ocr.extract_text_from_image('image.png', output_format='text')")
    print("  result = ocr.extract_text_from_image('image.png', output_format='json')")
    print("  pdf_result = ocr.extract_text_from_pdf('document.pdf', output_format='json')")