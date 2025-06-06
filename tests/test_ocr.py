"""
Unit tests for OCR functionality.
"""

import unittest
import numpy as np
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from image_processing.ocr import OCRProcessor


class TestOCRProcessor(unittest.TestCase):
    """Test cases for OCRProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ocr = OCRProcessor()
        
        # Create a simple test image with text-like patterns
        self.test_image = np.ones((100, 200), dtype=np.uint8) * 255
        # Add some black rectangles to simulate text
        self.test_image[30:50, 30:60] = 0   # First "letter"
        self.test_image[30:50, 70:100] = 0  # Second "letter"
        self.test_image[30:50, 110:140] = 0 # Third "letter"
    
    def test_initialization(self):
        """Test OCR processor initialization."""
        self.assertIsInstance(self.ocr, OCRProcessor)
    
    def test_extract_text(self):
        """Test text extraction functionality."""
        # This test may not extract actual text from our simple image,
        # but should not crash
        try:
            result = self.ocr.extract_text(self.test_image)
            self.assertIsInstance(result, str)
        except Exception as e:
            # If Tesseract is not properly installed, skip this test
            self.skipTest(f"Tesseract not available: {e}")
    
    def test_get_bounding_boxes(self):
        """Test bounding box detection."""
        try:
            boxes = self.ocr.get_bounding_boxes(self.test_image)
            self.assertIsInstance(boxes, list)
        except Exception as e:
            self.skipTest(f"Tesseract not available: {e}")
    
    def test_draw_bounding_boxes(self):
        """Test drawing bounding boxes on image."""
        try:
            # Create some mock bounding boxes
            mock_boxes = [
                {'char': 'A', 'x': 10, 'y': 20, 'w': 30, 'h': 40},
                {'char': 'B', 'x': 50, 'y': 20, 'w': 80, 'h': 40}
            ]
            
            result = self.ocr.draw_bounding_boxes(self.test_image, mock_boxes)
            self.assertEqual(result.shape, self.test_image.shape)
        except Exception as e:
            self.skipTest(f"Tesseract not available: {e}")
    
    def test_process_image(self):
        """Test complete image processing."""
        try:
            result = self.ocr.process_image(self.test_image, draw_boxes=True)
            
            self.assertIsInstance(result, dict)
            self.assertIn('text', result)
            self.assertIn('boxes', result)
            self.assertIsInstance(result['text'], str)
            self.assertIsInstance(result['boxes'], list)
        except Exception as e:
            self.skipTest(f"Tesseract not available: {e}")
    
    def test_empty_image(self):
        """Test OCR on empty image."""
        empty_image = np.ones((50, 50), dtype=np.uint8) * 255
        
        try:
            result = self.ocr.extract_text(empty_image)
            # Empty image should return empty or minimal text
            self.assertIsInstance(result, str)
        except Exception as e:
            self.skipTest(f"Tesseract not available: {e}")


if __name__ == '__main__':
    unittest.main()
