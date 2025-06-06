"""
Unit tests for image preprocessing functionality.
"""

import unittest
import numpy as np
import cv2
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from image_processing.preprocessing import ImagePreprocessor


class TestImagePreprocessor(unittest.TestCase):
    """Test cases for ImagePreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = ImagePreprocessor()
        
        # Create a simple test image
        self.test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # Add some text-like patterns
        self.test_image[30:70, 30:70] = [255, 255, 255]  # White square
        self.test_image[40:60, 40:60] = [0, 0, 0]        # Black square inside
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        self.assertIsNone(self.preprocessor.original_image)
        self.assertIsNone(self.preprocessor.processed_image)
    
    def test_set_image(self):
        """Test setting image directly."""
        self.preprocessor.set_image(self.test_image)
        self.assertIsNotNone(self.preprocessor.processed_image)
        np.testing.assert_array_equal(self.preprocessor.processed_image, self.test_image)
    
    def test_grayscale_conversion(self):
        """Test grayscale conversion."""
        self.preprocessor.set_image(self.test_image)
        gray_image = self.preprocessor.grayscale()
        
        # Check that result is 2D (grayscale)
        self.assertEqual(len(gray_image.shape), 2)
        self.assertEqual(gray_image.shape, (100, 100))
        
        # Check that values are in valid range
        self.assertTrue(np.all(gray_image >= 0))
        self.assertTrue(np.all(gray_image <= 255))
    
    def test_median_blur(self):
        """Test median blur filtering."""
        # Create grayscale image first
        gray_image = np.ones((100, 100), dtype=np.uint8) * 128
        gray_image[30:70, 30:70] = 255
        gray_image[40:60, 40:60] = 0
        
        self.preprocessor.set_image(gray_image)
        blurred = self.preprocessor.median_blur(kernel_size=3)
        
        # Check output dimensions
        self.assertEqual(blurred.shape, gray_image.shape)
        
        # Check that values are in valid range
        self.assertTrue(np.all(blurred >= 0))
        self.assertTrue(np.all(blurred <= 255))
    
    def test_contrast_stretching(self):
        """Test contrast stretching."""
        # Create image with limited contrast
        low_contrast = np.ones((100, 100), dtype=np.uint8) * 100
        low_contrast[30:70, 30:70] = 150
        
        self.preprocessor.set_image(low_contrast)
        stretched = self.preprocessor.contrast_stretching()
        
        # Check that contrast is improved
        self.assertGreater(np.max(stretched) - np.min(stretched), 
                          np.max(low_contrast) - np.min(low_contrast))
    
    def test_otsu_threshold(self):
        """Test Otsu thresholding."""
        # Create grayscale image
        gray_image = np.ones((100, 100), dtype=np.uint8) * 128
        gray_image[30:70, 30:70] = 255
        gray_image[40:60, 40:60] = 0
        
        self.preprocessor.set_image(gray_image)
        binary = self.preprocessor.otsu_threshold()
        
        # Check that result is binary (only 0 and 255)
        unique_values = np.unique(binary)
        self.assertTrue(len(unique_values) <= 2)
        self.assertTrue(all(val in [0, 255] for val in unique_values))
    
    def test_morphological_opening(self):
        """Test morphological opening."""
        # Create binary image
        binary_image = np.zeros((100, 100), dtype=np.uint8)
        binary_image[30:70, 30:70] = 255
        
        self.preprocessor.set_image(binary_image)
        opened = self.preprocessor.morphological_opening()
        
        # Check output dimensions
        self.assertEqual(opened.shape, binary_image.shape)
        
        # Check that values are binary
        unique_values = np.unique(opened)
        self.assertTrue(all(val in [0, 255] for val in unique_values))
    
    def test_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        self.preprocessor.set_image(self.test_image)
        result = self.preprocessor.preprocess_pipeline()
        
        self.assertIsNotNone(result)
        # Should be grayscale after processing
        self.assertEqual(len(result.shape), 2)
    
    def test_error_handling(self):
        """Test error handling for empty images."""
        with self.assertRaises(ValueError):
            self.preprocessor.grayscale()
        
        with self.assertRaises(ValueError):
            self.preprocessor.median_blur()
        
        with self.assertRaises(ValueError):
            self.preprocessor.contrast_stretching()


class TestImageProcessingEdgeCases(unittest.TestCase):
    """Test edge cases for image processing."""
    
    def setUp(self):
        self.preprocessor = ImagePreprocessor()
    
    def test_single_pixel_image(self):
        """Test processing of single pixel image."""
        single_pixel = np.array([[[128, 128, 128]]], dtype=np.uint8)
        self.preprocessor.set_image(single_pixel)
        
        # Should handle without crashing
        gray = self.preprocessor.grayscale()
        self.assertEqual(gray.shape, (1, 1))
    
    def test_all_black_image(self):
        """Test processing of all black image."""
        black_image = np.zeros((50, 50, 3), dtype=np.uint8)
        self.preprocessor.set_image(black_image)
        
        result = self.preprocessor.preprocess_pipeline()
        self.assertIsNotNone(result)
    
    def test_all_white_image(self):
        """Test processing of all white image."""
        white_image = np.ones((50, 50, 3), dtype=np.uint8) * 255
        self.preprocessor.set_image(white_image)
        
        result = self.preprocessor.preprocess_pipeline()
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
