"""
Alphabetic Recognition Module
============================

Module untuk integrasi alphabetic recognition dengan aplikasi utama.
Menggunakan model yang sudah di-train untuk klasifikasi karakter A-Z, 0-9.
"""

import cv2
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import sys
import os
import warnings
import time
import threading
from functools import lru_cache
import multiprocessing as mp

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import centralized configuration
from .config import (
    MODEL_PATH, CHAR_IMAGE_SIZE, HOG_PARAMS, FEATURE_WEIGHTS,
    MIN_CONFIDENCE_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD,
    GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA_X, GAUSSIAN_SIGMA_Y,
    MORPH_OPEN_KERNEL, MORPH_CLOSE_KERNEL, MORPH_ITERATIONS,
    THRESHOLD_TYPE, THRESHOLD_MAX_VALUE, MEDIAN_FILTER_SIZE,
    RESIZE_INTERPOLATION, CONTOUR_RETRIEVAL, CONTOUR_APPROXIMATION,
    get_preprocessing_config, get_feature_extraction_config,
    get_classification_config, validate_config,
    USE_MULTIPROCESSING, MAX_WORKERS, BATCH_SIZE, FEATURE_CACHE_SIZE,
    PRELOAD_MODEL, LAZY_LOADING
)

class AlphabeticRecognizer:
    """
    Enhanced alphabetic recognizer with centralized configuration, advanced performance optimization,
    and comprehensive error handling. Supports feature caching and batch processing.
    """
    
    def __init__(self, model_path: Optional[str] = None, config_validation: bool = True):
        """
        Initialize alphabetic recognizer with enhanced configuration and performance optimization.
        
        Args:
            model_path: Path to trained model (uses default if None)
            config_validation: Whether to validate configuration on startup
        """
        # Validate configuration if requested
        if config_validation:
            self._validate_configuration()
        
        # Use centralized model path if not specified
        self.model_path = model_path or str(MODEL_PATH)
        
        # Model components
        self.model_data = None
        self.classifier = None
        self.classes = None
        self.feature_size = None
        self.is_loaded = False
        
        # Performance tracking
        self.prediction_count = 0
        self.total_confidence = 0.0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Performance optimization
        self.feature_cache = {}
        self.max_cache_size = FEATURE_CACHE_SIZE
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Load configuration
        self.preprocessing_config = get_preprocessing_config()
        self.feature_config = get_feature_extraction_config()
        self.classification_config = get_classification_config()
        
        # Load model
        if PRELOAD_MODEL:
            self.load_model()
    
    def _validate_configuration(self) -> None:
        """Validate the current configuration and log warnings if needed."""
        try:
            validate_config()
        except Exception as e:
            warnings.warn(f"Configuration validation warning: {e}", UserWarning)
    
    def _log_performance_metrics(self) -> Dict[str, float]:
        """Calculate and return performance metrics."""
        if self.prediction_count == 0:
            return {"avg_confidence": 0.0, "prediction_count": 0}
        
        avg_confidence = self.total_confidence / self.prediction_count
        return {
            "avg_confidence": round(avg_confidence, 3),
            "prediction_count": self.prediction_count
        }
    
    def load_model(self) -> bool:
        """
        Load the trained model with enhanced error handling.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = Path(self.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load model data with error checking
            self.model_data = joblib.load(self.model_path)
            
            # Validate model data structure
            required_keys = ['model', 'classes', 'feature_size']
            missing_keys = [key for key in required_keys if key not in self.model_data]
            if missing_keys:
                raise ValueError(f"Model data missing required keys: {missing_keys}")
            
            self.classifier = self.model_data['model']
            self.classes = self.model_data['classes']
            self.feature_size = self.model_data['feature_size']
            
            # Validate model capabilities
            if not hasattr(self.classifier, 'predict'):
                raise ValueError("Loaded model does not have predict method")
            
            self.is_loaded = True
            
            # Log successful loading with additional info
            model_type = self.model_data.get('model_type', 'Unknown')
            print(f"✓ Alphabetic recognition model loaded successfully")
            print(f"  Model type: {model_type}")
            print(f"  Classes: {len(self.classes)} ({', '.join(map(str, self.classes[:10]))}{'...' if len(self.classes) > 10 else ''})")
            print(f"  Feature size: {self.feature_size}")
            
            return True
            
        except FileNotFoundError as e:
            print(f"❌ Model file error: {e}")
            self.is_loaded = False
            return False
        except ValueError as e:
            print(f"❌ Model validation error: {e}")
            self.is_loaded = False
            return False
        except Exception as e:
            print(f"❌ Unexpected error loading model: {e}")
            self.is_loaded = False
            return False
    
    def preprocess_character_image(self, image_roi: np.ndarray, 
                                  target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Enhanced character preprocessing using centralized configuration.
        
        Args:
            image_roi: ROI image containing character
            target_size: Target size for normalization (uses config default if None)
            
        Returns:
            Preprocessed binary image
            
        Raises:
            ValueError: If image_roi is invalid
        """
        if image_roi is None or image_roi.size == 0:
            raise ValueError("Invalid image ROI provided")
        
        # Use configured target size if not specified
        if target_size is None:
            target_size = CHAR_IMAGE_SIZE
        
        try:
            # Convert to grayscale if needed
            if len(image_roi.shape) == 3:
                gray_image = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image_roi.copy()
            
            # Enhanced noise reduction using configuration
            denoised = cv2.GaussianBlur(
                gray_image, 
                GAUSSIAN_KERNEL_SIZE, 
                GAUSSIAN_SIGMA_X, 
                sigmaY=GAUSSIAN_SIGMA_Y
            )
            denoised = cv2.medianBlur(denoised, MEDIAN_FILTER_SIZE)
            
            # Improved binarization
            _, binary_image = cv2.threshold(
                denoised, 
                0, 
                THRESHOLD_MAX_VALUE, 
                THRESHOLD_TYPE
            )
            
            # Morphological operations using configured kernels
            cleaned = cv2.morphologyEx(
                binary_image, 
                cv2.MORPH_OPEN, 
                MORPH_OPEN_KERNEL, 
                iterations=MORPH_ITERATIONS
            )
            cleaned = cv2.morphologyEx(
                cleaned, 
                cv2.MORPH_CLOSE, 
                MORPH_CLOSE_KERNEL, 
                iterations=MORPH_ITERATIONS
            )
            
            # Resize to target size with better interpolation
            normalized = cv2.resize(
                cleaned, 
                target_size, 
                interpolation=cv2.INTER_AREA
            )
            
            # Ensure proper orientation (background black, foreground white)
            white_pixels = np.count_nonzero(normalized == 255)
            total_pixels = normalized.shape[0] * normalized.shape[1]
            
            if white_pixels > total_pixels * 0.5:
                normalized = cv2.bitwise_not(normalized)
            
            return normalized
            
        except cv2.error as e:
            raise ValueError(f"OpenCV error in preprocessing: {e}")
        except Exception as e:
            # Enhanced fallback preprocessing
            print(f"Warning: Preprocessing error, using fallback: {e}")
            return self._fallback_preprocessing(image_roi, target_size)
    
    def _fallback_preprocessing(self, image_roi: np.ndarray, 
                               target_size: Tuple[int, int]) -> np.ndarray:
        """Fallback preprocessing with minimal operations."""
        try:
            if len(image_roi.shape) == 3:
                gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_roi.copy()
            
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            resized = cv2.resize(binary, target_size, interpolation=cv2.INTER_AREA)
            return resized
        except Exception:
            # Ultimate fallback - return zeros
            return np.zeros(target_size, dtype=np.uint8)
    
    def extract_features_from_character(self, binary_char_image: np.ndarray) -> np.ndarray:
        """
        Enhanced feature extraction with caching and optimization.
        
        Args:
            binary_char_image: Preprocessed binary character image
            
        Returns:
            Weighted feature vector
        """
        start_time = time.time()
        
        # Generate image hash for caching
        image_hash = self._generate_image_hash(binary_char_image)
        
        # Check cache first
        cached_features = self._get_cached_features(image_hash)
        if cached_features is not None:
            return cached_features
        
        try:
            # Initialize feature list
            features = []
            weights = FEATURE_WEIGHTS
            
            # 1. Hu Moments (7 features) - Shape descriptors
            try:
                moments = cv2.moments(binary_char_image)
                if moments['m00'] != 0:
                    hu_moments = cv2.HuMoments(moments).flatten()
                    hu_moments = np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
                else:
                    hu_moments = np.zeros(7)
                features.extend(hu_moments * weights['hu_moments'])
            except Exception:
                features.extend(np.zeros(7) * weights['hu_moments'])
            
            # 2. HOG Features (~36 features) - Optimized with cached descriptor
            try:
                if binary_char_image.shape[:2] != CHAR_IMAGE_SIZE:
                    resized_img = cv2.resize(binary_char_image, CHAR_IMAGE_SIZE, 
                                           interpolation=RESIZE_INTERPOLATION)
                else:
                    resized_img = binary_char_image
                
                hog_descriptor = self._get_optimized_hog_descriptor()
                hog_features = hog_descriptor.compute(resized_img)
                if hog_features is not None:
                    hog_features = hog_features.flatten()
                else:
                    hog_features = np.zeros(36)  # Fallback size
                features.extend(hog_features * weights['hog'])
            except Exception:
                features.extend(np.zeros(36) * weights['hog'])
            
            # 3. Geometric Features (5 features) - Optimized contour processing
            try:
                contours, _ = cv2.findContours(binary_char_image, CONTOUR_RETRIEVAL, 
                                             CONTOUR_APPROXIMATION)
                if contours:
                    # Use largest contour
                    max_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(max_contour)
                    perimeter = cv2.arcLength(max_contour, True)
                    
                    if area > 0:
                        x, y, w, h = cv2.boundingRect(max_contour)
                        aspect_ratio = float(w) / h
                        area_ratio = area / (w * h)
                        
                        if perimeter > 0:
                            perimeter_area_ratio = perimeter / area
                        else:
                            perimeter_area_ratio = 0
                        
                        # Solidity
                        hull = cv2.convexHull(max_contour)
                        hull_area = cv2.contourArea(hull)
                        if hull_area > 0:
                            solidity = area / hull_area
                        else:
                            solidity = 0
                        
                        # Extent
                        extent = area / (w * h)
                        
                        geometric_features = [aspect_ratio, area_ratio, perimeter_area_ratio, 
                                            solidity, extent]
                    else:
                        geometric_features = [1.0, 0.0, 0.0, 0.0, 0.0]
                else:
                    geometric_features = [1.0, 0.0, 0.0, 0.0, 0.0]
                features.extend(np.array(geometric_features) * weights['geometric'])
            except Exception:
                features.extend(np.zeros(5) * weights['geometric'])
            
            # 4. Projection Features (6 features) - Optimized computation
            try:
                height, width = binary_char_image.shape
                if height > 0 and width > 0:
                    # Horizontal projection
                    h_proj = np.sum(binary_char_image, axis=1)
                    h_mean = np.mean(h_proj) if len(h_proj) > 0 else 0
                    h_std = np.std(h_proj) if len(h_proj) > 0 else 0
                    h_max = np.max(h_proj) if len(h_proj) > 0 else 0
                    
                    # Vertical projection
                    v_proj = np.sum(binary_char_image, axis=0)
                    v_mean = np.mean(v_proj) if len(v_proj) > 0 else 0
                    v_std = np.std(v_proj) if len(v_proj) > 0 else 0
                    v_max = np.max(v_proj) if len(v_proj) > 0 else 0
                    
                    projection_features = [h_mean, h_std, h_max, v_mean, v_std, v_max]
                else:
                    projection_features = [0, 0, 0, 0, 0, 0]
                features.extend(np.array(projection_features) * weights['projection'])
            except Exception:
                features.extend(np.zeros(6) * weights['projection'])
            
            # 5. Zoning Features (16 features) - 4x4 grid optimization
            try:
                height, width = binary_char_image.shape
                if height >= 4 and width >= 4:
                    zone_h, zone_w = height // 4, width // 4
                    zoning_features = []
                    for i in range(4):
                        for j in range(4):
                            zone = binary_char_image[i*zone_h:(i+1)*zone_h, j*zone_w:(j+1)*zone_w]
                            if zone.size > 0:
                                density = np.sum(zone) / (255 * zone.size)
                            else:
                                density = 0
                            zoning_features.append(density)
                else:
                    zoning_features = [0] * 16
                features.extend(np.array(zoning_features) * weights['zoning'])
            except Exception:
                features.extend(np.zeros(16) * weights['zoning'])
            
            # 6. Crossing Features (2 features) - Optimized crossing counts
            try:
                height, width = binary_char_image.shape
                if height > 1 and width > 1:
                    # Horizontal crossings (middle row)
                    mid_row = height // 2
                    h_crossings = 0
                    row = binary_char_image[mid_row, :]
                    for k in range(1, len(row)):
                        if row[k] != row[k-1]:
                            h_crossings += 1
                    
                    # Vertical crossings (middle column)
                    mid_col = width // 2
                    v_crossings = 0
                    col = binary_char_image[:, mid_col]
                    for k in range(1, len(col)):
                        if col[k] != col[k-1]:
                            v_crossings += 1
                    
                    crossing_features = [h_crossings, v_crossings]
                else:
                    crossing_features = [0, 0]
                features.extend(np.array(crossing_features) * weights['crossing'])
            except Exception:
                features.extend(np.zeros(2) * weights['crossing'])
            
            # Convert to numpy array
            feature_vector = np.array(features, dtype=np.float32)
            
            # Cache the computed features
            self._cache_feature_vector(image_hash, feature_vector)
            
            # Update timing
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            return feature_vector
            
        except Exception as e:
            warnings.warn(f"Feature extraction failed: {e}", UserWarning)
            # Return fallback feature vector
            fallback_size = sum([
                7 * FEATURE_WEIGHTS['hu_moments'],
                36 * FEATURE_WEIGHTS['hog'], 
                5 * FEATURE_WEIGHTS['geometric'],
                6 * FEATURE_WEIGHTS['projection'],
                16 * FEATURE_WEIGHTS['zoning'],
                2 * FEATURE_WEIGHTS['crossing']
            ])
            return np.zeros(int(fallback_size), dtype=np.float32)
    
    def predict_character(self, image_roi: np.ndarray, 
                         return_probabilities: bool = False) -> Tuple[str, float]:
        """
        Enhanced character prediction with confidence scoring and validation.
        
        Args:
            image_roi: ROI image containing character
            return_probabilities: Whether to return full probability distribution
            
        Returns:
            Tuple (predicted_character, confidence_score)
            
        Raises:
            RuntimeError: If model is not loaded
            ValueError: If image_roi is invalid
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if image_roi is None or image_roi.size == 0:
            raise ValueError("Invalid image ROI provided")
        
        try:
            # Preprocess image
            preprocessed = self.preprocess_character_image(image_roi)
            
            # Extract features
            features = self.extract_features_from_character(preprocessed)
            
            # Validate features
            if features.size != self.feature_size:
                print(f"Warning: Feature size mismatch. Expected {self.feature_size}, got {features.size}")
                # Resize features if needed
                if features.size < self.feature_size:
                    padded_features = np.zeros(self.feature_size, dtype=np.float32)
                    padded_features[:features.size] = features
                    features = padded_features
                else:
                    features = features[:self.feature_size]
            
            # Make prediction
            prediction = self.classifier.predict([features])[0]
            
            # Calculate confidence score
            confidence = self._calculate_confidence(features)
            
            # Update performance tracking
            self.prediction_count += 1
            self.total_confidence += confidence
            
            # Apply confidence thresholds
            if confidence < MIN_CONFIDENCE_THRESHOLD:
                prediction = "?"
                confidence = 0.0
            
            return str(prediction), float(confidence)
            
        except Exception as e:
            print(f"Error in character prediction: {e}")
            return "?", 0.0
    
    def _calculate_confidence(self, features: np.ndarray) -> float:
        """
        Calculate confidence score using multiple methods.
        
        Args:
            features: Feature vector
            
        Returns:
            Normalized confidence score (0.0 to 1.0)
        """
        try:
            confidence = 0.0
            
            if hasattr(self.classifier, 'predict_proba'):
                # Probabilistic classifier (Random Forest, etc.)
                probabilities = self.classifier.predict_proba([features])[0]
                confidence = float(np.max(probabilities))
                
            elif hasattr(self.classifier, 'decision_function'):
                # SVM with decision function
                decision_scores = self.classifier.decision_function([features])
                if len(self.classes) == 2:
                    # Binary classification
                    confidence = float(abs(decision_scores[0]))
                else:
                    # Multi-class classification
                    decision_scores = decision_scores[0]
                    max_score = np.max(decision_scores)
                    second_max = np.partition(decision_scores, -2)[-2]
                    confidence = float(max_score - second_max)  # Margin-based confidence
                
                # Normalize SVM confidence to [0, 1]
                confidence = min(1.0, max(0.0, confidence / 2.0))
                
            else:
                # Fallback: use basic prediction confidence
                confidence = 0.5
            
            return confidence
            
        except Exception as e:
            print(f"Error calculating confidence: {e}")
            return 0.0
    
    def predict_characters_from_image(self, image: np.ndarray, 
                                     bboxes: List[Tuple[int, int, int, int]], 
                                     min_confidence: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Enhanced batch character prediction with detailed results.
        
        Args:
            image: Original image
            bboxes: List of bounding boxes [(x, y, w, h), ...]
            min_confidence: Minimum confidence threshold (uses config default if None)
            
        Returns:
            List of prediction dictionaries with character, confidence, bbox, etc.
        """
        if min_confidence is None:
            min_confidence = MIN_CONFIDENCE_THRESHOLD
        
        results = []
        
        for i, bbox in enumerate(bboxes):
            try:
                x, y, w, h = bbox
                
                # Validate bounding box
                if x < 0 or y < 0 or w <= 0 or h <= 0:
                    results.append({
                        'bbox_id': i,
                        'bbox': bbox,
                        'character': '?',
                        'confidence': 0.0,
                        'status': 'invalid_bbox'
                    })
                    continue
                
                # Extract ROI with bounds checking
                roi = image[max(0, y):min(image.shape[0], y+h), 
                           max(0, x):min(image.shape[1], x+w)]
                
                if roi.size == 0:
                    results.append({
                        'bbox_id': i,
                        'bbox': bbox,
                        'character': '?',
                        'confidence': 0.0,
                        'status': 'empty_roi'
                    })
                    continue
                
                # Predict character
                char, conf = self.predict_character(roi)
                
                # Determine status
                status = 'success'
                if conf < min_confidence:
                    status = 'low_confidence'
                elif conf < HIGH_CONFIDENCE_THRESHOLD:
                    status = 'uncertain'
                else:
                    status = 'high_confidence'
                
                results.append({
                    'bbox_id': i,
                    'bbox': bbox,
                    'character': char,
                    'confidence': conf,
                    'status': status
                })
                
            except Exception as e:
                results.append({
                    'bbox_id': i,
                    'bbox': bbox,
                    'character': '?',
                    'confidence': 0.0,
                    'status': f'error: {str(e)}'
                })
        
        return results
    
    def clear_cache(self) -> None:
        """Clear feature cache to free memory."""
        with self._lock:
            self.feature_cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
    
    def reset_performance_tracking(self) -> None:
        """Reset all performance tracking counters."""
        self.prediction_count = 0
        self.total_confidence = 0.0
        self.total_processing_time = 0.0
        self.clear_cache()
    
    def _cache_feature_vector(self, image_hash: str, features: np.ndarray) -> None:
        """
        Cache feature vector for faster repeated processing.
        
        Args:
            image_hash: Hash of preprocessed image
            features: Extracted feature vector
        """
        with self._lock:
            if len(self.feature_cache) >= self.max_cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.feature_cache))
                del self.feature_cache[oldest_key]
            
            self.feature_cache[image_hash] = features.copy()
    
    def _get_cached_features(self, image_hash: str) -> Optional[np.ndarray]:
        """
        Get cached feature vector if available.
        
        Args:
            image_hash: Hash of preprocessed image
            
        Returns:
            Cached feature vector or None
        """
        with self._lock:
            if image_hash in self.feature_cache:
                self.cache_hits += 1
                return self.feature_cache[image_hash].copy()
            else:
                self.cache_misses += 1
                return None
    
    def _generate_image_hash(self, image: np.ndarray) -> str:
        """
        Generate hash for image caching.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Image hash string
        """
        return str(hash(image.tobytes()))
    
    @lru_cache(maxsize=128)
    def _get_optimized_hog_descriptor(self) -> cv2.HOGDescriptor:
        """
        Get cached HOG descriptor for better performance.
        
        Returns:
            Configured HOG descriptor
        """
        return cv2.HOGDescriptor(
            _winSize=(CHAR_IMAGE_SIZE[0], CHAR_IMAGE_SIZE[1]),
            _blockSize=HOG_PARAMS['_blockSize'],
            _blockStride=HOG_PARAMS['_blockStride'],
            _cellSize=HOG_PARAMS['_cellSize'],
            _nbins=HOG_PARAMS['_nbins'],
            _derivAperture=HOG_PARAMS['_derivAperture'],
            _winSigma=HOG_PARAMS['_winSigma'],
            _histogramNormType=HOG_PARAMS['_histogramNormType'],
            _L2HysThreshold=HOG_PARAMS['_L2HysThreshold'],
            _gammaCorrection=HOG_PARAMS['_gammaCorrection'],
            _nlevels=HOG_PARAMS['_nlevels'],
            _signedGradient=HOG_PARAMS['_signedGradient']
        )


# Global instance untuk digunakan dalam aplikasi
_recognizer_instance = None

def get_alphabetic_recognizer() -> AlphabeticRecognizer:
    """
    Get singleton instance dari AlphabeticRecognizer.
    
    Returns:
        AlphabeticRecognizer instance
    """
    global _recognizer_instance
    if _recognizer_instance is None:
        _recognizer_instance = AlphabeticRecognizer()
    return _recognizer_instance

def predict_character(image_roi: np.ndarray) -> Tuple[str, float]:
    """
    Helper function untuk predict single character.
    
    Args:
        image_roi: ROI image yang berisi karakter
        
    Returns:
        Tuple (character, confidence)
    """
    recognizer = get_alphabetic_recognizer()
    return recognizer.predict_character(image_roi)

def predict_characters(image: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> List[Tuple[str, float]]:
    """
    Helper function untuk predict multiple characters.
    
    Args:
        image: Original image
        bboxes: List of bounding boxes
        
    Returns:
        List of (character, confidence) tuples
    """
    recognizer = get_alphabetic_recognizer()
    results = recognizer.predict_characters_from_image(image, bboxes)
    # Convert to simple tuple format for backward compatibility
    return [(result['character'], result['confidence']) for result in results]

# Test function
if __name__ == "__main__":
    print("Testing Alphabetic Recognition Module...")
    
    # Test dengan dummy image
    dummy_image = np.ones((50, 50, 3), dtype=np.uint8) * 255
    cv2.rectangle(dummy_image, (10, 10), (40, 40), (0, 0, 0), 2)
    
    char, conf = predict_character(dummy_image)
    print(f"Prediction: {char}, Confidence: {conf:.4f}")
    
    # Get model info
    recognizer = get_alphabetic_recognizer()
    print(f"Model loaded: {recognizer.is_loaded}")
    print(f"Model path: {recognizer.model_path}")
