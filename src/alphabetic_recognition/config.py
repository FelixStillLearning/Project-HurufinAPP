"""
Configuration file for Alphabetic Recognition Module
===================================================

Centralized configuration for all paths, parameters, and settings
used in alphabetic recognition system.
"""

import cv2
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_DIR = PROJECT_ROOT / "dataset"

# ============================================================================
# PATH CONFIGURATIONS
# ============================================================================

# Model paths
MODEL_PATH = MODELS_DIR / "alphabetic_classifier_model.pkl"
FEATURE_CONFIG_PATH = MODELS_DIR / "feature_extractor_config.pkl"

# Dataset paths
DATASET_PATH = DATASET_DIR / "alphabets"
IMAGES_PATH = DATASET_PATH / "images"
ANNOTATIONS_PATH = DATASET_PATH / "annotations"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
DATASET_DIR.mkdir(exist_ok=True)
DATASET_PATH.mkdir(exist_ok=True)
IMAGES_PATH.mkdir(exist_ok=True)
ANNOTATIONS_PATH.mkdir(exist_ok=True)

# ============================================================================
# IMAGE PREPROCESSING PARAMETERS
# ============================================================================

# Character image normalization size
CHAR_IMAGE_SIZE = (28, 28)  # Standard size for character recognition
CHAR_WIDTH, CHAR_HEIGHT = CHAR_IMAGE_SIZE

# Thresholding parameters
THRESHOLD_TYPE = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
THRESHOLD_MAX_VALUE = 255

# Gaussian blur for noise reduction
GAUSSIAN_KERNEL_SIZE = (3, 3)
GAUSSIAN_SIGMA_X = 1
GAUSSIAN_SIGMA_Y = 1

# Noise reduction parameters
MEDIAN_FILTER_SIZE = 3          # Standard 3x3 median filter

# Morphological operations - optimized for character cleanup
MORPH_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
MORPH_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
MORPH_ITERATIONS = 1            # Single iteration to preserve character details

# ============================================================================
# OPTIMIZED FEATURE EXTRACTION PARAMETERS
# ============================================================================

# HOG (Histogram of Oriented Gradients) parameters - Optimized for character recognition
HOG_PARAMS = {
    '_winSize': CHAR_IMAGE_SIZE,           # Window size (28, 28)
    '_blockSize': (14, 14),                # Block size - optimal for 28x28 images
    '_blockStride': (7, 7),                # Block stride - 50% overlap for better feature capture
    '_cellSize': (7, 7),                   # Cell size - 4x4 cells per image
    '_nbins': 9,                           # Number of orientation bins
    '_derivAperture': 1,                   # Sobel kernel size
    '_winSigma': -1,                       # Auto sigma calculation
    '_histogramNormType': 0,               # L2-Hys normalization (most robust)
    '_L2HysThreshold': 0.2,                # Clipping threshold
    '_gammaCorrection': True,              # Gamma correction for better contrast
    '_nlevels': 64,                        # Number of pyramid levels
    '_signedGradient': False               # Unsigned gradients (standard for OCR)
}

# Feature extraction weights (for ensemble feature vectors)
FEATURE_WEIGHTS = {
    'hu_moments': 1.0,          # Shape descriptors
    'hog': 2.0,                 # Most important for character recognition
    'geometric': 1.5,           # Important structural features
    'projection': 1.2,          # Useful for character discrimination
    'zoning': 1.3,              # Spatial density information
    'crossing': 0.8             # Less critical but still useful
}

# ============================================================================
# CONTOUR FILTERING PARAMETERS (used in validation)
# ============================================================================

# Area filters - used in validate_config()
MIN_CONTOUR_AREA = 30           # Minimum contour area for validation
# Aspect ratio filters - used in validate_config()
MIN_ASPECT_RATIO = 0.1          # Minimum aspect ratio for validation
MAX_ASPECT_RATIO = 4.0          # Maximum aspect ratio for validation

# ============================================================================
# CLASSIFICATION PARAMETERS
# ============================================================================

# Confidence thresholds
MIN_CONFIDENCE_THRESHOLD = 0.15  # Reduced minimum confidence for more detections
HIGH_CONFIDENCE_THRESHOLD = 0.7  # High confidence threshold

# Model parameters
SVM_C_PARAM = 1.0               # SVM regularization parameter
SVM_GAMMA = 'scale'             # SVM gamma parameter
RF_N_ESTIMATORS = 150           # Random Forest trees (increased from 100)
RF_MAX_DEPTH = 20               # Maximum tree depth
RF_MIN_SAMPLES_SPLIT = 5        # Minimum samples to split
RF_MIN_SAMPLES_LEAF = 2         # Minimum samples per leaf

# Cross-validation parameters
CV_FOLDS = 5                    # Number of folds for cross-validation
RANDOM_STATE = 42               # Fixed random state for reproducibility

# ============================================================================
# PERFORMANCE OPTIMIZATION PARAMETERS
# ============================================================================

# Processing optimization
USE_MULTIPROCESSING = True      # Enable multiprocessing where possible
MAX_WORKERS = 4                 # Maximum worker processes
BATCH_SIZE = 32                 # Batch size for processing

# Memory optimization
FEATURE_CACHE_SIZE = 1000       # Maximum cached feature vectors
PRELOAD_MODEL = True            # Preload model at startup
LAZY_LOADING = False            # Disable lazy loading for better performance

# Image processing optimization
RESIZE_INTERPOLATION = cv2.INTER_AREA  # Best for downsampling
CONTOUR_RETRIEVAL = cv2.RETR_EXTERNAL  # Only external contours
CONTOUR_APPROXIMATION = cv2.CHAIN_APPROX_SIMPLE  # Compressed contours

# ============================================================================
# CONFIGURATION FUNCTIONS
# ============================================================================

def validate_config():
    """Validate that all required paths and parameters are properly set."""
    import warnings
    
    # Create required directories
    required_paths = [MODELS_DIR, DATASET_DIR, DATASET_PATH, IMAGES_PATH, ANNOTATIONS_PATH]
    
    for path in required_paths:
        if not path.exists():
            print(f"Info: Creating directory: {path}")
            path.mkdir(parents=True, exist_ok=True)
    
    # Validate image size parameters
    if CHAR_WIDTH != CHAR_HEIGHT:
        warnings.warn("Non-square character images may affect HOG performance", UserWarning)
    
    # Validate HOG parameters consistency
    win_size = HOG_PARAMS['_winSize']
    if win_size != CHAR_IMAGE_SIZE:
        warnings.warn(f"HOG window size {win_size} != character image size {CHAR_IMAGE_SIZE}", UserWarning)
    
    # Validate block size compatibility
    block_size = HOG_PARAMS['_blockSize']
    if CHAR_WIDTH % block_size[0] != 0 or CHAR_HEIGHT % block_size[1] != 0:
        warnings.warn(f"Image size {CHAR_IMAGE_SIZE} not compatible with block size {block_size}", UserWarning)
    
    # Validate filtering parameters
    if MIN_CONTOUR_AREA <= 0:
        warnings.warn("MIN_CONTOUR_AREA should be positive", UserWarning)
    
    if MIN_ASPECT_RATIO >= MAX_ASPECT_RATIO:
        warnings.warn("MIN_ASPECT_RATIO should be less than MAX_ASPECT_RATIO", UserWarning)
    
    # Validate confidence thresholds
    if not (0 <= MIN_CONFIDENCE_THRESHOLD <= 1):
        warnings.warn("MIN_CONFIDENCE_THRESHOLD should be between 0 and 1", UserWarning)
    
    print("Configuration validation complete")

def get_preprocessing_config():
    """Get preprocessing configuration dictionary."""
    return {
        'target_size': CHAR_IMAGE_SIZE,
        'gaussian_kernel': GAUSSIAN_KERNEL_SIZE,
        'gaussian_sigma_x': GAUSSIAN_SIGMA_X,
        'gaussian_sigma_y': GAUSSIAN_SIGMA_Y,
        'median_filter_size': MEDIAN_FILTER_SIZE,
        'morph_open_kernel': MORPH_OPEN_KERNEL,
        'morph_close_kernel': MORPH_CLOSE_KERNEL,
        'morph_iterations': MORPH_ITERATIONS,
        'threshold_type': THRESHOLD_TYPE,
        'threshold_max_value': THRESHOLD_MAX_VALUE
    }

def get_feature_config():
    """Get feature extraction configuration dictionary."""
    return {
        'hog_params': HOG_PARAMS.copy(),
        'feature_weights': FEATURE_WEIGHTS.copy(),
        'use_hu_moments': True,
        'use_hog': True,
        'use_geometric': True,
        'use_projection': True,
        'use_zoning': True,
        'use_crossing': True
    }

def get_model_config():
    """Get model training configuration dictionary."""
    return {
        'svm_c': SVM_C_PARAM,
        'svm_gamma': SVM_GAMMA,
        'rf_n_estimators': RF_N_ESTIMATORS,
        'rf_max_depth': RF_MAX_DEPTH,
        'rf_min_samples_split': RF_MIN_SAMPLES_SPLIT,
        'rf_min_samples_leaf': RF_MIN_SAMPLES_LEAF,
        'cv_folds': CV_FOLDS,
        'random_state': RANDOM_STATE
    }

# Alias functions for backward compatibility
def get_feature_extraction_config():
    """Get feature extraction configuration dictionary (alias for get_feature_config)."""
    return get_feature_config()

def get_classification_config():
    """Get classification configuration dictionary (alias for get_model_config)."""
    return get_model_config()

# Run validation when module is imported
if __name__ != "__main__":
    validate_config()
