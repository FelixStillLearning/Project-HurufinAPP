#!/usr/bin/env python3

import cv2
import numpy as np
from src.alphabetic_recognition.config import CHAR_IMAGE_SIZE, HOG_PARAMS

# Test HOG descriptor to see actual feature count
hog_descriptor = cv2.HOGDescriptor(
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

# Create a test image
test_image = np.ones(CHAR_IMAGE_SIZE, dtype=np.uint8) * 255
cv2.rectangle(test_image, (5, 5), (27, 27), 0, 2)

# Compute HOG features
hog_features = hog_descriptor.compute(test_image)
hog_size = len(hog_features) if hog_features is not None else 0

print(f'HOG window size: {CHAR_IMAGE_SIZE}')
print(f'HOG features produced: {hog_size}')
print(f'Expected in code: 36')
print(f'Difference: {360 - 72 - (hog_size - 36)} features missing')

# Calculate what the total should be with actual HOG size
actual_total = 7 + hog_size + 5 + 6 + 16 + 2
print(f'Actual total with real HOG: {actual_total}')
print(f'Model expects: 360')
print(f'Still missing: {360 - actual_total} features')
