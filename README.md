# Smart Gait Using AI: Human Identification Using Deep Learning

An automated, end-to-end computer vision and deep learning framework for contactless biometric person identification using gait analysis.

## Overview
This system processes surveillance video frames using spatial-temporal motion modeling:
1. **Detection & Segmentation:** Isolating human subjects using YOLOv8 and adaptive background subtraction.
2. **Morphological Filtering:** Noise suppression and edge smoothing via morphological operators (dilation, erosion, closing).
3. **GEI Representation:** Condensing complete gait cycles into 128x128 Gait Energy Images (GEI).
4. **Deep Classification:** Custom CNN architecture trained to extract discriminative features and classify identities.

## Benchmark Performance
Evaluated on the **CASIA-B Benchmark Dataset** across varying conditions (normal walking, carrying bags, wearing coats):
- **Overall Recognition Accuracy:** 97.5%
- **Precision:** 0.96
- **Recall:** 0.95
- **Inference Speed:** ~15–20 FPS

## Tech Stack
- Python 3.x
- OpenCV
- TensorFlow / Keras
- NumPy
