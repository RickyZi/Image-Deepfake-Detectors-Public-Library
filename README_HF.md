---
title: Deepfake Detection Library
emoji: 🔍
colorFrom: red
colorTo: orange
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Deepfake Detection Library

This Space provides a unified interface to test multiple state-of-the-art deepfake detection models on your images.

## Available Detectors

- **R50_TF** - ResNet-50 based detector trained on TrueFake dataset
- **R50_nodown** - ResNet-50 without downsampling operations
- **CLIP-D** - CLIP-based deepfake detector
- **P2G** - Prompt2Guard: Conditioned prompt-optimization for continual deepfake detection
- **NPR** - Neural Posterior Regularization

## Usage

1. Upload an image
2. Select a detector from the dropdown
3. Click "Detect" to get the prediction

The detector will return:
- **Prediction**: Real or Fake
- **Confidence**: Model confidence score (0-1)
- **Elapsed Time**: Processing time

## Models

All models have been pretrained on images generated with StyleGAN2 and StableDiffusionXL, and real images from the FFHQ Dataset and the FORLAB Dataset.

## References

For more information about the implementation and benchmarking, visit the [GitHub repository](https://github.com/truebees-ai/Image-Deepfake-Detectors-Public-Library).

## Note

⚠️ Due to file size limitations, model weights need to be downloaded automatically on first use. This may take a few moments.
