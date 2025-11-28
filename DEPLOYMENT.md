# Hugging Face Spaces Deployment Guide

## Prerequisites

1. A Hugging Face account
2. Git LFS installed locally: `git lfs install`
3. Model weights downloaded to the correct directories

## Deployment Steps

### 1. Prepare Model Weights

You have two options:

#### Option A: Upload weights via Git LFS (Recommended for public spaces)

```bash
# Initialize Git LFS
git lfs install

# Track large files
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.pkl"

# Add weights
git add .gitattributes
git add detectors/*/checkpoint/pretrained/weights/best.pt
git add detectors/P2G/src/utils/classes.pkl
git commit -m "Add model weights"
```

#### Option B: Configure automatic download

1. Upload your model weights to Google Drive or another host
2. Update `download_weights.py` with the correct URLs
3. Weights will download automatically when the Space starts

### 2. Create Hugging Face Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Choose:
   - **Name**: deepfake-detection-library (or your preferred name)
   - ** SDK**: Gradio
   - **License**: MIT
   - **Hardware**: CPU Basic (free) or upgrade to GPU if needed

### 3. Push to Hugging Face

```bash
# Add HF remote (replace YOUR_USERNAME and SPACE_NAME)
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME

# Rename README for HF
mv README.md README_github.md
mv README_HF.md README.md

# Push to Hugging Face
git add .
git commit -m "Initial commit for HF Spaces"
git push hf main
```

### 4. Configure Space

In your Space settings on Hugging Face:

- **Hardware**: Start with CPU Basic (free), upgrade to GPU if needed
- **Secrets**: Add any API keys if needed (none required currently)
- **Variables**: No special environment variables needed

### 5. Verify Deployment

1. Wait for the Space to build (may take 5-10 minutes)
2. Test each detector with sample images
3. Check logs for any errors

## File Size Considerations

- **Git LFS** is required for files >10MB
- Each model weight file (~100-500MB) will be stored via LFS
- Free HF Spaces have storage limits; consider:
  - Upgrading to Pro for more storage
  - Using automatic download instead of uploading weights

## Troubleshooting

### Space fails to build

- Check `requirements.txt` for incompatible versions
- Review build logs in the Space interface
- Ensure all dependencies are listed

### Weights not loading

- Verify Git LFS tracked the files: `git lfs ls-files`
- Check file sizes: LFS pointer files are ~130 bytes
- Update `download_weights.py` if using automatic download

### Out of memory errors

- Upgrade to GPU hardware (T4 small recommended)
- Reduce batch size or model size if possible
- Use CPU inference for deployment (already configured)

## Cost Optimization

- **CPU Basic** (free): Works but slower
- **CPU Upgrade** ($0.03/hour): Faster inference
- **T4 Small GPU** ($0.60/hour):  Needed for real-time performance

## Maintenance

- Monitor Space usage in HF dashboard
- Update models by pushing new weights via Git LFS
- Check Gradio version compatibility: `pip list | grep gradio`

## Support

For issues specific to this deployment, check:
- [Gradio Documentation](https://gradio.app/docs/)
- [HF Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [GitHub Repository](https://github.com/truebees-ai/Image-Deepfake-Detectors-Public-Library)
