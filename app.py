import gradio as gr
import os
import sys
import json
import argparse
from types import SimpleNamespace
from support.detect import run_detect

# Download weights on first run (for HF Spaces)
if os.environ.get("SPACE_ID"):
    try:
        from download_weights import download_all_weights
        download_all_weights()
    except Exception as e:
        print(f"Warning: Could not download weights: {e}")

# Available detectors based on launcher.py
DETECTORS = ['R50_TF', 'R50_nodown', 'CLIP-D', 'P2G', 'NPR']

def predict(image_path, detector_name):
    if not image_path:
        return {"error": "Please upload an image."}
    
    # Create a temporary output file path
    output_path = "temp_result.json"
    
    # Mock args object
    args = SimpleNamespace(
        image=image_path,
        detector=detector_name,
        config_dir='configs',
        output=output_path,
        weights='pretrained', # Use default/pretrained
        device='cpu', # Force CPU
        dry_run=False,
        verbose=False
    )
    
    try:
        # Run detection
        # We need to capture stdout/stderr or just trust the function
        # run_detect might raise FileNotFoundError if weights are missing
        run_detect(args)
        
        # Read results
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                result = json.load(f)
            
            # Format output
            prediction = result.get('prediction', 'Unknown')
            confidence = result.get('confidence', 0.0)
            elapsed_time = result.get('elapsed_time', 0.0)
            
            return {
                "Prediction": prediction,
                "Confidence": f"{confidence:.4f}",
                "Elapsed Time": f"{elapsed_time:.3f}s"
            }
        else:
            return {"error": "No result file generated. Check console logs for details."}
            
    except FileNotFoundError as e:
        return {"error": str(e), "message": f"Please ensure you have downloaded the weights for {detector_name}."}
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Cleanup
        if os.path.exists(output_path):
            os.remove(output_path)

# Create Gradio Interface
with gr.Blocks(title="Deepfake Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 Deepfake Detection Library")
    gr.Markdown("""
    Upload an image and select a detector to check if it's real or fake.
    
    **Available Detectors:**
    - **R50_TF**: ResNet-50 based detector
    - **R50_nodown**: ResNet-50 without downsampling
    - **CLIP-D**: CLIP-based detector
    - **P2G**: Prompt2Guard detector
    - **NPR**: Neural Posterior Regularization
    """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Input Image", height=400)
            detector_input = gr.Dropdown(
                choices=DETECTORS, 
                value=DETECTORS[0], 
                label="Select Detector",
                info="Choose which deepfake detection model to use"
            )
            submit_btn = gr.Button("🔍 Detect", variant="primary")
        
        with gr.Column():
            output_json = gr.JSON(label="Detection Results")
    
    gr.Markdown("""
    ---
    ### About
    This Space provides access to multiple state-of-the-art deepfake detection models. 
    All models are trained on StyleGAN2, StableDiffusionXL, FFHQ, and FORLAB datasets.
    
    **Note:** First detection may be slower due to model loading.
    """)
    
    submit_btn.click(
        fn=predict,
        inputs=[image_input, detector_input],
        outputs=output_json
    )

if __name__ == "__main__":
    # For HF Spaces, share is automatically enabled
    demo.launch()
