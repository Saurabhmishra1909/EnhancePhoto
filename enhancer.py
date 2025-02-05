# Import statements
import gdown
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import io
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(__file__))

# Import RRDBNet from architecture file
from RRDBNet_arch import RRDBNet

# Set page config (must be the first Streamlit command)
st.set_page_config(page_title="AI Photo Enhancer", layout="wide")

# Prevent Streamlit file watcher error with PyTorch
if 'torch.classes' in sys.modules:
    del sys.modules['torch.classes']

# Google Drive model download function
def download_model_from_drive():
    file_id = "1M_4u0EEq1ZUeHhrudY8P_P_08F8utv5f"  # Replace with your actual Google Drive file ID
    destination = "models/RRDB_ESRGAN_x4.pth"
    if not os.path.exists(destination):  # Check if model already exists
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, destination, quiet=False)

# Download model if not exists
download_model_from_drive()

# Load ESRGAN model with caching
@st.cache_resource
def load_model():
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)  # Match ESRGAN architecture
    model.load_state_dict(torch.load('models/RRDB_ESRGAN_x4.pth', map_location='cpu'), strict=True)
    model.eval()
    return model

# Initialize model
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.warning("Running without super-resolution capability.")
    model = None

# Utility functions
def check_image_size(image):
    """Check if image is too large to process"""
    width, height = image.size
    pixels = width * height
    if pixels > 4000 * 3000:  # 12MP limit
        return False
    return True

def enhance_image(image, contrast=1.5, sharpness=2.0, brightness=1.2, super_res=False):
    """Enhance image using contrast, sharpness, and optionally super-resolution"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Convert image to OpenCV format
    status_text.text("Applying contrast enhancement...")
    img = np.array(image.convert("RGB"))
    
    # Contrast enhancement using CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_img = cv2.merge((l, a, b))
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2RGB)
    progress_bar.progress(30)

    # Convert to PIL Image for sharpness & brightness enhancement
    status_text.text("Adjusting sharpness and brightness...")
    enhanced_img = Image.fromarray(enhanced_img)
    enhanced_img = ImageEnhance.Sharpness(enhanced_img).enhance(sharpness)
    enhanced_img = ImageEnhance.Brightness(enhanced_img).enhance(brightness)
    progress_bar.progress(60)

    # Apply Super-Resolution if selected
    if super_res and model is not None:
        status_text.text("Applying AI Super-Resolution (this may take a while)...")
        enhanced_img = apply_super_resolution(enhanced_img)
    
    progress_bar.progress(100)
    status_text.text("Enhancement complete!")
    return enhanced_img

def apply_super_resolution(image):
    """Apply AI Super-Resolution using ESRGAN"""
    try:
        # Preprocess (keep 0-255 range)
        img_tensor = torch.from_numpy(np.array(image).transpose((2, 0, 1))).float().unsqueeze(0)
        
        # Pad to multiple of 32
        h, w = img_tensor.shape[2], img_tensor.shape[3]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        # Process
        with torch.no_grad():
            output = model(img_tensor)
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :h*4, :w*4]
        
        # Post-process
        output_image = output.squeeze().cpu().numpy().transpose((1, 2, 0))
        output_image = np.clip(output_image, 0, 255).astype(np.uint8)
        
        return Image.fromarray(output_image)
    except Exception as e:
        st.error(f"Error during super-resolution: {e}")
        return image

def pil_to_bytes(image):
    """Convert PIL Image to byte format"""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=95)
    return img_byte_arr.getvalue()

# Streamlit UI
st.title("📸 AI-Powered Photo Enhancer")
st.write("""Enhance your photos using advanced AI and computer vision techniques! Upload an image and adjust the enhancement settings to your liking.""")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display original image
    image = Image.open(uploaded_file)
    
    # Check image size
    if not check_image_size(image):
        st.warning("Image is too large. Please upload a smaller image (max 4000x3000 pixels).")
        st.stop()
    
    # Create two columns for before/after comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    # Enhancement options
    st.sidebar.subheader("Enhancement Settings")
    contrast = st.sidebar.slider("Contrast", 1.0, 3.0, 1.5)
    sharpness = st.sidebar.slider("Sharpness", 1.0, 5.0, 2.0)
    brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.2)
    super_res = st.sidebar.checkbox("Apply AI Super-Resolution", disabled=model is None)
    
    if model is None and super_res:
        st.sidebar.warning("Super-resolution model not loaded. Check model path.")

    # Enhance Image
    if st.sidebar.button("Enhance Image"):
        with col2:
            st.subheader("Enhanced Image")
            enhanced_image = enhance_image(image, contrast, sharpness, brightness, super_res)
            st.image(enhanced_image, use_column_width=True)
            
            # Download button
            img_byte_arr = pil_to_bytes(enhanced_image)
            st.download_button(
                "Download Enhanced Image",
                img_byte_arr,
                "enhanced_image.jpg",
                "image/jpeg",
                use_container_width=True
            )

# Add footer
st.markdown("""--- Made by Saurabh with ❤️ using Streamlit, PyTorch, and OpenCV""")
