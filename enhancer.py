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
import os

# Set page config (must be the first Streamlit command)
st.set_page_config(page_title="AI Photo Enhancer", layout="wide")

# Import RRDBNet architecture
from RRDBNet_arch import RRDBNet

# Ensure model directory exists
os.makedirs("models", exist_ok=True)

# Google Drive model link
MODEL_URL = "https://drive.google.com/uc?id=1M_4u0EEq1ZUeHhrudY8P_P_08F8utv5f"  # Modify with correct ID

# Function to download model
@st.cache_resource
def download_model():
    model_path = "models/RRDB_ESRGAN_x4.pth"
    if not os.path.exists(model_path):
        st.info("Downloading ESRGAN model...")
        gdown.download(MODEL_URL, model_path, quiet=False)
    return model_path

# Function to load ESRGAN model
@st.cache_resource
def load_model():
    model_path = download_model()
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    model.eval()
    return model

# Load model
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.warning("Super-resolution disabled.")
    model = None

# Function to check image size
def check_image_size(image):
    """Check if image is too large to process"""
    width, height = image.size
    pixels = width * height
    if pixels > 4000 * 3000:  # 12MP limit
        return False
    return True

# Function to enhance image
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

# Function to apply AI Super-Resolution
def apply_super_resolution(image, tile_size=256):
    """Apply AI Super-Resolution using Tiling to Prevent Crashes"""
    img = np.array(image.convert("RGB")) / 255.0  # Normalize
    h, w, _ = img.shape

    # Create output image placeholder
    output_img = np.zeros((h * 4, w * 4, 3), dtype=np.float32)

    # Process the image tile-by-tile
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = img[y:y+tile_size, x:x+tile_size, :]
            tile = torch.from_numpy(tile.transpose((2, 0, 1))).float().unsqueeze(0)

            # Apply the ESRGAN model
            with torch.no_grad():
                out_tile = model(tile).squeeze().cpu().numpy().transpose((1, 2, 0))

            # Store the processed tile in the output image
            output_img[y*4:(y+tile_size)*4, x*4:(x+tile_size)*4, :] = out_tile

    # Convert back to an image
    output_img = np.clip(output_img * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(output_img)

# Function to convert PIL Image to bytes
def pil_to_bytes(image):
    """Convert PIL Image to byte format"""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=95)
    return img_byte_arr.getvalue()

# Streamlit UI
st.title("📸 AI-Powered Photo Enhancer")
st.write("Enhance your photos using AI Super-Resolution and OpenCV!")

# File uploader
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
        st.image(image, use_container_width=True)

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
            st.image(enhanced_image, use_container_width=True)
            
            # Download button
            img_byte_arr = pil_to_bytes(enhanced_image)
            st.download_button(
                "Download Enhanced Image",
                img_byte_arr,
                "enhanced_image.jpg",
                "image/jpeg",
                use_container_width=True
            )

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Saurabh using Streamlit, PyTorch, and OpenCV")
