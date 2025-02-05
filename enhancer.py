# Import required libraries
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
import sys

# Set Streamlit page configuration
st.set_page_config(page_title="AI Photo Enhancer", layout="wide")

# Prevent Streamlit file watcher error with PyTorch
if 'torch.classes' in sys.modules:
    del sys.modules['torch.classes']

# ✅ **Download ESRGAN Model from Google Drive**
MODEL_PATH = "models/RRDB_ESRGAN_x4.pth"
MODEL_URL = "https://drive.google.com/file/d/1M_4u0EEq1ZUeHhrudY8P_P_08F8utv5f/view?usp=drive_link"  # 

if not os.path.exists(MODEL_PATH):
    st.warning("Downloading ESRGAN model (This happens only once)...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ✅ **Define ESRGAN Model Architecture**
class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=32, gc=16, bias=True):  # ✅ Reduced channels for speed
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, padding=1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, padding=1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, padding=1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, padding=1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf, gc=16):  # ✅ Reduced GC
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock(nf, gc)
        self.RDB2 = ResidualDenseBlock(nf, gc)
        self.RDB3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=32, nb=10, gc=16):  # ✅ Optimized model
        super(RRDBNet, self).__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, padding=1, bias=True)
        self.RRDB_trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, padding=1, bias=True)
        self.upconv1 = nn.Conv2d(nf, nf, 3, padding=1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, padding=1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, padding=1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, padding=1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out

# ✅ **Load AI Model**
@st.cache_resource
def load_model():
    model = RRDBNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'), strict=True)
    model.eval()
    return model

model = load_model()

# ✅ **Image Enhancement Function**
def enhance_image(image, contrast=1.5, sharpness=2.0, brightness=1.2, super_res=False):
    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("Applying enhancements...")
    img = np.array(image.convert("RGB"))

    # Contrast Enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_img = cv2.merge((l, a, b))
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2RGB)

    # Convert to PIL Image
    enhanced_img = Image.fromarray(enhanced_img)
    enhanced_img = ImageEnhance.Sharpness(enhanced_img).enhance(sharpness)
    enhanced_img = ImageEnhance.Brightness(enhanced_img).enhance(brightness)

    progress_bar.progress(60)

    if super_res:
        status_text.text("Applying AI Super-Resolution...")
        enhanced_img = apply_super_resolution(enhanced_img)

    progress_bar.progress(100)
    status_text.text("Enhancement complete!")
    return enhanced_img

# ✅ **Apply Super-Resolution**
def apply_super_resolution(image):
    img_tensor = torch.from_numpy(np.array(image).transpose((2, 0, 1))).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        output = model(img_tensor)
    output_image = output.squeeze().cpu().numpy().transpose((1, 2, 0))
    output_image = np.clip(output_image * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(output_image)

# ✅ **Streamlit UI**
st.title("📸 AI-Powered Photo Enhancer")
st.write("Enhance your photos using AI-powered super-resolution and image processing.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    st.sidebar.subheader("Enhancement Settings")
    contrast = st.sidebar.slider("Contrast", 1.0, 3.0, 1.5)
    sharpness = st.sidebar.slider("Sharpness", 1.0, 5.0, 2.0)
    brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.2)
    super_res = st.sidebar.checkbox("Apply AI Super-Resolution", disabled=model is None)

    if st.sidebar.button("Enhance Image"):
        with col2:
            enhanced_image = enhance_image(image, contrast, sharpness, brightness, super_res)
            st.image(enhanced_image, use_container_width=True)
            st.download_button("Download Enhanced Image", io.BytesIO(), "enhanced_image.jpg", "image/jpeg")

st.markdown("---\nMade with ❤️ using Streamlit, PyTorch, and OpenCV")
