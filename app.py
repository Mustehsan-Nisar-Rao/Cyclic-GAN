import streamlit as st
import torch
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import tempfile
import os
import torch.nn as nn

st.set_page_config(page_title="Sketch to Photo - CycleGAN", page_icon="🎨")

# Generator class
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_res_blocks=6):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        in_f, out_f = 64, 128
        for _ in range(2):
            model += [nn.Conv2d(in_f, out_f, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_f), nn.ReLU(inplace=True)]
            in_f, out_f = out_f, out_f * 2
        for _ in range(n_res_blocks):
            model += [ResidualBlock(in_f)]
        out_f = in_f // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_f, out_f, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_f), nn.ReLU(inplace=True)]
            in_f, out_f = out_f, out_f // 2
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, out_channels, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model():
    # 🔥 CHANGE TO YOUR ACTUAL URL
    MODEL_URL = "https://github.com/Mustehsan-Nisar-Rao/Cyclic-GAN/releases/tag/v.1/cyclegan_best_model.pth"
    
    with st.spinner("🔄 Loading model (245 MB)..."):
        # Download checkpoint
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp_path = tmp.name
        
        # 🔥 IMPORTANT: torch.load() use karo, torch.jit.load() nahi
        checkpoint = torch.load(tmp_path, map_location='cpu')
        
        # Create model and load weights
        model = Generator()
        model.load_state_dict(checkpoint['G_S2P'])
        model.eval()
        
        # Cleanup
        os.unlink(tmp_path)
        
        return model

def preprocess(image):
    image = image.resize((128, 128))
    img = np.array(image).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

def postprocess(tensor):
    tensor = tensor.squeeze(0).permute(1, 2, 0).numpy()
    tensor = (tensor + 1.0) / 2.0
    tensor = np.clip(tensor, 0, 1)
    return Image.fromarray((tensor * 255).astype(np.uint8))

st.title("🎨 Sketch to Photo Translation")
st.markdown("Powered by CycleGAN")

model = load_model()
st.success("✅ Model loaded successfully!")

uploaded = st.file_uploader("Upload a sketch...", type=['png', 'jpg', 'jpeg'])

if uploaded:
    sketch = Image.open(uploaded).convert('RGB')
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(sketch, caption="Input Sketch", use_container_width=True)
    
    with st.spinner("Generating photo..."):
        input_tensor = preprocess(sketch)
        with torch.no_grad():
            output = model(input_tensor)
        photo = postprocess(output)
    
    with col2:
        st.image(photo, caption="Generated Photo", use_container_width=True)
    
    buf = BytesIO()
    photo.save(buf, format="PNG")
    st.download_button("📥 Download Photo", buf.getvalue(), "photo.png")

st.sidebar.markdown("""
**Model:** CycleGAN  
**Training:** 50 epochs on Sketchy dataset  
**Input/Output:** 128×128 pixels
""")
