import streamlit as st
import torch
from PIL import Image
import numpy as np
import requests
from io import BytesIO

st.set_page_config(page_title="Sketch to Photo", page_icon="🎨")

@st.cache_resource
def load_model():
    # 🔥 CHANGE THIS TO YOUR GITHUB RELEASE URL
    MODEL_URL = "https://github.com/Mustehsan-Nisar-Rao/Cyclic-GAN/releases/tag/v.1/cyclegan_best_model.pth"
    
    with st.spinner("Loading model..."):
        response = requests.get(MODEL_URL)
        with open("/tmp/model.pt", "wb") as f:
            f.write(response.content)
        return torch.jit.load("/tmp/model.pt", map_location='cpu')

def preprocess(image):
    image = image.resize((128, 128))
    img = np.array(image).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

def postprocess(tensor):
    tensor = tensor.squeeze(0).permute(1, 2, 0).numpy()
    tensor = (tensor + 1.0) / 2.0
    return Image.fromarray((tensor * 255).astype(np.uint8))

st.title("🎨 Sketch to Photo Translation")

model = load_model()

uploaded = st.file_uploader("Upload a sketch...", type=['png', 'jpg', 'jpeg'])

if uploaded:
    sketch = Image.open(uploaded).convert('RGB')
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(sketch, caption="Input Sketch")
    
    with st.spinner("Generating..."):
        input_tensor = preprocess(sketch)
        with torch.no_grad():
            output = model(input_tensor)
        photo = postprocess(output)
    
    with col2:
        st.image(photo, caption="Generated Photo")
    
    buf = BytesIO()
    photo.save(buf, format="PNG")
    st.download_button("Download", buf.getvalue(), "photo.png")
