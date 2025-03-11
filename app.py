import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from io import BytesIO

# Authentication token for Hugging Face
from authtoken import auth_token

# Page Configuration
st.set_page_config(page_title="Stable Bud - AI Image Generator", layout="centered")
st.title("🌟 Stable Bud - AI Image Generator")

# Load Model
@st.cache_resource()
def load_model():
    model_id = "stabilityai/stable-diffusion-2-1-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        use_auth_token=auth_token  # Corrected usage
    )
    pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()  # Further VRAM optimization
    return pipe

pipe = load_model()

# UI Elements
prompt = st.text_input("Enter a prompt:", "A fantasy landscape with castles and dragons")
guidance_scale = st.slider("Guidance Scale:", 1.0, 20.0, 8.5, 0.5)
generate_btn = st.button("Generate Image")

# Image Generation
if generate_btn:
    with st.spinner("Generating Image..."):
        image = pipe(prompt, guidance_scale=guidance_scale, height=512, width=512).images[0]
        
        # Convert image to bytes
        img_bytes = BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        # Display Image
        st.image(image, caption="Generated Image", use_column_width=True)
        st.success("✅ Image generated successfully!")

        # Download Option
        st.download_button("📥 Download Image", img_bytes, "generated_image.png", "image/png")
    
    # Free memory
    del image
    torch.cuda.empty_cache()

