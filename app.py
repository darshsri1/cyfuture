import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline

# Authentication token for Hugging Face
from authtoken import auth_token

# Page Configuration
st.set_page_config(page_title="Stable Bud - AI Image Generator", layout="centered")
st.title("🌟 Stable Bud - AI Image Generator")

# Load Model
@st.cache_resource()
def load_model():
    model_id = "stabilityai/stable-diffusion-2-1-base"  # Lighter and optimized model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype, 
            use_auth_token=auth_token  # Fixed token usage
        )
        pipe.to(device)
        pipe.enable_attention_slicing()  # Reduce VRAM usage
        if device == "cuda":
            pipe.enable_model_cpu_offload()  # Further reduce memory load
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device
    
    return pipe, device

pipe, device = load_model()

# UI Elements
prompt = st.text_input("Enter a prompt:", "A fantasy landscape with castles and dragons")
guidance_scale = st.slider("Guidance Scale:", min_value=1.0, max_value=20.0, value=8.5, step=0.5)
generate_btn = st.button("Generate Image")

# Image Generation
if generate_btn and pipe is not None:
    with st.spinner("Generating Image..."):
        try:
            image = pipe(prompt, guidance_scale=guidance_scale, height=512, width=512).images[0]  # Reduce image size
            image.save("generated_image.png")
            
            # Display the generated image
            st.image(image, caption="Generated Image", use_column_width=True)
            st.success("Image generated successfully!")

            # Download Option
            with open("generated_image.png", "rb") as file:
                st.download_button("Download Image", file, "generated_image.png", "image/png")
        
            # Free memory
            del image
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            st.error("CUDA out of memory. Try lowering guidance scale or reducing image size.")
        except Exception as e:
            st.error(f"Error generating image: {e}")
