import streamlit as st
from PIL import Image
from diffusers import StableDiffusionPipeline
from io import BytesIO
import asyncio

try:
    loop = asyncio.get_event_loop()
except RuntimeError as e:
    if "no current event loop" in str(e):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    else:
        raise e

st.set_page_config(page_title="Stable Bud - AI Image Generator", layout="centered")
st.title("🌟 cyfuture - AI Image Generator")

@st.cache_resource()
def load_model():
    model_id = "CompVis/stable-diffusion-v1-4"
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id)
        pipe.to("cpu")
        return pipe
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        return None

pipe = load_model()

prompt = st.text_input("Enter a prompt:", "A fantasy landscape with castles and dragons")
guidance_scale = st.slider("Guidance Scale:", 1.0, 20.0, 7.5, 0.5)
generate_btn = st.button("Generate Image")

if generate_btn and pipe is not None:
    with st.spinner("Generating Image..."):
        try:
            image = pipe(prompt, guidance_scale=guidance_scale, height=256, width=256).images[0]
            
            img_bytes = BytesIO()
            image.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            st.image(image, caption="Generated Image", use_column_width=True)
            st.success(" Image generated successfully!")

            st.download_button("📥 Download Image", img_bytes, "generated_image.png", "image/png")
        
        except Exception as e:
            st.error(f"Error generating image: {e}")
