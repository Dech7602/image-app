import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image

# Load model and processor
@st.cache_resource
def load_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    # Ensure decoder_start_token_id is defined
    model.config.decoder_start_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, feature_extractor, tokenizer, device

# Load once at startup
model, feature_extractor, tokenizer, device = load_model()

# Generation configuration
gen_kwargs = {
    "max_length": 64,
    "do_sample": True,         # enables sampling instead of greedy/beam
    "top_k": 50,               # restricts to top 50 tokens
    "top_p": 0.95,             # nucleus sampling
    "temperature": 1.0,        # randomness level
    "num_return_sequences": 1
}

# Caption generation function
def generate_caption(image: Image.Image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    with torch.no_grad():
        output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return preds[0].strip()

# Streamlit UI
st.title("🖼️ Image Captioning App with ViT-GPT2")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    with st.spinner("Generating caption..."):
        caption = generate_caption(image)
    st.success("Caption Generated!")
    st.write("**Caption:**", caption)
