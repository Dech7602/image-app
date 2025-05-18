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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, feature_extractor, tokenizer, device

model, feature_extractor, tokenizer, device = load_model()
gen_kwargs = {"max_length": 16, "num_beams": 4}

# Captioning function
def generate_caption(image: Image.Image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return preds[0].strip()

# Streamlit app
st.title("🖼️ Image Captioning App with ViT-GPT2")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    with st.spinner("Generating caption..."):
        caption = generate_caption(image)
    st.success("Caption Generated!")
    st.write("**Caption:**", caption)
