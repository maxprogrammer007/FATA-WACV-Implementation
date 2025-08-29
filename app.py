import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T
import requests
import json
from pathlib import Path
import sys
import os

# --- Path Setup ---
# Add the project root to the Python path to ensure robust imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Project Imports ---
from fata.model import setup_model
from fata.augmenter import FATA_Augmenter
from fata.adapt import adapt_on_image
from utils.corruptions import add_gaussian_noise, apply_gaussian_blur

# --- App Configuration ---
st.set_page_config(page_title="FATA Interactive Demo", layout="wide")

# --- Caching Functions for Performance ---
@st.cache_resource
def load_labels():
    """Downloads and caches the ImageNet class labels."""
    labels_path = Path("imagenet_class_index.json")
    if not labels_path.exists():
        url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        try:
            with open(labels_path, "w") as f:
                f.write(requests.get(url).text)
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to download labels: {e}")
            return None
    with open(labels_path) as f:
        labels = json.load(f)
    return {int(k): v[1] for k, v in labels.items()}

@st.cache_resource
def load_fata_model():
    """Loads and caches the FATA model, augmenter, and optimizer."""
    with st.spinner("Loading ResNet-50 model... This may take a moment."):
        model, p1, p2, opt = setup_model()
        aug = FATA_Augmenter()
    return model, p1, p2, opt, aug

def get_prediction_text(softmax_tensor, labels):
    """Formats the model's prediction into a readable string."""
    if softmax_tensor is None or labels is None:
        return "Prediction failed."
    conf, class_idx = torch.max(softmax_tensor, 1)
    class_name = labels.get(class_idx.item(), "Unknown").replace("_", " ").title()
    return f"**{class_name}** (Confidence: {conf.item():.2%})"

# --- Main UI Layout ---
st.title("🔬 FATA: Interactive Test-Time Adaptation Demo")
st.markdown(
    "This app demonstrates the core idea of the **Feature Augmentation based Test-Time Adaptation** paper. "
    "Corrupt an image to simulate a 'domain shift' and then run a real adaptation step to see the model's prediction improve."
)

# Load essential resources
labels = load_labels()
model, m_p1, m_p2, optimizer, augmenter = load_fata_model()

# Create main layout
col1, col2 = st.columns(2)

with col1:
    st.header("1. Input & Corruption")
    uploaded_file = st.file_uploader("Upload your own image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        original_image = Image.open(uploaded_file).convert("RGB")
    else:
        # Provide a default image for immediate use
        default_url = "https://images.unsplash.com/photo-1583337130417-3346a1be7dee?auto=format&fit=crop&w=500"
        try:
            original_image = Image.open(requests.get(default_url, stream=True).raw).convert("RGB")
        except requests.exceptions.RequestException:
            st.error("Failed to load default image. Please upload your own.")
            st.stop()
            
    # Interactive corruption controls
    noise_level = st.slider("Gaussian Noise Level", 0, 100, 30, 5, help="Simulates sensor noise.")
    blur_radius = st.slider("Gaussian Blur Radius", 0, 10, 0, 1, help="Simulates a slightly out-of-focus camera.")

    # Apply corruptions to the image
    corrupted_image = add_gaussian_noise(original_image, noise_level)
    corrupted_image = apply_gaussian_blur(corrupted_image, blur_radius)

    # Display the images
    img_col1, img_col2 = st.columns(2)
    img_col1.image(original_image, caption="Original Image", use_column_width=True)
    img_col2.image(corrupted_image, caption="Corrupted Image (Model Input)", use_column_width=True)

with col2:
    st.header("2. Model Adaptation & Results")

    if st.button("⚡ Adapt Model on Corrupted Image", type="primary", use_container_width=True):
        # Prepare the image tensor for the model
        preprocess = T.Compose([
            T.Resize(256), T.CenterCrop(224), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image_tensor = preprocess(corrupted_image).unsqueeze(0)

        with st.spinner("Running adaptation step... (Performing forward/backward pass)"):
            loss, pre_softmax, post_softmax = adapt_on_image(
                image_tensor, model, m_p1, m_p2, optimizer, augmenter
            )

        if loss == 0.0:
            st.warning(f"Adaptation Skipped: The model's initial prediction was not confident enough (entropy too high). Try reducing the corruption.")
        else:
            st.success(f"Adaptation complete! (Calculated Loss: {loss:.4f})")
        
        # Display results in a structured way
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.subheader("Prediction BEFORE")
            st.markdown(get_prediction_text(pre_softmax, labels))
        with res_col2:
            st.subheader("Prediction AFTER")
            st.markdown(get_prediction_text(post_softmax, labels))

        st.info("The model's parameters have been updated for this session. Running again will adapt from this new state.", icon="ℹ️")
    else:
        st.info("Click the button to run one Test-Time Adaptation step and see the model's confidence change.", icon="💡")

