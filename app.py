import streamlit as st
import torch
import torch.nn as nn
import pickle
import os
from PIL import Image
from io import BytesIO
from train import ImageCaptioner, greedy_search, load_captions, Vocabulary
from torchvision import transforms, models

# Page configuration
st.set_page_config(
    page_title="Image Captioning",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton > button {
        width: 100%;
        height: 3rem;
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model_and_data():
    """Load model, vocabulary, and features with caching."""
    checkpoint_path = "caption_model.pt"
    features_path = "flickr30k_features.pkl"
    
    if not os.path.isfile(checkpoint_path):
        st.error(f"Model not found: {checkpoint_path}. Please train with train.py first.")
        st.stop()
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        vocab = checkpoint["vocab"]
        model = ImageCaptioner(len(vocab.word2idx)).to("cpu")
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        
        if os.path.isfile(features_path):
            with open(features_path, "rb") as f:
                features = pickle.load(f)
        else:
            features = {}
            st.warning("Features file not found. Pre-extracted features unavailable.")
        
        pairs = load_captions("data/captions.txt") if os.path.isfile("data/captions.txt") else []
        pairs = [(img, cap) for img, cap in pairs if img in features]
        
        return model, vocab, features, pairs
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()


@st.cache_resource
def load_feature_extractor():
    """Load and cache the ResNet-50 feature extractor."""
    device = torch.device("cpu")
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Remove the last classification layer to get features only
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor, device


def extract_features(image, feature_extractor, device):
    """Extract ResNet-50 features from an image."""
    # Ensure image is in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = feature_extractor(img_tensor)
        features = features.view(features.size(0), -1)  # Flatten to (1, 2048)
    
    return features


def main():
    # Title and description
    st.title("Image Captioning")
    st.write("Generate automatic captions for images using a deep learning model.")
    
    # Load model and data
    model, vocab, features, pairs = load_model_and_data()
    feature_extractor, device = load_feature_extractor()
    
    # Sidebar with info
    with st.sidebar:
        st.header("About")
        st.write("""
        This app uses an **Image Captioning Model** that combines:
        - **ResNet-50**: Extracts visual features from images
        - **LSTM**: Generates natural language descriptions
        
        The model is trained on the Flickr30K dataset.
        """)
    
    # Tabs for different interfaces
    
    st.subheader("Upload Your Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Generate caption button
        with col2:
            st.write("")
            st.write("")
            if st.button("Generate Caption", use_container_width=True):
                with st.spinner("Generating caption..."):
                    try:
                        # Extract features from the uploaded image
                        feature = extract_features(image, feature_extractor, device)
                        
                        # Generate caption
                        caption = greedy_search(model, feature.squeeze(0), vocab, device=device)
                        
                        st.success("Caption Generated!")
                        st.markdown(f"### Caption:\n**{caption}**")
                    except Exception as e:
                        st.error(f"Error generating caption: {str(e)}")


if __name__ == "__main__":
    main()
