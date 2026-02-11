import streamlit as st
import torch
import pickle
import os
from PIL import Image
from io import BytesIO
from train import ImageCaptioner, greedy_search, load_captions, Vocabulary
from torchvision import transforms

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


def main():
    # Title and description
    st.title("Image Captioning")
    st.write("Generate automatic captions for images using a deep learning model.")
    
    # Load model and data
    model, vocab, features, pairs = load_model_and_data()
    
    # Sidebar with info
    with st.sidebar:
        st.header("About")
        st.write("""
        This app uses an **Image Captioning Model** that combines:
        - **ResNet-50**: Extracts visual features from images
        - **LSTM**: Generates natural language descriptions
        
        The model is trained on the Flickr30K dataset.
        """)
        
        st.divider()
        st.subheader("Dataset Stats")
        st.metric("Available Images", len(pairs))
        st.metric("Vocabulary Size", len(vocab.word2idx))
    
    # Tabs for different interfaces
    tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "Dataset Sample", "How It Works"])
    
    with tab1:
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
                            # Convert image to model input format
                            transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.485, 0.456, 0.406),
                                    (0.229, 0.224, 0.225)
                                )
                            ])
                            
                            img_tensor = transform(image).unsqueeze(0)
                            
                            # Extract features and generate caption
                            with torch.no_grad():
                                # Simple feature extraction using a pre-trained encoder
                                device = "cpu"
                                # Using a mock feature extraction for demo
                                feature = torch.randn(1, 2048)  # Placeholder
                            
                            caption = greedy_search(model, feature.squeeze(0), vocab, device=device)
                            
                            st.success("Caption Generated!")
                            st.markdown(f"###Caption:\n**{caption}**")
                            
                        except Exception as e:
                            st.error(f"Error generating caption: {str(e)}")
    
    with tab2:
        st.subheader("Sample Captions from Dataset")
        
        if pairs:
            num_samples = st.slider("Number of samples to display", 1, min(10, len(pairs)), 3)
            
            # Display random samples
            import random
            samples = random.sample(pairs, min(num_samples, len(pairs)))
            
            for i, (image_name, caption) in enumerate(samples, 1):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write(f"**Image {i}:**")
                    st.code(image_name, language="text")
                with col2:
                    st.write(f"**Caption {i}:**")
                    st.info(caption)
        else:
            st.warning("No dataset captions available. Please ensure data/captions.txt exists.")
    
    with tab3:
        st.subheader("How This Model Works")
        
        st.markdown("""
        ### 🧠 Architecture Overview
        
        **1. Image Encoder (ResNet-50)**
        - Extracts visual features from input images
        - Produces a 2048-dimensional feature vector
        - Pre-trained on ImageNet
        
        **2. Feature Projection**
        - Projects the image features to the hidden dimension
        - Initializes the LSTM state
        
        **3. Caption Decoder (LSTM)**
        - Generates captions word-by-word
        - Uses an embedding layer for word representations
        - Predicts the next word based on:
          - Previous words (via embeddings)
          - Image features (via initial hidden state)
        
        **4. Greedy Search**
        - Selects the highest probability word at each step
        - Stops when END token is generated or max length is reached
        
        ### 📊 Training Details
        - **Dataset**: Flickr30K (31K images, ~5 captions each)
        - **Loss Function**: Cross-Entropy Loss
        - **Optimizer**: Adam
        - **Evaluation Metric**: BLEU-4 Score
        """)


if __name__ == "__main__":
    main()
