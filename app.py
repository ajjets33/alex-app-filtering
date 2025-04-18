import streamlit as st
import numpy as np
from skimage import filters, feature, exposure
from PIL import Image
import io

def load_image(image_file):
    if image_file is not None:
        image = Image.open(image_file)
        return np.array(image)
    return None

def apply_sobel_edge_detection(image):
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image
    
    edges_h = filters.sobel_h(gray)
    edges_v = filters.sobel_v(gray)
    magnitude = np.sqrt(edges_h**2 + edges_v**2)
    
    # Normalize to 0-255 range
    magnitude = (magnitude * 255 / magnitude.max()).astype(np.uint8)
    return np.stack([magnitude] * 3, axis=-1)

def apply_custom_convolution(image, sigma):
    if len(image.shape) == 3:
        result = np.zeros_like(image)
        for i in range(3):
            result[:,:,i] = filters.gaussian(image[:,:,i], sigma=sigma)
        return (result * 255).astype(np.uint8)
    else:
        filtered = filters.gaussian(image, sigma=sigma)
        return (filtered * 255).astype(np.uint8)

def apply_artistic_filter(image, intensity):
    if len(image.shape) == 3:
        # Convert to LAB color space-like enhancement
        result = np.zeros_like(image)
        for i in range(3):
            result[:,:,i] = exposure.equalize_adapthist(
                image[:,:,i], 
                kernel_size=int(image.shape[0]/8),
                clip_limit=intensity/5
            )
        return (result * 255).astype(np.uint8)
    else:
        enhanced = exposure.equalize_adapthist(
            image,
            kernel_size=int(image.shape[0]/8),
            clip_limit=intensity/5
        )
        return (enhanced * 255).astype(np.uint8)

def main():
    st.set_page_config(page_title="Image Filter App", layout="wide")
    
    st.title("Creative Image Filter Application")
    st.sidebar.header("Filter Controls")
    
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        
        filter_type = st.sidebar.selectbox(
            "Select Filter",
            ["Original", "Sobel Edge Detection", "Custom Convolution", "Artistic Enhancement"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Filtered Image")
            
            if filter_type == "Original":
                filtered_image = image
            
            elif filter_type == "Sobel Edge Detection":
                filtered_image = apply_sobel_edge_detection(image)
            
            elif filter_type == "Custom Convolution":
                sigma = st.sidebar.slider("Blur Intensity", 0.1, 5.0, 1.0, step=0.1)
                filtered_image = apply_custom_convolution(image, sigma)
            
            elif filter_type == "Artistic Enhancement":
                intensity = st.sidebar.slider("Enhancement Intensity", 1.0, 5.0, 2.0, step=0.1)
                filtered_image = apply_artistic_filter(image, intensity)
            
            st.image(filtered_image, use_column_width=True)
            
            # Add download button for filtered image
            if st.button("Download Filtered Image"):
                filtered_pil = Image.fromarray(filtered_image)
                buf = io.BytesIO()
                filtered_pil.save(buf, format="PNG")
                st.download_button(
                    label="Download",
                    data=buf.getvalue(),
                    file_name="filtered_image.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()
