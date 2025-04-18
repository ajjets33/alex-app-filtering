import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

def load_image(image_file):
    if image_file is not None:
        image = Image.open(image_file)
        return np.array(image)
    return None

def apply_sobel_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))
    return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2RGB)

def apply_custom_convolution(image, kernel_size, sigma):
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel_2d = kernel @ kernel.T
    filtered = cv2.filter2D(image, -1, kernel_2d)
    return filtered

def apply_artistic_filter(image, intensity):
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=intensity, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge channels
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

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
                kernel_size = st.sidebar.slider("Kernel Size", 3, 15, 5, step=2)
                sigma = st.sidebar.slider("Sigma", 0.1, 5.0, 1.0, step=0.1)
                filtered_image = apply_custom_convolution(image, kernel_size, sigma)
            
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
