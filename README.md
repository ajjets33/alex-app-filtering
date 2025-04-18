# Image Filtering Application

This interactive image filtering application allows users to apply various image processing techniques including edge detection and convolution filters.

## Features
- Upload and process images in real-time
- Multiple filter options including:
  - Sobel Edge Detection
  - Custom Convolution Kernels
  - Gaussian Blur
  - Custom filters
- Interactive parameter adjustment
- Live preview of filter effects

## Setup
1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

## Usage
1. Upload an image using the file uploader
2. Select a filter from the sidebar
3. Adjust parameters as needed
4. View the results in real-time

## Development
- Built with Streamlit
- Uses OpenCV and NumPy for image processing
- Developed using Cursor and Google Colab
