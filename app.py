import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import urllib.request
import tempfile
import time
import sys
import subprocess
import pkg_resources

# Check and install required packages
required_packages = ['streamlit', 'opencv-python', 'numpy', 'pillow']
installed_packages = {pkg.key for pkg in pkg_resources.working_set}

for package in required_packages:
    if package.replace('-', '_') not in installed_packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

st.set_page_config(
    page_title="Face and Age Detector",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.2rem;
        color: #424242;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# App header
st.markdown('<p class="main-header">🧠 Face and Age Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Upload an image to detect faces and estimate ages using AI</p>', unsafe_allow_html=True)

# Updated model URLs using more reliable sources
MODEL_URLS = {
    "face_proto": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "face_model": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    "age_proto": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/age_gender_classification/age_deploy.prototxt",
    "age_model": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/age_net.caffemodel"
}

# Alternative backup URLs
BACKUP_MODEL_URLS = {
    "face_proto": "https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt",
    "face_model": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    "age_proto": "https://github.com/opencv/opencv/raw/master/samples/dnn/age_gender_classification/age_deploy.prototxt",
    "age_model": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/age_net.caffemodel"
}

age_list = ['0-2 years', '4-6 years', '8-12 years', '15-20 years',
            '25-32 years', '38-43 years', '48-53 years', '60-100 years']

# Function to download models if not already downloaded
@st.cache_resource
def load_models():
    # Create a models directory in the user's home directory instead of system directories
    models_dir = os.path.join(os.path.expanduser("~"), "face_age_detector_models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_paths = {}
    
    for name, url in MODEL_URLS.items():
        filename = os.path.join(models_dir, os.path.basename(url))
        model_paths[name] = filename
        
        if not os.path.exists(filename):
            with st.spinner(f"Downloading {name} model..."):
                try:
                    urllib.request.urlretrieve(url, filename)
                    st.success(f"Downloaded {name} model to {filename}")
                except Exception as e:
                    st.warning(f"Failed to download from primary URL: {str(e)}")
                    try:
                        # Try backup URL
                        backup_url = BACKUP_MODEL_URLS.get(name)
                        if backup_url:
                            st.info(f"Trying backup URL for {name}...")
                            urllib.request.urlretrieve(backup_url, filename)
                            st.success(f"Downloaded {name} model from backup URL to {filename}")
                        else:
                            raise Exception("No backup URL available")
                    except Exception as e2:
                        st.error(f"Failed to download {name} model from backup URL: {str(e2)}")
                        # Provide a hint for manual download
                        st.error(f"Please download the model manually from {url} and place it in the {models_dir} directory.")
                        raise
    
    # Load networks
    try:
        face_net = cv2.dnn.readNet(model_paths["face_model"], model_paths["face_proto"])
        age_net = cv2.dnn.readNet(model_paths["age_model"], model_paths["age_proto"])
        return face_net, age_net
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        raise

def detect_face_and_age(image, face_net, age_net, min_confidence=0.5):
    frame = np.array(image.convert('RGB'))
    h, w = frame.shape[:2]
    result_img = frame.copy()
    
    # Detect faces
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                               (104.0, 177.0, 123.0), swapRB=False)
    face_net.setInput(blob)
    detections = face_net.forward()
    
    faces_found = 0
    face_data = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > min_confidence:
            faces_found += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            
            # Ensure box coordinates are within image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Extract face
            face = frame[y1:y2, x1:x2]
            if face.size == 0:  # Skip if face has no area
                continue
                
            # Predict age
            try:
                # Ensure face is not empty and has valid dimensions
                if face.shape[0] > 0 and face.shape[1] > 0:
                    blob_face = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                                  (78.426, 87.769, 114.896), swapRB=False)
                    age_net.setInput(blob_face)
                    age_preds = age_net.forward()
                    age_idx = age_preds[0].argmax()
                    age = age_list[age_idx]

                    # Draw bounding box and label
                    cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Create a filled rectangle for text background
                    text_size = cv2.getTextSize(f"Age: {age}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(result_img, 
                                 (x1, y1 - text_size[1] - 10), 
                                 (x1 + text_size[0], y1), 
                                 (0, 255, 0), -1)
                    
                    # Add age text
                    cv2.putText(result_img, f"Age: {age}", (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    # Save face data for display
                    face_data.append({
                        "age": age,
                        "confidence": f"{confidence:.2f}",
                        "position": (x1, y1, x2, y2)
                    })
            except Exception as e:
                st.error(f"Error processing face: {str(e)}")
                continue
    
    return result_img, faces_found, face_data

# Main app functionality
with st.sidebar:
    st.subheader("Settings")
    confidence_threshold = st.slider("Face Detection Confidence", 0.1, 1.0, 0.5, 0.05)
    
    st.markdown("""
    <div class="info-box">
    <h3>About this app</h3>
    <p>This application uses deep learning models to:</p>
    <ul>
    <li>Detect faces in images</li>
    <li>Estimate the age range of each detected face</li>
    </ul>
    <p>The models are based on OpenCV's DNN module and pre-trained caffe models.</p>
    </div>
    """, unsafe_allow_html=True)

# Load models
try:
    with st.spinner("Loading AI models..."):
        face_net, age_net = load_models()
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.error("Please check your internet connection and make sure you can access GitHub repositories.")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    
    # Process image
    with st.spinner("Detecting faces and estimating ages..."):
        start_time = time.time()
        result_img, faces_count, face_data = detect_face_and_age(
            image, face_net, age_net, min_confidence=confidence_threshold
        )
        process_time = time.time() - start_time
    
    with col2:
        st.subheader("Processed Image")
        st.image(result_img, use_column_width=True)
    
    # Display results
    st.subheader("Detection Results")
    st.write(f"⏱️ Processing time: {process_time:.2f} seconds")
    
    if faces_count > 0:
        st.write(f"🔍 Detected {faces_count} {'face' if faces_count == 1 else 'faces'}")
        
        # Display face data in a table
        face_table_data = []
        for i, face in enumerate(face_data):
            face_table_data.append({
                "Face #": i+1,
                "Age Range": face["age"],
                "Confidence": face["confidence"]
            })
        
        if face_table_data:
            st.table(face_table_data)
    else:
        st.warning("No faces detected in the image. Try adjusting the confidence threshold.")

    # Allow downloading the processed image
    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(result_rgb)
    
    # Create a byte buffer for the image
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    
    st.download_button(
        label="Download Processed Image",
        data=buf.getvalue(),
        file_name="processed_image.jpg",
        mime="image/jpeg"
    )
else:
    st.markdown("""
    <div class="info-box">
    <h3>How to use:</h3>
    <ol>
    <li>Upload an image using the file uploader above</li>
    <li>Wait for the AI to process the image</li>
    <li>View the detected faces and estimated ages</li>
    <li>Adjust the confidence threshold in the sidebar if needed</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Demo message
    st.info("Upload an image to get started!")

if __name__ == "__main__":
    # Add instructions for running in VSCode
    if len(sys.argv) <= 1:
        print("\n" + "="*50)
        print("FACE AND AGE DETECTION APP")
        print("="*50)
        print("\nTo run this application properly in VS Code:")
        print("1. Open a terminal in VS Code (Terminal > New Terminal)")
        print("2. Run the command: streamlit run app.py")
        print("\nNote: If you're seeing this message, you may have tried to run")
        print("      the script directly (e.g., with the Run button or F5).")
        print("      Streamlit apps need to be run using the 'streamlit run' command.")
        print("="*50 + "\n")
        
        # Try to launch streamlit automatically if this is run directly in VS Code
        try:
            print("Attempting to start the Streamlit app automatically...")
            # Get the full path to the current script
            script_path = os.path.abspath(__file__)
            subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', script_path])
            print(f"Started Streamlit for {script_path}")
            print("Check your browser, or look for a URL in the terminal output above.")
        except Exception as e:
            print(f"Error launching Streamlit automatically: {str(e)}")
            print("Please use the manual command above.")