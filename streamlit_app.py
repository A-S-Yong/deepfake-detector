import pathlib, gdown, streamlit as st
import os
import requests
import sys
import time
import tempfile
import cv2
import numpy as np
import torch
from PIL import Image
import uuid
import io
from datetime import datetime
import json

# Add the current directory to path
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add model downloading functionality
@st.cache_resource(show_spinner=False)
def download_model() -> pathlib.Path:
    """
    Ensure the Swin‑Transformer weight file is on disk and return its Path.
    The file is pulled once from Google Drive and then cached by Streamlit.
    """
    FILE_ID = "19JPcDF8NEJFkFt41WTBbw3TjEDrxq0Xz"
    URL     = f"https://drive.google.com/uc?id={FILE_ID}"

    weight_path = pathlib.Path("model/best_swin_transformer_model.pth")
    weight_path.parent.mkdir(parents=True, exist_ok=True)

    if not weight_path.exists():
        with st.spinner("Downloading Swin‑Transformer weights…"):
            dst = gdown.download(URL, str(weight_path), quiet=False)
            if dst is None or not weight_path.exists():
                raise RuntimeError("Download failed or Drive link is private.")
        st.success("Model downloaded successfully.")

    return weight_path

# Import the DeepfakeDetector class from the detector module
class DeepfakeDetector:
    """Service for detecting deepfakes in video files using the loaded Swin Transformer model"""
    
    def __init__(self, device: str | None = None):
        """
        Initialize the deepfake detector with the PyTorch Swin Transformer model
        
        Args:
            model_path: Path to the PyTorch model file (.pth)
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.model_path = download_model()        # <‑‑ ensures file exists
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        st.write(f"Using device: {self.device}")
        self.model = self._load_model()
    
    def _load_model(self):
        """Load the PyTorch Swin Transformer model from disk"""
        try:
            # Import required libraries
            import timm
            import torch.nn as nn
            
            # Define Swin Transformer model architecture
            class SwinTransformerClassifier(nn.Module):
                def __init__(self, num_classes=2):
                    super(SwinTransformerClassifier, self).__init__()
                    self.model = timm.create_model("swin_base_patch4_window7_224", pretrained=True)

                    # Freeze initial layers
                    for name, param in self.model.named_parameters():
                        if "layers.0" in name or "layers.1" in name:
                            param.requires_grad = False

                    # Unfreeze last few layers
                    for param in self.model.head.parameters():
                        param.requires_grad = True

                    # Get correct input size for classifier head
                    in_features = self.model.num_features
                    
                    # Fix: Ensure correct feature processing
                    self.pooling = nn.AdaptiveAvgPool1d(1)
                    self.dropout = nn.Dropout(0.3)
                    self.fc = nn.Linear(in_features, num_classes)

                def forward(self, x):
                    x = self.model.forward_features(x)
                    x = x.mean(dim=[1, 2])
                    x = x.view(x.shape[0], -1)
                    x = self.dropout(x)
                    x = self.fc(x)
                    return x
            
            # Initialize the model architecture
            model = SwinTransformerClassifier(num_classes=2)
            
            # Load the trained weights
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            
            # Set the model to evaluation mode
            model.eval()
            model = model.to(self.device)
            
            # Test the model with a dummy input
            dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
            try:
                with torch.no_grad():
                    output = model(dummy_input)
                    st.write(f"Model test output: {output}")
            except Exception as e:
                st.error(f"Model test failed: {e}")

            return model
            
        except Exception as e:
            st.error(f"Error loading Swin Transformer model: {e}")
            # Return a placeholder model for development/testing
            return self._get_placeholder_model()
    
    def _get_placeholder_model(self):
        """
        Create a placeholder model for testing when the real model isn't available
        This is just for development - replace with actual model loading logic
        """
        try:
            import timm
            import torch.nn as nn
            
            # Create a simple classifier using timm's base model but with minimal setup
            class PlaceholderSwinModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    # Use the smallest Swin model for faster performance during testing
                    self.backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False)
                    self.fc = nn.Linear(768, 2)  # Swin tiny has 768 features
                    self.softmax = nn.Softmax(dim=1)
                
                def forward(self, x):
                    with torch.no_grad():
                        # Extract features
                        features = self.backbone.forward_features(x)
                        # Global pooling
                        features = features.mean(dim=[1, 2])
                        # Classification
                        logits = self.fc(features)
                        # Return probabilities
                        return self.softmax(logits)
            
            model = PlaceholderSwinModel().to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            st.error(f"Error creating placeholder model: {e}")
            return None
    
    def _extract_frames(self, video_path, max_frames=30):
        """
        Extract frames from a video file
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of frames as numpy arrays
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error(f"Error: Cannot open video file {video_path}")
            return frames
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            st.error(f"Error: Video has no frames or frame count couldn't be determined")
            return frames
        
        # Calculate frame sampling interval
        interval = max(1, total_frames // max_frames)
        
        frame_count = 0
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % interval == 0:
                # Verify frame integrity
                if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
                    frames.append(frame)
                else:
                    st.warning(f"Warning: Invalid frame at position {frame_count}")
            
            frame_count += 1
        
        cap.release()
        st.write(f"Extracted {len(frames)} frames from video")
        return frames
    
    def _preprocess_frame(self, frame):
        """
        Preprocess a frame for model input
        
        Args:
            frame: Raw frame as numpy array
            
        Returns:
            Processed frame as PyTorch tensor
        """
        try:
            # Ensure frame is a valid numpy array
            if frame is None or not isinstance(frame, np.ndarray):
                st.error(f"Invalid frame type: {type(frame)}")
                return None
                
            # Check if frame has valid dimensions
            if frame.ndim != 3 or frame.shape[2] != 3:
                st.error(f"Invalid frame shape: {frame.shape}")
                return None
                
            # Resize to expected input size
            resized = cv2.resize(frame, (224, 224))
            
            # Convert to RGB if needed
            if frame.shape[2] == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                
            # Normalize pixel values
            normalized = resized / 255.0
            
            # Convert to PyTorch tensor and add batch dimension
            tensor = torch.from_numpy(normalized).float()
            tensor = tensor.permute(2, 0, 1)  # HWC to CHW format
            tensor = tensor.unsqueeze(0)  # Add batch dimension
            tensor = tensor.to(self.device)
            
            return tensor
        except Exception as e:
            st.error(f"Error preprocessing frame: {e}")
            return None
    
    def _detect_face_regions(self, frame):
        """
        Detect face regions in a frame
        This is a placeholder - replace with actual face detection logic
        
        Args:
            frame: Video frame as numpy array
            
        Returns:
            List of face bounding boxes as dictionaries
        """
        # This is a placeholder - in a real implementation you would use
        # a face detection model like dlib, OpenCV's face detector, or
        # a more advanced deep learning model
        
        # For this example, we'll return a dummy face region
        # in the center of the frame
        h, w = frame.shape[0], frame.shape[1]
        center_x, center_y = w // 2, h // 2
        
        return [
            {
                "x1": max(0, center_x - w//4),
                "y1": max(0, center_y - h//4),
                "x2": min(w, center_x + w//4),
                "y2": min(h, center_y + h//4)
            }
        ]
    
    def analyze_video(self, video_path):
        """
        Analyze a video file to detect if it's a deepfake
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with detection results including confidence score
        """
        start_time = time.time()
        
        # Progress placeholder for Streamlit
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Extract frames from the video
        status_text.text("Extracting frames from video...")
        frames = self._extract_frames(video_path)
        
        if not frames:
            status_text.text("No frames could be extracted")
            return {
                "is_deepfake": False,
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "frames_analyzed": 0,
                "detection_areas": []
            }
        
        # Verify the model is valid
        if not self.model or not hasattr(self.model, 'forward'):
            status_text.text("Error: Model not properly initialized")
            return {
                "is_deepfake": False,
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "frames_analyzed": len(frames),
                "detection_areas": []
            }
        
        # Process frames with the model
        predictions = []
        confidence_values = []
        fake_probs = []  # Store fake probabilities for debugging
        real_probs = []  # Store real probabilities for debugging
        valid_frames = 0
        frames_with_detections = []
        
        # Use torch.no_grad() to disable gradient calculation for inference
        with torch.no_grad():
            for frame_idx, frame in enumerate(frames):
                # Update progress
                progress_percentage = (frame_idx + 1) / len(frames)
                progress_bar.progress(progress_percentage)
                status_text.text(f"Analyzing frame {frame_idx+1}/{len(frames)}...")
                
                # Convert frame to PyTorch tensor
                input_tensor = self._preprocess_frame(frame)
                
                # Skip invalid frames
                if input_tensor is None:
                    continue
                
                # Get model prediction
                try:
                    output = self.model(input_tensor)
                    
                    # Force softmax if needed
                    import torch.nn.functional as F
                    softmax_probs = F.softmax(output, dim=1).cpu().numpy()[0]
                    
                    # Use the softmax probabilities for better calibration
                    real_prob = softmax_probs[1]  # Probability of being real
                    fake_prob = softmax_probs[0]  # Probability of being fake
                    
                    # Store both probabilities for debugging
                    real_probs.append(real_prob)
                    fake_probs.append(fake_prob)
                    
                    # Determine if deepfake (higher fake probability means it's more likely a deepfake)
                    pred = fake_prob > real_prob
                    
                    # Store the prediction and confidence
                    predictions.append(pred)
                    confidence_values.append(real_prob)  # Store real probability as our confidence
                    
                    # If this frame is predicted as fake, store it for visualization
                    if pred:
                        frames_with_detections.append((frame_idx, frame, fake_prob))
                    
                    valid_frames += 1
                    
                except Exception as e:
                    st.error(f"Error during model inference: {e}")
                    continue
        
        # Clear the progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # If no valid frames were processed, return default result
        if not valid_frames:
            return {
                "is_deepfake": False,
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "frames_analyzed": len(frames),
                "detection_areas": [],
                "frames_with_detections": []
            }
        
        # Calculate overall confidence
        avg_confidence = sum(confidence_values) / len(confidence_values)
        
        # Ensure confidence is within proper range (0-1)
        avg_confidence = max(0.0, min(1.0, avg_confidence))
        
        # Determine if the video is a deepfake based on authenticity confidence
        # Lower authentic confidence (< 0.5) means it's more likely a deepfake
        is_deepfake = avg_confidence < 0.5
        
        # Report the confidence of our prediction, not the authenticity score
        # If is_deepfake is True, we should report (1 - avg_confidence) as our confidence
        # If is_deepfake is False, we should report avg_confidence as our confidence
        prediction_confidence = 1.0 - avg_confidence if is_deepfake else avg_confidence
        
        # Calculate the processing time
        processing_time = time.time() - start_time
        
        # Create detection areas
        detection_areas = []
        for i, frame in enumerate(frames):
            if i < len(predictions) and predictions[i]:
                # If this frame is predicted as deepfake
                face_regions = self._detect_face_regions(frame)
                for region in face_regions:
                    detection_areas.append({
                        "frame_number": i,
                        "coordinates": region,
                        "confidence": float(1.0 - confidence_values[i])  # Deep fake confidence = 1 - authenticity
                    })
        
        # Final result with detailed information
        result = {
            "is_deepfake": is_deepfake,
            "confidence": float(prediction_confidence),  # Now represents prediction confidence
            "processing_time": processing_time,
            "frames_analyzed": valid_frames,
            "detection_areas": detection_areas,
            "frames_with_detections": frames_with_detections
        }
        
        return result


# Set page config
st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🕵️",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
}
.result-box {
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.result-box-real {
    background-color: #d1e7dd;
    border: 1px solid #badbcc;
}
.result-box-fake {
    background-color: #f8d7da;
    border: 1px solid #f5c2c7;
}
.metadata-card {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #dee2e6;
}
.confidence-meter {
    height: 20px;
    border-radius: 10px;
    margin-top: 10px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Function to format file size
def format_file_size(size_in_bytes):
    """Format file size in bytes to human-readable format"""
    if size_in_bytes < 1024:
        return f"{size_in_bytes} bytes"
    elif size_in_bytes < 1024 * 1024:
        return f"{size_in_bytes / 1024:.2f} KB"
    elif size_in_bytes < 1024 * 1024 * 1024:
        return f"{size_in_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_in_bytes / (1024 * 1024 * 1024):.2f} GB"

# Function to format duration
def format_duration(seconds):
    """Format duration in seconds to MM:SS format"""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes}:{remaining_seconds:02d}"

# Function to extract video metadata
def extract_video_metadata(video_path):
    """Extract metadata from a video file"""
    metadata = {}
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return metadata
        
        # Get basic properties
        metadata["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        metadata["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        metadata["fps"] = cap.get(cv2.CAP_PROP_FPS)
        metadata["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate duration
        if metadata["fps"] > 0 and metadata["frame_count"] > 0:
            duration_sec = metadata["frame_count"] / metadata["fps"]
            metadata["duration"] = format_duration(duration_sec)
        
        # Get file size
        metadata["file_size"] = format_file_size(os.path.getsize(video_path))
        
        cap.release()
        
    except Exception as e:
        st.error(f"Error extracting video metadata: {e}")
    
    return metadata

# Function to display detection results
def display_detection_results(results, video_metadata):
    """Display deepfake detection results in a visually appealing format"""
    is_deepfake = results["is_deepfake"]
    confidence = results["confidence"]
    processing_time = results["processing_time"]
    frames_analyzed = results["frames_analyzed"]
    
    # Result header
    if is_deepfake:
        st.markdown('<div class="result-box result-box-fake">', unsafe_allow_html=True)
        st.markdown('<p class="big-font">⚠️ DEEPFAKE DETECTED</p>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box result-box-real">', unsafe_allow_html=True)
        st.markdown('<p class="big-font">✅ NO DEEPFAKE DETECTED</p>', unsafe_allow_html=True)
    
    # Confidence percentage
    confidence_percent = confidence * 100
    st.markdown(f"<p>Detection confidence: <strong>{confidence_percent:.2f}%</strong></p>", unsafe_allow_html=True)
    
    # Confidence meter
    if is_deepfake:
        st.markdown(
            f'<div style="background-color: #dc3545; width: {confidence_percent}%; height: 20px;" class="confidence-meter"></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div style="background-color: #198754; width: {confidence_percent}%; height: 20px;" class="confidence-meter"></div>',
            unsafe_allow_html=True
        )
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<strong>Analysis Details:</strong>", unsafe_allow_html=True)
        st.markdown(f"• Frames analyzed: {frames_analyzed}", unsafe_allow_html=True)
        st.markdown(f"• Processing time: {processing_time:.2f} seconds", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<strong>Video Properties:</strong>", unsafe_allow_html=True)
        if video_metadata:
            if "width" in video_metadata and "height" in video_metadata:
                st.markdown(f"• Resolution: {video_metadata['width']} × {video_metadata['height']}", unsafe_allow_html=True)
            if "duration" in video_metadata:
                st.markdown(f"• Duration: {video_metadata['duration']}", unsafe_allow_html=True)
            if "fps" in video_metadata:
                st.markdown(f"• Frame rate: {video_metadata['fps']:.2f} fps", unsafe_allow_html=True)
    
    # Close result box
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display detection areas
    if is_deepfake and results["frames_with_detections"]:
        st.subheader("Frames with Detected Manipulations")
        
        # Get up to 5 frames with highest fake probability
        frames_to_show = sorted(results["frames_with_detections"], key=lambda x: x[2], reverse=True)[:5]
        
        cols = st.columns(min(len(frames_to_show), 5))
        for i, (frame_idx, frame, prob) in enumerate(frames_to_show):
            if i < len(cols):
                with cols[i]:
                    # Convert frame to RGB for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Draw a red box around the center area as a placeholder for face detection
                    h, w = frame.shape[0], frame.shape[1]
                    cv2.rectangle(
                        frame_rgb,
                        (w//4, h//4),
                        (w*3//4, h*3//4),
                        (255, 0, 0),
                        2
                    )
                    st.image(frame_rgb, caption=f"Frame {frame_idx}: {prob*100:.1f}% fake")
    
    # Interpretation
    st.subheader("Analysis Interpretation")
    if is_deepfake:
        if confidence > 0.8:
            st.warning("High probability that this video has been manipulated. The AI has detected strong indicators of deepfake technology.")
        elif confidence > 0.6:
            st.info("Moderate indicators of potential manipulation were found in this video.")
        else:
            st.info("Some minor signs of manipulation detected, but with lower confidence.")
        st.info("Note: This is an automated analysis and should be used as one of multiple verification methods.")
    else:
        st.success("The video appears to be authentic with no significant signs of manipulation detected.")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# App title
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://img.icons8.com/color/96/000000/spy-male--v1.png", width=80)
with col2:
    st.title("Deepfake Detector")
    st.markdown("Upload a video to detect potential AI-generated facial manipulations")

# Sidebar for app navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload Video", "Analysis History", "About"])

if page == "Home":
    st.markdown("## See beyond the surface, uncover digital truth with Deepfake Detector!")

elif page == "Upload Video":
    st.header("Upload Video for Analysis")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded_file is not None:
        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": format_file_size(uploaded_file.size),
            "Uploaded at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Create expandable section for file details
        with st.expander("File Information"):
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")
        
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name
        
        try:
            # Display video preview
            st.subheader("Video Preview")
            st.video(temp_file_path)
            
            # Extract video metadata
            video_metadata = extract_video_metadata(temp_file_path)
            
            # Initialize detector
            with st.spinner("Loading deepfake detection model..."):
                detector = DeepfakeDetector()
            
            # Analyze for deepfakes
            with st.spinner("Analyzing video for deepfakes..."):
                results = detector.analyze_video(temp_file_path)
            
            # Display results
            st.header("Detection Results")
            display_detection_results(results, video_metadata)
            
            # Save to history
            history_entry = {
                "id": str(uuid.uuid4()),
                "filename": uploaded_file.name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "is_deepfake": results["is_deepfake"],
                "confidence": results["confidence"],
                "processing_time": results["processing_time"]
            }
            st.session_state.history.insert(0, history_entry)
            
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
        finally:
            # Clean up the temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

elif page == "Analysis History":
    st.header("Analysis History")
    
    if not st.session_state.history:
        st.info("No analysis history found. Upload a video to get started!")
    else:
        # Display history
        for i, entry in enumerate(st.session_state.history):
            with st.expander(f"{entry['filename']} - {entry['timestamp']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Result:**")
                    if entry["is_deepfake"]:
                        st.markdown("⚠️ **DEEPFAKE DETECTED**")
                    else:
                        st.markdown("✅ **NO DEEPFAKE DETECTED**")
                
                with col2:
                    st.write("**Confidence:**")
                    st.progress(entry["confidence"])
                    st.write(f"{entry['confidence'] * 100:.2f}%")
                
                with col3:
                    st.write("**Processing Time:**")
                    st.write(f"{entry['processing_time']:.2f} seconds")

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "© 2025 Deepfake Detector | "
    "This tool is for educational and verification purposes only."
)