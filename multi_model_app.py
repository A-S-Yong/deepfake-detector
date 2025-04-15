import pathlib, gdown, streamlit as st
import os
import sys
import time
import tempfile
import cv2
import numpy as np
import torch
import pickle
from PIL import Image
import uuid
from datetime import datetime

# Add the current directory to path
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom modules
from detectors.spatial import SpatialAnalyzer
from detectors.temporal import TemporalAnalyzer
from detectors.audio_visual import AudioVisualAnalyzer
from inference_utils import extract_video_metadata, format_file_size, format_duration

# Model file ID mappings
MODEL_FILE_IDS = {
    "spatial": "19JPcDF8NEJFkFt41WTBbw3TjEDrxq0Xz",  # Swin Transformer model
    "temporal": "1e2Vf3nJpMdwWKLT1b3FHPPq5bX4muRv6",   # Replace with actual file ID
    "audio_visual": "1j_0jJWOxUGkJKRCGxrdR2HFzt5csKwF0"  # Replace with actual file ID
}

MODEL_PATHS = {
    "spatial": "model/best_swin_transformer_model.pth",
    "temporal": "model/deepfake_cnn_model.pkl",
    "audio_visual": "model/audio-visual.pth"
}

# Add model downloading functionality
@st.cache_resource(show_spinner=False)
def download_model(model_type: str) -> pathlib.Path:
    """
    Ensure the model weight file is on disk and return its Path.
    The file is pulled once from Google Drive and then cached by Streamlit.
    
    Args:
        model_type: Type of model to download ('spatial', 'temporal', or 'audio_visual')
        
    Returns:
        Path to the downloaded model file
    """
    FILE_ID = MODEL_FILE_IDS.get(model_type)
    if not FILE_ID:
        raise ValueError(f"Unknown model type: {model_type}")
        
    URL = f"https://drive.google.com/uc?id={FILE_ID}"
    weight_path = pathlib.Path(MODEL_PATHS.get(model_type))
    weight_path.parent.mkdir(parents=True, exist_ok=True)

    if not weight_path.exists():
        with st.spinner(f"Downloading {model_type.title()} model weights…"):
            dst = gdown.download(URL, str(weight_path), quiet=False)
            if dst is None or not weight_path.exists():
                raise RuntimeError("Download failed or Drive link is private.")
        st.success(f"{model_type.title()} model downloaded successfully.")

    return weight_path

# Function to display detection results
def display_detection_results(results, video_metadata):
    """Display deepfake detection results in a visually appealing format"""
    is_deepfake = results["is_deepfake"]
    confidence = results["confidence"]
    processing_time = results["processing_time"]
    frames_analyzed = results["frames_analyzed"]
    model_type = results.get("model_type", "unknown")
    analysis_mode = results.get("analysis_mode", None)
    is_placeholder = results.get("is_placeholder", False)
    
    # Display a warning if using a placeholder model
    if is_placeholder:
        st.warning("⚠️ **Using placeholder model for demonstration purposes only. Results are not reliable!**")
    
    # Determine the analysis mode string to display
    display_model_type = model_type.upper()
    if model_type == "audio_visual" and analysis_mode:
        if analysis_mode == "visual-only":
            display_model_type = "AUDIO-VISUAL (VISUAL-ONLY MODE)"
    
    # Result header with model type
    if is_deepfake:
        st.markdown('<div class="result-box result-box-fake">', unsafe_allow_html=True)
        st.markdown(f'<p class="big-font">⚠️ DEEPFAKE DETECTED ({display_model_type} ANALYSIS)</p>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box result-box-real">', unsafe_allow_html=True)
        st.markdown(f'<p class="big-font">✅ NO DEEPFAKE DETECTED ({display_model_type} ANALYSIS)</p>', unsafe_allow_html=True)
    
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
        
        # Update model type display to show analysis mode
        if model_type == "audio_visual" and analysis_mode == "visual-only":
            st.markdown(f"• Model type: {model_type.title()} (Visual-only mode)", unsafe_allow_html=True)
        else:
            st.markdown(f"• Model type: {model_type.title()}", unsafe_allow_html=True)
    
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
    
    # Add a note about analysis mode if it's audio-visual but running in visual-only mode
    if model_type == "audio_visual" and analysis_mode == "visual-only":
        st.info("Note: This video doesn't contain audio or audio extraction failed. Analysis was performed using visual information only.")
    
    # Display detection areas
    if is_deepfake and "frames_with_detections" in results and results["frames_with_detections"]:
        st.subheader("Frames with Detected Manipulations")
        
        # Get frames with highest fake probability
        frames_to_show = results["frames_with_detections"]
        
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
    
    # Interpretation based on model type
    st.subheader("Analysis Interpretation")
    
    # Add disclaimer for placeholder model
    if is_placeholder:
        st.warning("**Model Status:** Using placeholder model. The actual model could not be loaded correctly. Results should not be used for real analysis.")
    
    if is_deepfake:
        if model_type == "spatial":
            if confidence > 0.8:
                st.warning("High probability that this video has been manipulated. The spatial analysis has detected strong visual indicators of deepfake technology.")
            else:
                st.info("Some visual indicators of manipulation were detected in individual frames.")
        elif model_type == "temporal":
            if confidence > 0.8:
                st.warning("High probability that this video has been manipulated. The temporal analysis has detected unnatural motion patterns between frames.")
            else:
                st.info("Some temporal inconsistencies were detected that may indicate manipulation.")
        elif model_type == "audio_visual":
            if analysis_mode == "visual-only":
                if confidence > 0.8:
                    st.warning("High probability that this video has been manipulated. The visual analysis has detected strong indicators of deepfake technology.")
                else:
                    st.info("Some visual indicators of manipulation were detected.")
            else:
                if confidence > 0.8:
                    st.warning("High probability that this video has been manipulated. The audio-visual analysis has detected inconsistencies between the audio and visual components.")
                else:
                    st.info("Some minor audio-visual misalignments were detected that may indicate manipulation.")
        st.info("Note: This is an automated analysis and should be used as one of multiple verification methods.")
    else:
        if model_type == "spatial":
            st.success("The video appears visually authentic with no significant signs of manipulation detected.")
        elif model_type == "temporal":
            st.success("The motion patterns in this video appear natural with no temporal inconsistencies detected.")
        elif model_type == "audio_visual":
            if analysis_mode == "visual-only":
                st.success("The video appears visually authentic with no significant signs of manipulation detected.")
            else:
                st.success("The audio and visual components of this video appear consistent with no significant misalignments detected.")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

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
.model-select {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 20px;
    border: 1px solid #dee2e6;
}
</style>
""", unsafe_allow_html=True)

# App title
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://img.icons8.com/color/96/000000/spy-male--v1.png", width=80)
with col2:
    st.title("Deepfake Detector")
    st.markdown("Upload a video to detect potential AI-generated manipulations using multiple analysis methods")

# Sidebar for app navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload Video", "Analysis History", "About Us"])

if page == "Home":
    st.markdown("## See beyond the surface, uncover digital truth with our advanced Deepfake Detector!")
    
    # Added a compatibility notice
    st.markdown("""
    <div class="compatibility-notice">
        <h4>📹 Video Compatibility Notice</h4>
        <p>Videos will only be displayed if they are browser compatible. For best results:</p>
        <ul>
            <li><strong>File format:</strong> Convert your videos to .mp4 format</li>
            <li><strong>Video codec:</strong> Use H.264 or HEVC (H.265) encoding</li>
            <li><strong>Maximum file size:</strong> 200MB</li>
        </ul>
        <p>Incompatible videos may still be analyzed, but preview functionality will be limited.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Our Multi-Model Approach")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Spatial Analysis")
        st.markdown("Examines individual frames for visual inconsistencies using a Swin Transformer model.")
        st.markdown("**Best for:** Detecting visual artifacts, face swaps, and unrealistic features.")
    
    with col2:
        st.markdown("#### Temporal Analysis")
        st.markdown("Analyzes motion and consistency between frames over time.")
        st.markdown("**Best for:** Detecting unnatural movements, flickering, and temporal inconsistencies.")
    
    with col3:
        st.markdown("#### Audio-Visual Analysis")
        st.markdown("Examines the relationship between audio and visual components.")
        st.markdown("**Best for:** Detecting lip-sync issues and audio-visual misalignments.")
    
    st.markdown("### How to Use")
    st.markdown("1. Navigate to 'Upload Video' using the sidebar")
    st.markdown("2. Upload your video file")
    st.markdown("3. Select which analysis model you want to use")
    st.markdown("4. View detailed results and save to your analysis history")

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
            
            # Model selection
            st.markdown('<div class="model-select">', unsafe_allow_html=True)
            st.subheader("Select Analysis Method")
            model_type = st.radio(
                "Choose which model to use for analysis:",
                ["Spatial Analysis", "Temporal Analysis", "Audio-Visual Analysis"],
                help="Each model specializes in different aspects of deepfake detection"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Run analysis button
            if st.button("Run Analysis", type="primary"):
                try:
                    # Initialize appropriate detector based on selection
                    with st.spinner(f"Loading {model_type.split()[0].lower()} detection model..."):
                        if "Spatial" in model_type:
                            model_path = download_model("spatial")
                            detector = SpatialAnalyzer(model_path)
                        elif "Temporal" in model_type:
                            model_path = download_model("temporal")
                            detector = TemporalAnalyzer(model_path)
                        else:  # Audio-Visual
                            model_path = download_model("audio_visual")
                            detector = AudioVisualAnalyzer(model_path)
                    
                    # Analyze for deepfakes
                    with st.spinner(f"Analyzing video using {model_type}..."):
                        results = detector.analyze_video(temp_file_path)
                    
                    # Display results
                    st.header("Detection Results")
                    display_detection_results(results, video_metadata)
                    
                    # Save to history with analysis mode information
                    history_entry = {
                        "id": str(uuid.uuid4()),
                        "filename": uploaded_file.name,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "is_deepfake": results["is_deepfake"],
                        "confidence": results["confidence"],
                        "processing_time": results["processing_time"],
                        "model_type": results.get("model_type", "unknown"),
                        "analysis_mode": results.get("analysis_mode", None)
                    }
                    st.session_state.history.insert(0, history_entry)
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
            
        except Exception as e:
            st.error(f"An error occurred while processing the video: {e}")
        finally:
            # Clean up the temp file when navigating away
            pass  # We'll clean up later to allow multiple analyses

elif page == "Analysis History":
    st.header("Analysis History")
    
    if not st.session_state.history:
        st.info("No analysis history found. Upload a video to get started!")
    else:
        # Display history
        for i, entry in enumerate(st.session_state.history):
            # Create display title
            display_title = f"{entry['filename']} - {entry['timestamp']}"
            
            with st.expander(display_title):
                col1, col2, col3, col4 = st.columns(4)
                
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
                    model_type = entry.get('model_type', 'unknown').title()
                    analysis_mode = entry.get('analysis_mode', None)
                    
                    st.write("**Model Used:**")
                    if model_type == "Audio_visual" and analysis_mode == "visual-only":
                        st.write(f"{model_type} (Visual-only mode)")
                    else:
                        st.write(f"{model_type}")
                
                with col4:
                    st.write("**Processing Time:**")
                    st.write(f"{entry['processing_time']:.2f} seconds")

elif page == "About Us":
    st.header("About Us")

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "© 2025 Deepfake Detector | "
    "This tool is for educational and verification purposes only."
)

# Clean up any temp files when the session ends
def cleanup_temp_files():
    """Clean up temporary files when the Streamlit session ends"""
    for file in tempfile.gettempdir():
        if file.startswith('deepfake_analysis_'):
            try:
                os.remove(file)
            except:
                pass

# Register the cleanup function
import atexit
atexit.register(cleanup_temp_files)