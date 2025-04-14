import cv2
import numpy as np
import torch
import os
import tempfile
from typing import List, Dict, Any, Tuple, Optional
import streamlit as st
import torchaudio
import torchaudio.transforms as transforms

def extract_frames(video_path: str, max_frames: int = 30) -> List[np.ndarray]:
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

def preprocess_frame_spatial(frame: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> Optional[torch.Tensor]:
    """
    Preprocess a frame for spatial model input
    
    Args:
        frame: Raw frame as numpy array
        target_size: Target size for resizing (width, height)
        
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
        resized = cv2.resize(frame, target_size)
        
        # Convert to RGB if needed
        if frame.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
        # Normalize pixel values
        normalized = resized / 255.0
        
        # Convert to PyTorch tensor and add batch dimension
        tensor = torch.from_numpy(normalized).float()
        tensor = tensor.permute(2, 0, 1)  # HWC to CHW format
        
        return tensor
    except Exception as e:
        st.error(f"Error preprocessing frame: {e}")
        return None

def preprocess_frame_temporal(frame: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
    """
    Preprocess a frame for temporal model input
    
    Args:
        frame: Raw frame as numpy array
        target_size: Target size for resizing (width, height)
        
    Returns:
        Processed frame as numpy array in the format expected by the temporal model
    """
    try:
        # Resize to the target size expected by the model
        resized = cv2.resize(frame, target_size)
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to [0, 1]
        normalized = rgb_frame.astype(np.float32) / 255.0
        
        return normalized
    except Exception as e:
        st.error(f"Error preprocessing frame for temporal analysis: {e}")
        return None

def extract_audio_features(video_path: str, device: str) -> torch.Tensor:
    audio_path = video_path.rsplit('.', 1)[0] + '.wav'
    spectrogram = extract_spectrogram(audio_path)  # shape: (128, 128)
    # Convert to tensor and add channel dimension (expected: [batch, 1, H, W])
    audio_tensor = torch.tensor(spectrogram, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    return audio_tensor

def extract_spectrogram(audio_path, n_mels=128, max_length=128):
    """
    Extract a log-scaled Mel spectrogram from the given audio file.

    Args:
        audio_path (str): Path to the audio file.
        n_mels (int): Number of Mel frequency bins to use.
        max_length (int): Maximum length (number of time frames) for the spectrogram.

    Returns:
        np.ndarray: A spectrogram of shape (n_mels, max_length), or a blank spectrogram if processing fails.
    """
    try:
        # Load the audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Check if the audio is empty
        if waveform.shape[1] == 0:
            print(f"Warning: Empty audio in {audio_path}")
            return np.zeros((n_mels, max_length))
        
        # Create a MelSpectrogram transform and apply it
        mel_transform = transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
        spectrogram = mel_transform(waveform)
        
        # Apply logarithmic scaling
        spectrogram = torch.log1p(spectrogram)
        
        # Remove any extra channel if needed (assuming mono channel here)
        spectrogram = spectrogram.squeeze(0)
        
        # Normalize the spectrogram
        spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()
        
        # Adjust the spectrogram size: pad or truncate to max_length
        if spectrogram.shape[1] < max_length:
            padding = torch.zeros((n_mels, max_length - spectrogram.shape[1]))
            spectrogram = torch.cat((spectrogram, padding), dim=1)
        else:
            spectrogram = spectrogram[:, :max_length]
        
        return spectrogram.numpy()
    
    except Exception as e:
        print(f"Error processing audio {audio_path}: {e}")
        return np.zeros((n_mels, max_length))


def detect_face_regions(frame: np.ndarray) -> List[Dict[str, int]]:
    """
    Detect face regions in a frame (placeholder implementation)
    
    Args:
        frame: Video frame as numpy array
        
    Returns:
        List of face bounding boxes as dictionaries
    """
    # This is a placeholder - in a real implementation you would use
    # a face detection model like dlib, OpenCV's face detector, or
    # a more advanced deep learning model
    
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

def format_file_size(size_in_bytes: int) -> str:
    """Format file size in bytes to human-readable format"""
    if size_in_bytes < 1024:
        return f"{size_in_bytes} bytes"
    elif size_in_bytes < 1024 * 1024:
        return f"{size_in_bytes / 1024:.2f} KB"
    elif size_in_bytes < 1024 * 1024 * 1024:
        return f"{size_in_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_in_bytes / (1024 * 1024 * 1024):.2f} GB"

def format_duration(seconds: float) -> str:
    """Format duration in seconds to MM:SS format"""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes}:{remaining_seconds:02d}"

def extract_video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Extract metadata from a video file
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary containing video metadata
    """
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

def save_temp_video(uploaded_file) -> str:
    """
    Save an uploaded file to a temporary file
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Path to the temporary file
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name