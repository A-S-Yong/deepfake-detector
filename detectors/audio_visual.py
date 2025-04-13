import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import streamlit as st
import time
from typing import List, Dict, Any, Tuple, Optional
import sys
import os
import torchaudio
import torchaudio.transforms as transforms
from torchvision import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 1. Implement the extract_frames function from the notebook
def extract_frames(video_path, num_frames=30):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract
        
    Returns:
        List of frames extracted from the video
    """
    cap = cv2.VideoCapture(video_path)
    frames = []  # Initialize frames here

    if not cap.isOpened():
        st.warning(f"Cannot open video {video_path}")
        return frames  # Return empty list

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        st.warning(f"No frames found in {video_path}")
        return frames

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Store the original frame for visualization
            frames.append(frame)

    cap.release()
    return frames

# 2. Implement preprocess_frame_spatial function
def preprocess_frame_spatial(frame):
    """
    Preprocess a frame for the spatial CNN model
    
    Args:
        frame: Input frame (numpy array)
        
    Returns:
        Preprocessed frame tensor
    """
    try:
        # Resize to model input size
        frame_resized = cv2.resize(frame, (224, 224))
        
        # Convert to RGB if it's BGR
        if frame_resized.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame_resized
            
        # Normalize pixel values
        frame_normalized = frame_rgb / 255.0
        
        # Convert to tensor and change to (C, H, W) format
        frame_tensor = torch.from_numpy(frame_normalized).float().permute(2, 0, 1)
        
        return frame_tensor
    except Exception as e:
        st.error(f"Error preprocessing frame: {e}")
        return None

# 3. Implement the extract_audio_features function using extract_spectrogram
def extract_audio_features(video_path, device, n_mels=128, max_length=128):
    """
    Extract audio features from a video file
    
    Args:
        video_path: Path to the video file
        device: Device to run the model on ('cuda' or 'cpu')
        n_mels: Number of mel bands
        max_length: Maximum length of the spectrogram
        
    Returns:
        Audio features tensor
    """
    try:
        # Extract audio from video to a temporary file
        audio_path = video_path.replace('.mp4', '.wav')
        if not os.path.exists(audio_path):
            # If audio file doesn't exist, extract it from video
            import subprocess
            command = f"ffmpeg -i {video_path} -y -ab 160k -ac 2 -ar 44100 -vn {audio_path}"
            subprocess.call(command, shell=True)
        
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if waveform.shape[1] == 0:
            st.warning(f"Empty audio in {video_path}")
            return torch.zeros(1, 1024, device=device)
        
        # Create mel spectrogram
        transform = transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
        spectrogram = transform(waveform)
        
        # Apply log scaling
        spectrogram = torch.log1p(spectrogram)
        spectrogram = spectrogram.squeeze(0)  # Remove extra channel
        
        # Normalize
        spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-8)
        
        # Ensure fixed size
        if spectrogram.shape[1] < max_length:
            pad = torch.zeros((n_mels, max_length - spectrogram.shape[1]))
            spectrogram = torch.cat((spectrogram, pad), dim=1)
        else:
            spectrogram = spectrogram[:, :max_length]
            
        # Reshape to match model input (convert to 1D feature vector of size 1024)
        # This is an approximation - adjust based on your actual model architecture
        spectrogram = spectrogram.flatten()[:1024]
        if spectrogram.shape[0] < 1024:
            pad_size = 1024 - spectrogram.shape[0]
            spectrogram = torch.cat([spectrogram, torch.zeros(pad_size)])
            
        return spectrogram.unsqueeze(0).to(device)
        
    except Exception as e:
        st.error(f"Error extracting audio features: {e}")
        return torch.zeros(1, 1024, device=device)

# 4. Implement the detect_face_regions function
def detect_face_regions(frame):
    """
    Detect face regions in a frame
    
    Args:
        frame: Input frame
        
    Returns:
        List of face regions (x, y, w, h)
    """
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Load face cascade (you need to have this file available)
        face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        face_regions = []
        for (x, y, w, h) in faces:
            face_regions.append([int(x), int(y), int(w), int(h)])
            
        # If no faces detected, return a default region (center of the frame)
        if len(face_regions) == 0:
            h, w = frame.shape[:2]
            center_x = w // 4
            center_y = h // 4
            face_regions.append([center_x, center_y, w // 2, h // 2])
            
        return face_regions
    
    except Exception as e:
        st.error(f"Error detecting face regions: {e}")
        h, w = frame.shape[:2]
        return [[0, 0, w, h]]  # Return full frame as fallback

# 5. Update the AudioVisualModel to align with the notebook's architecture
class AudioVisualModel(nn.Module):
    """Audio-Visual model for deepfake detection"""
    
    def __init__(self):
        super().__init__()
        # Visual feature extractor (using ResNet instead of simple CNN)
        self.visual_features = nn.Sequential(
            *list(models.resnet18(pretrained=True).children())[:-1],
            nn.Flatten(),
            nn.Linear(512, 512),  # Match the notebook's CNNFeatureExtractor output size
            nn.ReLU(),
        )
        
        # Audio feature extractor 
        self.audio_features = nn.Sequential(
            nn.Linear(1024, 512),  # Simplified version of the notebook's AudioTransformer
            nn.ReLU(),
        )
        
        # Combined features
        self.combined = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def forward(self, visual, audio=None):
        # If audio is None, generate dummy features
        if audio is None:
            audio = torch.zeros(visual.shape[0], 1024, device=visual.device)
        
        # Extract features
        vis_feat = self.visual_features(visual)
        aud_feat = self.audio_features(audio)
        
        # Combine features
        combined = torch.cat([vis_feat, aud_feat], dim=1)
        
        # Final prediction
        output = self.combined(combined)
        return output

# The rest of your AudioVisualAnalyzer class can remain the same