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

# 5. Update the DeepfakeDetector to align with the notebook's architecture
class DeepfakeDetector(nn.Module):
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

# Add AudioVisualAnalyzer class that follows the same interface as other analyzers
class AudioVisualAnalyzer:
    """Audio-Visual analysis for deepfake detection"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the audio-visual analyzer
        
        Args:
            model_path: Path to the PyTorch model file (.pth)
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        st.write(f"Using device for audio-visual analysis: {self.device}")
        self.model = self._load_model()
    
    def _load_model(self) -> nn.Module:
        """
        Load the PyTorch audio-visual model from disk
        
        Returns:
            Loaded PyTorch model
        """
        try:
            # Initialize the model architecture
            model = DeepfakeDetector().to(self.device)
            
            # Load the trained weights
            state_dict = torch.load(self.model_path, map_location=self.device)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            # Set the model to evaluation mode
            model.eval()
            
            # Test the model with a dummy input
            try:
                with torch.no_grad():
                    visual_input = torch.randn(1, 3, 224, 224, device=self.device)
                    audio_input = torch.randn(1, 1024, device=self.device)
                    output = model(visual_input, audio_input)
                    st.write(f"Audio-visual model test output: {output}")
            except Exception as e:
                st.error(f"Audio-visual model test failed: {e}")
                return self._get_placeholder_model()

            return model
            
        except Exception as e:
            st.error(f"Error loading Audio-visual model: {e}")
            return self._get_placeholder_model()
    
    def _get_placeholder_model(self) -> nn.Module:
        """
        Create a placeholder model for testing when the real model isn't available
        
        Returns:
            Placeholder PyTorch model
        """
        class PlaceholderAVModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(1, 2)
                self.softmax = nn.Softmax(dim=1)
            
            def forward(self, visual, audio=None):
                batch_size = visual.shape[0]
                x = torch.ones((batch_size, 1), device=visual.device)
                logits = self.fc(x)
                return self.softmax(logits)
        
        model = PlaceholderAVModel().to(self.device)
        model.eval()
        return model
    
    def analyze_frames(self, frames: List[np.ndarray], video_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a list of frames for deepfake detection
        
        Args:
            frames: List of video frames as numpy arrays
            video_path: Optional path to the video file for audio extraction
            
        Returns:
            Dictionary with detection results
        """
        start_time = time.time()
        
        # Progress placeholder for Streamlit
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Check if frames are available
        if not frames:
            status_text.text("No frames to analyze")
            if progress_bar:
                progress_bar.empty()
            return {
                "is_deepfake": False,
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "frames_analyzed": 0,
                "detection_areas": [],
                "frames_with_detections": []
            }
        
        # Verify the model is valid
        if not self.model or not hasattr(self.model, 'forward'):
            status_text.text("Error: Model not properly initialized")
            if progress_bar:
                progress_bar.empty()
            return {
                "is_deepfake": False,
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "frames_analyzed": len(frames),
                "detection_areas": [],
                "frames_with_detections": []
            }
        
        # Process frames with the model
        predictions = []
        confidence_values = []
        valid_frames = 0
        frames_with_detections = []
        
        # Use torch.no_grad() to disable gradient calculation for inference
        with torch.no_grad():
            # Extract audio features if video_path is provided
            audio_features = None
            if video_path:
                status_text.text("Extracting audio features...")
                progress_bar.progress(0.1)
                try:
                    audio_features = extract_audio_features(video_path, self.device)
                except Exception as e:
                    st.warning(f"Could not extract audio features: {e}. Proceeding with visual only.")
            
            for frame_idx, frame in enumerate(frames):
                # Update progress
                progress_percentage = 0.2 + (frame_idx + 1) / len(frames) * 0.8
                progress_bar.progress(progress_percentage)
                status_text.text(f"Analyzing frame {frame_idx+1}/{len(frames)} with audio-visual model...")
                
                # Preprocess frame
                visual_tensor = preprocess_frame_spatial(frame)
                if visual_tensor is None:
                    continue
                    
                # Add batch dimension and move to device
                visual_tensor = visual_tensor.unsqueeze(0).to(self.device)
                
                # Get model prediction
                try:
                    output = self.model(visual_tensor, audio_features)
                    
                    # Apply softmax to get probabilities if not already applied
                    if not torch.allclose(output.sum(dim=1), torch.tensor(1.0, device=output.device)):
                        softmax_probs = F.softmax(output, dim=1).cpu().numpy()[0]
                    else:
                        softmax_probs = output.cpu().numpy()[0]
                    
                    # Get fake and real probabilities
                    fake_prob = softmax_probs[0]  # Probability of being fake
                    real_prob = softmax_probs[1]  # Probability of being real
                    
                    # Determine if deepfake
                    pred = fake_prob > real_prob
                    
                    # Store the prediction and confidence
                    predictions.append(pred)
                    confidence_values.append(fake_prob if pred else real_prob)
                    
                    # If this frame is predicted as fake, store it for visualization
                    if pred:
                        frames_with_detections.append((frame_idx, frame, fake_prob))
                    
                    valid_frames += 1
                    
                except Exception as e:
                    st.error(f"Error during audio-visual model inference: {e}")
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
                "frames_analyzed": 0,
                "detection_areas": [],
                "frames_with_detections": []
            }
        
        # Calculate overall confidence
        avg_confidence = sum(confidence_values) / len(confidence_values)
        
        # Determine if the video is a deepfake based on majority voting
        num_deepfake_frames = sum(predictions)
        is_deepfake = num_deepfake_frames > (valid_frames / 2)
        
        # Calculate the processing time
        processing_time = time.time() - start_time
        
        # Create detection areas
        detection_areas = []
        for i, frame in enumerate(frames):
            if i < len(predictions) and predictions[i]:
                # If this frame is predicted as deepfake
                face_regions = detect_face_regions(frame)
                for region in face_regions:
                    detection_areas.append({
                        "frame_number": i,
                        "coordinates": region,
                        "confidence": float(confidence_values[i])
                    })
        
        # Sort and limit frames with detections to top 5
        frames_with_detections = sorted(frames_with_detections, key=lambda x: x[2], reverse=True)[:5]
        
        # Final result
        result = {
            "is_deepfake": is_deepfake,
            "confidence": float(avg_confidence),
            "processing_time": processing_time,
            "frames_analyzed": valid_frames,
            "detection_areas": detection_areas,
            "frames_with_detections": frames_with_detections
        }
        
        return result

    def analyze_video(self, video_path: str, max_frames: int = 30) -> Dict[str, Any]:
        """
        Analyze a video file to detect if it's a deepfake
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            Dictionary with detection results including confidence score
        """
        # Extract frames from video
        frames = extract_frames(video_path, max_frames)
        
        # Analyze the extracted frames, passing the video_path for audio extraction
        result = self.analyze_frames(frames, video_path)
        
        # Add model type to result
        result["model_type"] = "audio_visual"
        
        return result