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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference_utils import extract_frames, preprocess_frame_spatial, extract_audio_features, detect_face_regions

class AudioVisualModel(nn.Module):
    """Audio-Visual model for deepfake detection"""
    
    def __init__(self):
        super().__init__()
        # Visual feature extractor
        self.visual_features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
        )
        
        # Audio feature extractor (would connect to audio spectrogram)
        self.audio_features = nn.Sequential(
            nn.Linear(1024, 512),  # Assume audio spectrogram features
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
        Load the Audio-Visual model from disk
        
        Returns:
            Loaded PyTorch model
        """
        try:
            # Initialize the model architecture
            model = AudioVisualModel()
            
            # Load weights if file exists
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                st.write("Audio-Visual model loaded from saved weights")
            except Exception as e:
                st.warning(f"Using untrained Audio-Visual model (for demonstration): {e}")
            
            # Set to evaluation mode
            model.eval()
            model = model.to(self.device)
            
            # Test with dummy input
            dummy_visual = torch.randn(1, 3, 224, 224, device=self.device)
            dummy_audio = torch.randn(1, 1024, device=self.device)
            
            with torch.no_grad():
                output = model(dummy_visual, dummy_audio)
                st.write(f"Audio-Visual model test output: {output}")
                
            return model
            
        except Exception as e:
            st.error(f"Error loading Audio-Visual model: {e}")
            return self._get_placeholder_model()
    
    def _get_placeholder_model(self) -> nn.Module:
        """
        Create a placeholder audio-visual model for testing
        
        Returns:
            Placeholder PyTorch model
        """
        class PlaceholderAVModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(1, 2)
                
            def forward(self, visual, audio=None):
                batch_size = visual.shape[0]
                # Return random logits for testing
                logits = torch.randn(batch_size, 2, device=visual.device)
                return logits
        
        model = PlaceholderAVModel().to(self.device)
        model.eval()
        return model
    
    def analyze_video(self, video_path: str, max_frames: int = 30) -> Dict[str, Any]:
        """
        Analyze a video file to detect if it's a deepfake using audio-visual analysis
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            Dictionary with detection results including confidence score
        """
        start_time = time.time()
        
        # Progress placeholder for Streamlit
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Extract frames from the video
        status_text.text("Extracting frames for audio-visual analysis...")
        frames = extract_frames(video_path, max_frames)
        progress_bar.progress(0.3)
        
        if not frames:
            status_text.text("No frames could be extracted")
            progress_bar.empty()
            return {
                "is_deepfake": False,
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "frames_analyzed": 0,
                "detection_areas": [],
                "model_type": "audio_visual"
            }
        
        # Extract audio features
        status_text.text("Extracting audio features...")
        progress_bar.progress(0.5)
        audio_features = extract_audio_features(video_path, self.device)
        
        # Analyze frames with audio-visual model
        predictions = []
        confidence_scores = []
        frames_with_detections = []
        
        status_text.text("Running audio-visual analysis...")
        progress_bar.progress(0.7)
        
        with torch.no_grad():
            for frame_idx, frame in enumerate(frames):
                # Update progress
                progress_percentage = 0.7 + (0.3 * (frame_idx + 1) / len(frames))
                progress_bar.progress(progress_percentage)
                
                # Preprocess frame
                visual_features = preprocess_frame_spatial(frame)
                if visual_features is None:
                    continue
                
                # Add batch dimension and move to device
                visual_features = visual_features.unsqueeze(0).to(self.device)
                
                try:
                    # Run model inference
                    output = self.model(visual_features, audio_features)
                    
                    # Apply softmax to get probabilities
                    probs = F.softmax(output, dim=1).cpu().numpy()[0]
                    
                    # Get fake and real probabilities
                    fake_prob = probs[0]
                    real_prob = probs[1]
                    
                    # Determine if frame is fake
                    is_fake = fake_prob > real_prob
                    
                    # Store prediction and confidence
                    predictions.append(is_fake)
                    confidence_scores.append(real_prob)
                    
                    # Store frames with fake detection for visualization
                    if is_fake:
                        frames_with_detections.append((frame_idx, frame, fake_prob))
                        
                except Exception as e:
                    st.error(f"Error during audio-visual inference: {e}")
                    continue
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # If no valid predictions, return default result
        if not predictions:
            return {
                "is_deepfake": False,
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "frames_analyzed": len(frames),
                "detection_areas": [],
                "model_type": "audio_visual"
            }
        
        # Calculate overall results
        fake_count = sum(predictions)
        total_count = len(predictions)
        
        # A video is considered fake if more than 25% of frames are detected as fake
        is_deepfake = fake_count / total_count > 0.25
        
        # Calculate confidence score
        avg_real_confidence = sum(confidence_scores) / len(confidence_scores)
        confidence = 1.0 - avg_real_confidence if is_deepfake else avg_real_confidence
        
        # Get top 5 frames with highest fake probability
        frames_with_detections = sorted(frames_with_detections, key=lambda x: x[2], reverse=True)[:5]
        
        # Create detection areas
        detection_areas = []
        if is_deepfake and frames_with_detections:
            for frame_idx, frame, prob in frames_with_detections:
                face_regions = detect_face_regions(frame)
                for region in face_regions:
                    detection_areas.append({
                        "frame_number": frame_idx,
                        "coordinates": region,
                        "confidence": float(prob)
                    })
        
        # Final result
        result = {
            "is_deepfake": is_deepfake,
            "confidence": float(confidence),
            "processing_time": time.time() - start_time,
            "frames_analyzed": len(frames),
            "detection_areas": detection_areas,
            "frames_with_detections": frames_with_detections,
            "model_type": "audio_visual"
        }
        
        return result