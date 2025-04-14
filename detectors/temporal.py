import torch
import numpy as np
import cv2
import streamlit as st
import time
import pickle
from typing import List, Dict, Any, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference_utils import extract_frames, preprocess_frame_temporal, detect_face_regions

class TemporalAnalyzer:
    """Temporal analysis for deepfake detection"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the temporal analyzer
        
        Args:
            model_path: Path to the model file (.pkl)
            device: Device to run any torch components on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        st.write(f"Using device for temporal analysis: {self.device}")
        self.model = self._load_model()
    
    def _load_model(self):
        try:
            # ① import the real class
            from models import Deepfake3DCNN                         # :contentReference[oaicite:0]{index=0}
            from torch.serialization import add_safe_globals
            add_safe_globals([Deepfake3DCNN])             # ← NEW (trust this class)

            # ② make a fake module named "main" that points to the same class
            import types, sys
            fake_main = types.ModuleType("main")
            fake_main.Deepfake3DCNN = Deepfake3DCNN
            sys.modules["main"] = fake_main        # ← alias injected

            # ③ now it will un‑pickle correctly
            model = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=False,                # full object load
            )

            model.eval(); model.to(self.device)
            st.write("Temporal model loaded successfully")
            return model

        except Exception as e:
            st.error(f"Error loading Temporal model: {e}")
            return self._get_placeholder_model()

    
    def _get_placeholder_model(self):
        """
        Create a placeholder temporal model for testing
        
        Returns:
            Placeholder model with predict_proba method
        """
        # Simple class to mimic the behavior of the real model
        class PlaceholderTemporalModel:
            def predict_proba(self, features):
                # Return fake probabilities for testing
                batch_size = features.shape[0]
                # First column is fake probability, second is real probability
                # For testing, we'll generate random probabilities that sum to 1
                fake_probs = np.random.rand(batch_size)
                real_probs = 1 - fake_probs
                return np.column_stack((fake_probs, real_probs))
                
        return PlaceholderTemporalModel()
    
    def _extract_temporal_features(self, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Extract temporal features from a sequence of frames
        
        Args:
            frames: List of video frames as numpy arrays
            
        Returns:
            Numpy array of temporal features or None if extraction fails
        """
        if len(frames) < 2:
            st.error("Need at least 2 frames for temporal analysis")
            return None
            
        # Preprocess each frame
        processed_frames = []
        for frame in frames:
            processed = preprocess_frame_temporal(frame)
            if processed is not None:
                processed_frames.append(processed)
                
        if not processed_frames:
            return None
            
        # Calculate frame differences (simple temporal feature)
        features = []
        for i in range(1, len(processed_frames)):
            diff = processed_frames[i] - processed_frames[i-1]
            features.append(diff)
            
        return np.array(features)
    
    def analyze_frames(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze a list of frames for deepfake detection using temporal analysis
        
        Args:
            frames: List of video frames as numpy arrays
            
        Returns:
            Dictionary with detection results
        """
        start_time = time.time()
        
        # Progress placeholder for Streamlit
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Check if frames are available
        if not frames or len(frames) < 2:
            status_text.text("Not enough frames for temporal analysis")
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
        
        # Extract temporal features
        status_text.text("Extracting temporal features...")
        progress_bar.progress(0.2)  # Update progress
        
        features = self._extract_temporal_features(frames)
        
        if features is None or len(features) == 0:
            status_text.text("Failed to extract temporal features")
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
        
        # Get model predictions
        status_text.text("Running temporal model inference...")
        progress_bar.progress(0.5)  # Update progress
        
        try:
            # Predict with the temporal model
            probabilities = self.model.predict_proba(features)
            
            # Average the probabilities across frames
            avg_probs = np.mean(probabilities, axis=0)
            
            # Interpret results (assuming class 1 is real, class 0 is fake)
            real_prob = avg_probs[1] if avg_probs.shape[0] > 1 else 1.0 - avg_probs[0]
            fake_prob = avg_probs[0] if avg_probs.shape[0] > 1 else avg_probs[0]
            
            # Determine if deepfake
            is_deepfake = fake_prob > real_prob
            
            # Calculate confidence
            confidence = fake_prob if is_deepfake else real_prob
            
            # Create frames with detections for visualization
            frames_with_detections = []
            
            # Update progress
            progress_bar.progress(0.8)
            status_text.text("Identifying frames with manipulations...")
            
            for i, frame in enumerate(frames[1:]):  # Skip first frame
                if i < len(probabilities):
                    frame_fake_prob = probabilities[i][0]
                    if frame_fake_prob > 0.5:
                        frames_with_detections.append((i+1, frame, frame_fake_prob))
            
            # Get top 5 frames with highest fake probability
            frames_with_detections = sorted(frames_with_detections, key=lambda x: x[2], reverse=True)[:5]
            
        except Exception as e:
            st.error(f"Error during temporal model inference: {e}")
            if progress_bar:
                progress_bar.empty()
            if status_text:
                status_text.empty()
            return {
                "is_deepfake": False,
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "frames_analyzed": len(frames),
                "detection_areas": [],
                "frames_with_detections": []
            }
        
        # Create detection areas
        detection_areas = []
        progress_bar.progress(0.9)
        
        if is_deepfake and frames_with_detections:
            for frame_idx, frame, prob in frames_with_detections:
                face_regions = detect_face_regions(frame)
                for region in face_regions:
                    detection_areas.append({
                        "frame_number": frame_idx,
                        "coordinates": region,
                        "confidence": float(prob)
                    })
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Calculate the processing time
        processing_time = time.time() - start_time
        
        # Final result
        result = {
            "is_deepfake": bool(is_deepfake),
            "confidence": float(confidence),
            "processing_time": processing_time,
            "frames_analyzed": len(frames),
            "detection_areas": detection_areas,
            "frames_with_detections": frames_with_detections
        }
        
        return result

    def analyze_video(self, video_path: str, max_frames: int = 30) -> Dict[str, Any]:
        """
        Analyze a video file to detect if it's a deepfake using temporal analysis
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to extract (temporal needs more frames)
            
        Returns:
            Dictionary with detection results including confidence score
        """
        # Extract frames from video
        frames = extract_frames(video_path, max_frames)
        
        # Analyze the extracted frames
        result = self.analyze_frames(frames)
        
        # Add model type to result
        result["model_type"] = "temporal"
        
        return result