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
        self.model, self.is_placeholder = self._load_model()
    
    def _load_model(self):
        """
        Load the temporal model from disk, handling the fact that the
        pickle references main.Deepfake3DCNN (the class defined during
        training in the notebook).
        """
        try:
            # 1) Import the real class from models.py
            from models import Deepfake3DCNN

            # 2) Tell PyTorch that this class is safe to un‑pickle
            from torch.serialization import add_safe_globals
            add_safe_globals([Deepfake3DCNN])

            # 3) Create a fake module called "main" (and "__main__" for safety)
            import types, sys
            fake_main = types.ModuleType("main")
            fake_main.Deepfake3DCNN = Deepfake3DCNN
            sys.modules["main"] = fake_main
            sys.modules["__main__"] = fake_main   # covers both spellings

            # 4) Now load the full object pickle
            model = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=False,              # full‑object load
            )

            model.eval()
            model.to(self.device)
            st.write("Temporal model loaded successfully")
            
            # Test the model with dummy data
            try:
                dummy_features = np.zeros((10, 3, 224, 224), dtype=np.float32)
                dummy_output = model.predict_proba(dummy_features)
                st.write(f"Model test output shape: {dummy_output.shape}, values: min={dummy_output.min()}, max={dummy_output.max()}")
            except Exception as test_error:
                st.error(f"Model test failed: {test_error}")
            
            return model, False

        except Exception as e:
            st.error(f"Error loading Temporal model: {e}")
            st.warning("⚠️ Using placeholder model for demonstration purposes only. Results will not be accurate!")
            return self._get_placeholder_model(), True

    def _get_placeholder_model(self):
        """
        Create a placeholder temporal model for testing that returns more balanced predictions
        
        Returns:
            Placeholder model with predict_proba method
        """
        # Simple class to mimic the behavior of the real model but return balanced predictions
        class PlaceholderTemporalModel:
            def predict_proba(self, features):
                # Return balanced probabilities to avoid biased results
                batch_size = features.shape[0]
                # For most frames, predict "real" with high confidence
                fake_probs = np.ones(batch_size) * 0.2  # Assume most frames look real
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
            
        # Progress indicator
        progress_text = st.empty()
        progress_text.text("Processing frames for temporal analysis...")
            
        # Preprocess each frame - use the imported preprocess_frame_temporal function
        processed_frames = []
        for i, frame in enumerate(frames):
            processed = preprocess_frame_temporal(frame)
            if processed is not None:
                processed_frames.append(processed)
            
            # Update progress every few frames
            if i % 5 == 0:
                progress_text.text(f"Processing frames: {i+1}/{len(frames)}")
                
        if not processed_frames:
            progress_text.text("Failed to extract features from frames")
            return None
            
        progress_text.text("Calculating temporal differences between frames...")
        
        # Calculate frame differences
        features = np.array(processed_frames)
        
        # Clear progress indicator
        progress_text.empty()
        
        return features
    
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
                "frames_with_detections": [],
                "is_placeholder": True,
                "debug_info": {"error": "Not enough frames"}
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
                "frames_with_detections": [],
                "is_placeholder": True,
                "debug_info": {"error": "Failed to extract features"}
            }
        
        # Get model predictions
        status_text.text("Running temporal model inference...")
        progress_bar.progress(0.5)  # Update progress
        
        try:
            # Predict with the temporal model
            probabilities = self.model.predict_proba(features)
            
            # Add debug info
            fake_probs_mean = np.mean(probabilities[:, 0])
            real_probs_mean = np.mean(probabilities[:, 1])
            debug_info = {
                "probabilities_shape": probabilities.shape,
                "fake_probs_mean": float(fake_probs_mean),
                "real_probs_mean": float(real_probs_mean),
                "fake_probs_min": float(np.min(probabilities[:, 0])),
                "fake_probs_max": float(np.max(probabilities[:, 0])),
                "real_probs_min": float(np.min(probabilities[:, 1])),
                "real_probs_max": float(np.max(probabilities[:, 1])),
            }
            
            # Display debug info
            st.write("Debug info - Model probabilities:")
            st.write(f"- Average fake probability: {fake_probs_mean:.4f}")
            st.write(f"- Average real probability: {real_probs_mean:.4f}")
            
            # IMPORTANT FIX: Force lower threshold for deepfake detection to prevent false positives
            DEEPFAKE_THRESHOLD = 0.65  # Require higher confidence to classify as deepfake
            
            # Average the probabilities across frames
            avg_probs = np.mean(probabilities, axis=0)
            
            # Interpret results (assuming index 0 is fake, index 1 is real)
            fake_prob = avg_probs[0]
            real_prob = avg_probs[1]
            
            # Use a higher threshold for declaring deepfakes to reduce false positives
            is_deepfake = fake_prob > DEEPFAKE_THRESHOLD and fake_prob > real_prob
            
            # Calculate confidence
            confidence = fake_prob if is_deepfake else real_prob
            
            # Create frames with detections for visualization
            frames_with_detections = []
            
            # Update progress
            progress_bar.progress(0.8)
            status_text.text("Identifying frames with manipulations...")
            
            # Only consider frames as "detected" if they exceed the threshold
            for i, frame in enumerate(frames[1:]):  # Skip first frame
                if i < len(probabilities):
                    frame_fake_prob = probabilities[i][0]
                    if frame_fake_prob > DEEPFAKE_THRESHOLD:
                        frames_with_detections.append((i+1, frame, frame_fake_prob))
            
            # Get top 5 frames with highest fake probability
            frames_with_detections = sorted(frames_with_detections, key=lambda x: x[2], reverse=True)[:5]
            
        except Exception as e:
            st.error(f"Error during temporal model inference: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            
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
                "frames_with_detections": [],
                "is_placeholder": True,
                "debug_info": {"error": str(e), "traceback": traceback.format_exc()}
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
            "frames_with_detections": frames_with_detections,
            "is_placeholder": self.is_placeholder,
            "debug_info": debug_info
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