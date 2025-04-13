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
from inference_utils import extract_frames, preprocess_frame_spatial, detect_face_regions

class SwinTransformerClassifier(nn.Module):
    """Swin Transformer model for deepfake classification"""
    
    def __init__(self, num_classes=2):
        super(SwinTransformerClassifier, self).__init__()
        import timm
        
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
        
        # Ensure correct feature processing
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

class SpatialAnalyzer:
    """Spatial analysis for deepfake detection using Swin Transformer"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the spatial analyzer
        
        Args:
            model_path: Path to the PyTorch model file (.pth)
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        st.write(f"Using device for spatial analysis: {self.device}")
        self.model = self._load_model()
    
    def _load_model(self) -> nn.Module:
        """
        Load the PyTorch Swin Transformer model from disk
        
        Returns:
            Loaded PyTorch model
        """
        try:
            # Initialize the model architecture
            model = SwinTransformerClassifier(num_classes=2)
            
            # Load the trained weights
            state_dict = torch.load(self.model_path, map_location=self.device)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            # Set the model to evaluation mode
            model.eval()
            model = model.to(self.device)
            
            # Test the model with a dummy input
            dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
            try:
                with torch.no_grad():
                    output = model(dummy_input)
                    st.write(f"Spatial model test output: {output}")
            except Exception as e:
                st.error(f"Spatial model test failed: {e}")
                return self._get_placeholder_model()

            return model
            
        except Exception as e:
            st.error(f"Error loading Spatial model: {e}")
            return self._get_placeholder_model()
    
    def _get_placeholder_model(self) -> nn.Module:
        """
        Create a placeholder model for testing when the real model isn't available
        
        Returns:
            Placeholder PyTorch model
        """
        try:
            import timm
            
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
    
    def analyze_frames(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze a list of frames for deepfake detection
        
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
        if not frames:
            status_text.text("No frames to analyze")
            if progress_bar:
                progress_bar.empty()
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
            if progress_bar:
                progress_bar.empty()
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
                status_text.text(f"Analyzing frame {frame_idx+1}/{len(frames)} with spatial model...")
                
                # Preprocess frame
                input_tensor = preprocess_frame_spatial(frame)
                if input_tensor is None:
                    continue
                    
                # Add batch dimension and move to device
                input_tensor = input_tensor.unsqueeze(0).to(self.device)
                
                # Get model prediction
                try:
                    output = self.model(input_tensor)
                    
                    # Apply softmax to get probabilities
                    softmax_probs = F.softmax(output, dim=1).cpu().numpy()[0]
                    
                    # Get fake and real probabilities
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
                    st.error(f"Error during spatial model inference: {e}")
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
                face_regions = detect_face_regions(frame)
                for region in face_regions:
                    detection_areas.append({
                        "frame_number": i,
                        "coordinates": region,
                        "confidence": float(1.0 - confidence_values[i])  # Deep fake confidence = 1 - authenticity
                    })
        
        # Sort and limit frames with detections to top 5
        frames_with_detections = sorted(frames_with_detections, key=lambda x: x[2], reverse=True)[:5]
        
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
        
        # Analyze the extracted frames
        result = self.analyze_frames(frames)
        
        # Add model type to result
        result["model_type"] = "spatial"
        
        return result