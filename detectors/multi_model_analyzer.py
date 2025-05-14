import torch
import numpy as np
import streamlit as st
import time
from typing import List, Dict, Any, Optional
import sys
import os
import concurrent.futures
from pathlib import Path

# Add the current directory to path
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import analyzers
from detectors.spatial import SpatialAnalyzer
from detectors.audio_visual import AudioVisualAnalyzer
from inference_utils import extract_frames, extract_video_metadata

class MultiModelAnalyzer:
    """Ensemble analyzer that combines results from multiple deepfake detection models"""
    
    def __init__(self, model_paths: Dict[str, str], device: Optional[str] = None):
        """
        Initialize the multi-model analyzer
        
        Args:
            model_paths: Dictionary with model paths for each detector type
                         (keys: 'spatial', 'audio_visual')
            device: Device to run models on ('cuda' or 'cpu')
        """
        self.model_paths = model_paths
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        st.write(f"Using device for multi-model analysis: {self.device}")
        
        # Initialize individual analyzers
        self.analyzers = {}
        self.is_placeholder = False
        
        try:
            self.analyzers['spatial'] = SpatialAnalyzer(model_paths['spatial'], self.device)
            self.analyzers['audio_visual'] = AudioVisualAnalyzer(model_paths['audio_visual'], self.device)
        except Exception as e:
            st.error(f"Error initializing analyzers: {e}")
            self.is_placeholder = True
    
    def analyze_video(self, video_path: str, max_frames: int = 30, 
                      weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Analyze a video using both models and combine the results
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to extract
            weights: Dictionary with weights for each model type 
                    (keys: 'spatial', 'audio_visual')
                    Default weights will be used if not provided
            
        Returns:
            Dictionary with combined detection results
        """
        start_time = time.time()
        
        # Default weights if not provided
        if weights is None:
            weights = {
                'spatial': 0.5,   # Spatial analysis often provides good per-frame accuracy
                'audio_visual': 0.5  # Audio-visual examines audio-visual synchronization
            }
        
        # Use weights directly as they're already normalized in the UI
        normalized_weights = weights.copy() 
        
        # Extract metadata once
        video_metadata = extract_video_metadata(video_path)
        
        # Run individual analyzers
        results = {}
        
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Running multiple deepfake detection models...")
        
        # Run analyzers one by one (could be parallelized with ThreadPoolExecutor)
        for i, (model_type, analyzer) in enumerate(self.analyzers.items()):
            model_status = st.empty()
            model_status.text(f"Running {model_type} analysis...")
            
            # Update progress
            progress_percentage = i / len(self.analyzers)
            progress_bar.progress(progress_percentage)
            
            # Run the analyzer
            try:
                results[model_type] = analyzer.analyze_video(video_path, max_frames)
            except Exception as e:
                st.error(f"Error in {model_type} analysis: {e}")
                results[model_type] = {
                    "is_deepfake": False,
                    "confidence": 0.0,
                    "processing_time": 0.0,
                    "frames_analyzed": 0,
                    "model_type": model_type,
                    "is_placeholder": True
                }
            
            model_status.empty()
        
        # Calculate combined confidence score and final determination
        combined_confidence = 0.0
        weighted_deepfake_score = 0.0
        combined_detection_areas = []
        combined_frames_with_detections = []
        total_processing_time = 0.0
        total_frames_analyzed = 0
        all_placeholder = True
        
        # Process individual results
        for model_type, result in results.items():
            # Get weight for this model
            weight = normalized_weights.get(model_type, 0.0)
            
            # Extract confidence and add to weighted score
            confidence = result.get("confidence", 0.0)
            is_deepfake = result.get("is_deepfake", False)
            is_placeholder = result.get("is_placeholder", False)
            
            # If any model is not a placeholder, then the ensemble is not all placeholders
            if not is_placeholder:
                all_placeholder = False
            
            # For deepfake, contribution is positive; for real, it's negative
            deepfake_score = confidence if is_deepfake else -confidence
            weighted_deepfake_score += deepfake_score * weight
            
            # Accumulate processing time and frames analyzed
            total_processing_time += result.get("processing_time", 0.0)
            total_frames_analyzed += result.get("frames_analyzed", 0)
            
            # Collect detection areas
            if "detection_areas" in result:
                # Add model type to each detection area
                for area in result["detection_areas"]:
                    area["model_type"] = model_type
                combined_detection_areas.extend(result["detection_areas"])
            
            # Collect frames with detections
            if "frames_with_detections" in result:
                for frame_data in result["frames_with_detections"]:
                    # Make sure all frame data has model_type
                    # Check if frame_data already has model_type (is a 4-tuple)
                    if len(frame_data) == 3:
                        # frame_data is a tuple of (frame_idx, frame, prob)
                        combined_frames_with_detections.append((frame_data[0], frame_data[1], frame_data[2], model_type))
                    else:
                        # Already has model_type or other format, add as is
                        combined_frames_with_detections.append(frame_data)
        
        # Final determination based on weighted score
        final_is_deepfake = weighted_deepfake_score > 0
        
        # Calculate combined confidence as absolute value of weighted score, scaled to [0,1]
        # Clamp to [0,1] in case weights cause it to exceed 1
        combined_confidence = min(1.0, abs(weighted_deepfake_score))
        
        # Sort frames with detections by probability and take top 5
        combined_frames_with_detections = sorted(
            combined_frames_with_detections, 
            key=lambda x: x[2],  # Sort by probability
            reverse=True
        )[:5]
        
        # Combine all results
        combined_result = {
            "is_deepfake": final_is_deepfake,
            "confidence": float(combined_confidence),
            "processing_time": total_processing_time,
            "frames_analyzed": total_frames_analyzed,
            "detection_areas": combined_detection_areas,
            "frames_with_detections": combined_frames_with_detections,
            "model_type": "ensemble",
            "individual_results": results,
            "weights": normalized_weights,
            "is_placeholder": all_placeholder,
            "weighted_deepfake_score": weighted_deepfake_score
        }
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return combined_result