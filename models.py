import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Deepfake3DCNN(nn.Module):
    def __init__(self):
        super(Deepfake3DCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(3, 3, 3), stride=(1,1,1), padding=(1,1,1))
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=(1,1,1), padding=(1,1,1))
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), stride=(1,1,1), padding=(1,1,1))
        self.conv4 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3), stride=(1,1,1), padding=(1,1,1))
        self.conv5 = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(3, 3, 3), stride=(1,1,1), padding=(1,1,1))
        # Max pooling layer
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2,2,2), ceil_mode=True)
        # Fully connected layers
        self.fc1 = nn.Linear(25088, 512)
        self.fc2 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        # Flatten the output
        x = x.view(x.size(0), -1)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x
    
    def predict_proba(self, features):
        """
        Add a predict_proba method to make the model compatible with the TemporalAnalyzer class.
        This method handles the conversion from a sequence of frame features to the
        format expected by the 3D CNN.
        
        Args:
            features: Input features representing frame differences or features
            
        Returns:
            Array of shape (n_samples, 2) with probabilities for fake (0) and real (1) classes
        """
        # Prepare the features for the 3D CNN (batch, channels, temporal_frames, height, width)
        try:
            # First, check to see what format we have and adapt accordingly
            if isinstance(features, np.ndarray):
                # Print debug info to help with troubleshooting
                print(f"Input features shape: {features.shape}, type: {features.dtype}")
                
                if len(features.shape) == 2:  # [num_frames, flattened_pixels]
                    # Features are flattened, need to reshape
                    num_frames = features.shape[0]
                    
                    if features.shape[1] == 224*224*3:  # RGB flattened
                        # Reshape to [frames, height, width, channels]
                        features = features.reshape(num_frames, 224, 224, 3)
                        # Transpose to [frames, channels, height, width]
                        features = features.transpose(0, 3, 1, 2)
                        # Now we have [frames, channels, height, width]
                    elif features.shape[1] == 224*224:  # Grayscale flattened
                        # Reshape to [frames, height, width]
                        features = features.reshape(num_frames, 224, 224)
                        # Create three identical channels for RGB input
                        features = np.stack([features, features, features], axis=1)
                        # Now we have [frames, channels=3, height, width]
                    else:
                        # Try to infer dimensions from the flattened size
                        side_length = int(np.sqrt(features.shape[1] / 3))
                        if side_length**2 * 3 == features.shape[1]:  # RGB
                            features = features.reshape(num_frames, side_length, side_length, 3)
                            features = features.transpose(0, 3, 1, 2)
                        else:
                            side_length = int(np.sqrt(features.shape[1]))
                            if side_length**2 == features.shape[1]:  # Grayscale
                                features = features.reshape(num_frames, side_length, side_length)
                                features = np.stack([features, features, features], axis=1)
                            else:
                                raise ValueError(f"Cannot determine frame dimensions from shape {features.shape}")
                    
                    # At this point we have [frames, channels, height, width]
                    # Put channels first for PyTorch's 3D CNN format: [channels, frames, height, width]
                    features = np.transpose(features, (1, 0, 2, 3))
                    # Add batch dimension
                    features = np.expand_dims(features, 0)
                    # Now we have [batch=1, channels=3, frames, height, width]
                
                elif len(features.shape) == 4:  # [num_frames, height, width, channels]
                    # Features already have spatial dimensions, just need to reorder
                    # From [frames, height, width, channels] to [frames, channels, height, width]
                    if features.shape[3] == 3:  # RGB
                        features = np.transpose(features, (0, 3, 1, 2))  # [frames, channels, height, width]
                        # Put channels first, then add batch dimension
                        features = np.transpose(features, (1, 0, 2, 3))  # [channels, frames, height, width]
                        features = np.expand_dims(features, 0)  # [batch, channels, frames, height, width]
                    else:
                        raise ValueError(f"Unexpected number of channels: {features.shape[3]}")
                
                elif len(features.shape) == 5:  # Already 5D, but might need reordering
                    # Check if dimensions are [batch, frames, height, width, channels]
                    if features.shape[4] == 3:  # Last dimension is channels
                        # Reorder to [batch, channels, frames, height, width]
                        features = np.transpose(features, (0, 4, 1, 2, 3))
                    # Otherwise assume it's already in the correct format
                
                # Convert to torch tensor after reshaping
                features = torch.tensor(features, dtype=torch.float32)
            
            # Move to the model's device
            device = next(self.parameters()).device
            features = features.to(device)
            
            # Set model to evaluation mode
            self.eval()
            
            # Get predictions without gradient tracking
            with torch.no_grad():
                # Print tensor shape before forward pass for debugging
                print(f"Tensor shape before forward pass: {features.shape}")
                
                # Forward pass
                outputs = self(features)
                
                # Convert to probabilities
                fake_probs = outputs.cpu().numpy().flatten()
                real_probs = 1 - fake_probs
                
                # Return in the format expected by the analyzer
                return np.column_stack((fake_probs, real_probs))
                
        except Exception as e:
            print(f"Error in predict_proba: {e}")
            # Return default values on error to prevent crashing
            return np.array([[0.5, 0.5]])