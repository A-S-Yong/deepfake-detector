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
        # Check if we're dealing with frame differences that need to be restructured
        if len(features.shape) == 2:  # [num_frames, flattened_pixels]
            # Convert to proper format for 3D CNN
            # First, reshape back to 3D (temporal, height, width)
            num_frames = features.shape[0]
            
            if features[0].shape[0] == 224*224*3:  # RGB flattened
                # Reshape to [frames, channels, height, width]
                features = features.reshape(num_frames, 3, 224, 224)
                # Transpose to [channels, frames, height, width]
                features = features.transpose(1, 0, 2, 3)
                # Add batch dimension
                features = np.expand_dims(features, 0)
            elif features[0].shape[0] == 224*224:  # Grayscale flattened
                # Reshape to [frames, height, width]
                features = features.reshape(num_frames, 224, 224)
                # Create three identical channels to match RGB input
                features = np.stack([features, features, features], axis=1)
                # Add batch dimension
                features = np.expand_dims(features, 0)
            else:
                # Handle more general case
                side_length = int(np.sqrt(features.shape[1]))
                features = features.reshape(num_frames, side_length, side_length)
                features = np.stack([features, features, features], axis=1)
                features = np.expand_dims(features, 0)
        
        # Convert numpy array to torch tensor if needed
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32)
        
        # Move to the model's device
        device = next(self.parameters()).device
        features = features.to(device)
        
        # Set model to evaluation mode
        self.eval()
        
        # Get predictions without gradient tracking
        with torch.no_grad():
            # Forward pass
            outputs = self(features)
            
            # Convert to probabilities
            fake_probs = outputs.cpu().numpy().flatten()
            real_probs = 1 - fake_probs
            
            # Return in the format expected by the analyzer
            return np.column_stack((fake_probs, real_probs))