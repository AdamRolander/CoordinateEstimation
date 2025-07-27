import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

class DepthDatasetBuilder:
    def __init__(self, output_base_dir="output"):
        self.output_base_dir = output_base_dir
        self.features = []
        self.targets = []
        self.scene_info = []
        
    def extract_features_from_metadata(self):
        """Extract features from all metadata.json files"""
        print("Extracting features from metadata files...")
        
        scene_dirs = [d for d in os.listdir(self.output_base_dir) 
                     if d.startswith('scene_') and os.path.isdir(os.path.join(self.output_base_dir, d))]
        scene_dirs.sort()
        
        for scene_dir in scene_dirs:
            metadata_path = os.path.join(self.output_base_dir, scene_dir, "metadata.json")
            
            if not os.path.exists(metadata_path):
                print(f"Warning: No metadata.json in {scene_dir}")
                continue
                
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Extract features for each cube
            for cube_name, cube_data in metadata['cubes'].items():
                features = self.extract_cube_features(cube_data, metadata)
                target = self.calculate_true_distance(cube_data, metadata)
                
                self.features.append(features)
                self.targets.append(target)
                self.scene_info.append({
                    'scene_id': metadata['scene_id'],
                    'cube_name': cube_name,
                    'scene_dir': scene_dir
                })
        
        print(f"Extracted {len(self.features)} samples from {len(scene_dirs)} scenes")
        return np.array(self.features), np.array(self.targets)
    
    def extract_cube_features(self, cube_data, metadata):
        """Extract features for a single cube"""
        # Core geometric features
        mask_pixel_count = cube_data['mask_pixel_count']
        mask_coverage_percent = cube_data['mask_coverage_percent']
        object_true_size = cube_data['actual_size']
        
        # Position features (normalized to image center)
        img_width, img_height = metadata['camera']['resolution']
        obj_x, obj_y, obj_z = cube_data['location']
        
        # Project 3D position to 2D using camera parameters
        cam_matrix = np.array(metadata['camera']['pose_matrix'])
        intrinsics = np.array(metadata['camera']['intrinsic_matrix'])
        
        # Transform to camera coordinates
        world_pos = np.array([obj_x, obj_y, obj_z, 1.0])
        cam_pos = cam_matrix @ world_pos
        
        # Project to image coordinates
        if cam_pos[2] > 0:  # Avoid division by zero
            pixel_x = (intrinsics[0, 0] * cam_pos[0]) / cam_pos[2] + intrinsics[0, 2]
            pixel_y = (intrinsics[1, 1] * cam_pos[1]) / cam_pos[2] + intrinsics[1, 2]
        else:
            pixel_x, pixel_y = img_width/2, img_height/2
        
        # Normalize pixel coordinates
        norm_pixel_x = pixel_x / img_width
        norm_pixel_y = pixel_y / img_height
        
        # Camera parameters
        focal_length = metadata['camera']['focal_length_mm']
        
        # Object properties
        object_scale = cube_data['scale']
        
        # IMPROVED FEATURES - these should be much more predictive
        
        # 1. Direct analytical estimate (this should be highly correlated with true depth)
        if mask_pixel_count > 0:
            # Approximate object pixel radius
            pixel_radius = np.sqrt(mask_pixel_count / np.pi)
            # Analytical depth estimate using pinhole camera model
            analytical_depth = (object_true_size * focal_length * img_width) / (pixel_radius * 2 * 36)  # 36mm sensor
        else:
            analytical_depth = 10.0  # fallback
        
        # 2. Angular size features
        angular_size = mask_pixel_count / (img_width * img_height)  # fraction of image
        
        # 3. Distance from image center (objects further from center might have distortion)
        center_x, center_y = img_width/2, img_height/2
        distance_from_center = np.sqrt((pixel_x - center_x)**2 + (pixel_y - center_y)**2)
        norm_distance_from_center = distance_from_center / np.sqrt(center_x**2 + center_y**2)
        
        # 4. Aspect ratio-related features (cubes should be roughly square when viewed)
        # This requires calculating bounding box, let's approximate
        if mask_pixel_count > 0:
            estimated_bbox_side = np.sqrt(mask_pixel_count)
            bbox_aspect_ratio = 1.0  # For cubes, should be close to 1
        else:
            estimated_bbox_side = 0
            bbox_aspect_ratio = 1.0
        
        # 5. Scale-normalized features
        scale_normalized_pixel_count = mask_pixel_count / (object_scale ** 2)

        # Get new bounding box data from metadata
        bbox_width = cube_data['bbox_width']
        bbox_height = cube_data['bbox_height']
        pixel_density = cube_data['pixel_density']
        bbox_aspect_ratio = cube_data['bbox_aspect_ratio']

        # New, improved analytical depth calculation using bbox_width
        # We will use this as our main baseline.
        if bbox_width > 0:
            # analytical_depth = (object_true_size * focal_length * img_width) / (bbox_width * 36)
            analytical_depth = (object_true_size * focal_length * img_width) / (np.sqrt(bbox_width * bbox_height) * 36)
        else:
            analytical_depth = 10.0 # fallback
        
        features = [
            # Core geometric
            mask_pixel_count,
            mask_coverage_percent,
            object_true_size,

            # Bounding Box Features
            bbox_width,
            bbox_height,
            bbox_aspect_ratio,
            pixel_density,

            # Object properties
            object_scale,

            # ANALYTICAL FEATURE (this is our new, improved baseline)
            analytical_depth,

            # Other features (can be removed if desired, but may still be useful)
            norm_pixel_x,
            norm_pixel_y,
            norm_distance_from_center
        ]
        
        # features = [
        #     # Core geometric
        #     mask_pixel_count,
        #     mask_coverage_percent,
        #     object_true_size,
            
        #     # Position
        #     norm_pixel_x,
        #     norm_pixel_y,
        #     norm_distance_from_center,
            
        #     # Camera
        #     focal_length,
            
        #     # Object properties  
        #     object_scale,
            
        #     # ANALYTICAL FEATURES (these should be highly predictive)
        #     analytical_depth,  # Direct estimate
        #     angular_size,      # Alternative size measure
        #     scale_normalized_pixel_count,  # Scale-adjusted pixel count
        #     estimated_bbox_side,  # Estimated bounding box dimension
        # ]
        
        return features
    
    def calculate_true_distance(self, cube_data, metadata):
        """Calculate true distance from camera to object center"""
        obj_pos = np.array(cube_data['location'])
        cam_pos = np.array(metadata['camera']['location'])
        
        # Euclidean distance
        distance = np.linalg.norm(obj_pos - cam_pos)
        return distance
    
    def get_feature_names(self):
        return [
            'mask_pixel_count',
            'mask_coverage_percent',
            'object_true_size',
            'bbox_width',
            'bbox_height',
            'bbox_aspect_ratio',
            'pixel_density',
            'object_scale',
            'analytical_depth', # This is still at index 8!
            'norm_pixel_x',
            'norm_pixel_y',
            'norm_distance_from_center'
        ]
        # return [
        #     'mask_pixel_count',
        #     'mask_coverage_percent', 
        #     'object_true_size',
        #     'norm_pixel_x',
        #     'norm_pixel_y',
        #     'norm_distance_from_center',
        #     'focal_length',
        #     'object_scale',
        #     'analytical_depth',
        #     'angular_size',
        #     'scale_normalized_pixel_count',
        #     'estimated_bbox_side'
        # ]
    
    def create_dataframe(self):
        """Create a pandas DataFrame with all features and targets"""
        features, targets = self.extract_features_from_metadata()
        
        feature_names = self.get_feature_names()
        df = pd.DataFrame(features, columns=feature_names)
        df['target_distance'] = targets
        
        # Add scene info
        for i, info in enumerate(self.scene_info):
            df.loc[i, 'scene_id'] = info['scene_id']
            df.loc[i, 'cube_name'] = info['cube_name']
            df.loc[i, 'scene_dir'] = info['scene_dir']
        
        return df

class DepthDataset(Dataset):
    def __init__(self, features, analytical_depths, targets):
        self.features = torch.FloatTensor(features)
        self.analytical_depths = torch.FloatTensor(analytical_depths)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Return three items now
        return self.features[idx], self.analytical_depths[idx], self.targets[idx]

class MetadataDepthEstimator(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim//2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim//4, 1)
        )
    
    def forward(self, x):
        return self.network(x).squeeze()

class ImprovedDepthEstimator(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64):
        super().__init__()
        # This network learns to predict a *correction* value
        self.correction_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.3),
            
            # The final output is a single value: the correction
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x_scaled, analytical_pred):
        """
        x_scaled: The full, standardized feature vector.
        analytical_pred: The original, unscaled analytical depth prediction.
        """
        # The network learns a correction based on all scaled features
        correction = self.correction_network(x_scaled).squeeze()
        
        # Add the learned correction to the original analytical prediction
        return analytical_pred + correction

def train_model(features, targets, test_size=0.2, epochs=200, lr=0.001):
    """Train the depth estimation model"""

    # 1. Separate the analytical_depth feature (at index 8) BEFORE splitting
    analytical_depth_feature = features[:, 8]
    
    # Split all data
    X_train, X_test, y_train, y_test, analytical_train, analytical_test = train_test_split(
        features, targets, analytical_depth_feature, test_size=test_size, random_state=42
    )
    
    # 2. Normalize features (as before)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Create datasets with the new structure
    train_dataset = DepthDataset(X_train_scaled, analytical_train, y_train)
    test_dataset = DepthDataset(X_test_scaled, analytical_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 4. Initialize your new model
    model = ImprovedDepthEstimator(input_dim=X_train.shape[1])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    train_losses = []
    test_losses = []
    
    print("Starting training...")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_features, batch_analytical, batch_targets in train_loader:
            optimizer.zero_grad()
            # Pass both sets of features to the model
            predictions = model(batch_features, batch_analytical)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_features, batch_analytical, batch_targets in test_loader:
                predictions = model(batch_features, batch_analytical)
                loss = criterion(predictions, batch_targets)
                test_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        test_losses.append(test_loss / len(test_loader))
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_losses[-1]:.4f}, Test Loss = {test_losses[-1]:.4f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        # Pass both the scaled features and the unscaled analytical predictions
        train_preds = model(torch.FloatTensor(X_train_scaled), torch.FloatTensor(analytical_train)).numpy()
        test_preds = model(torch.FloatTensor(X_test_scaled), torch.FloatTensor(analytical_test)).numpy()
    
    train_mae = mean_absolute_error(y_train, train_preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    
    print(f"\nFinal Results:")
    print(f"Train MAE: {train_mae:.4f}, Train RMSE: {train_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}")
    
    return model, scaler, train_losses, test_losses, {
        'X_test': X_test_scaled,
        'y_test': y_test,
        'test_preds': test_preds,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse
    }

def plot_results(train_losses, test_losses, results):
    """Plot training curves and predictions"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training curves
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training Curves')
    ax1.legend()
    ax1.grid(True)
    
    # Predictions vs actual
    ax2.scatter(results['y_test'], results['test_preds'], alpha=0.6)
    ax2.plot([results['y_test'].min(), results['y_test'].max()], 
             [results['y_test'].min(), results['y_test'].max()], 'r--', lw=2)
    ax2.set_xlabel('True Distance')
    ax2.set_ylabel('Predicted Distance')
    ax2.set_title(f'Predictions vs True (MAE: {results["test_mae"]:.3f})')
    ax2.grid(True)
    
    # Residuals
    residuals = results['test_preds'] - results['y_test']
    ax3.scatter(results['y_test'], residuals, alpha=0.6)
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_xlabel('True Distance')
    ax3.set_ylabel('Residuals (Pred - True)')
    ax3.set_title('Residual Plot')
    ax3.grid(True)
    
    # Error distribution
    ax4.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Residuals')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Error Distribution (RMSE: {results["test_rmse"]:.3f})')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Build dataset
    builder = DepthDatasetBuilder("outputbb")  # Adjust path as needed
    df = builder.create_dataframe()

    # Save the dataframe to a CSV file to allow for external analysis
    csv_filename = "depth_featuresbb_dataset.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nDataset saved to '{csv_filename}'\n")
    
    print("Dataset Overview:")
    print(df.describe())
    print(f"\nFeature names: {builder.get_feature_names()}")
    
    # Extract features and targets
    feature_cols = builder.get_feature_names()
    X = df[feature_cols].values
    y = df['target_distance'].values
    
    print(f"\nTraining on {len(X)} samples with {X.shape[1]} features")
    
    # Train model
    model, scaler, train_losses, test_losses, results = train_model(X, y)
    
    # Plot results
    plot_results(train_losses, test_losses, results)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_names': builder.get_feature_names()
    }, 'depth_model.pth')
    
    print("\nModel saved as 'depth_model.pth'")