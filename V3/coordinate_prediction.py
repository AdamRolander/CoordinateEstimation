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

            for cube_name, cube_data in metadata['cubes'].items():
                features = self.extract_cube_features(cube_data, metadata)
                target = self.calculate_true_distance(cube_data, metadata)

                self.features.append(features)
                self.targets.append(target)
                self.scene_info.append({
                    'scene_id': metadata['scene_id'],
                    'cube_name': cube_name,
                    'scene_dir': scene_dir,
                    # Store data needed for reconstruction
                    'bbox_center': cube_data.get('bbox_center', [0, 0]),
                    'intrinsics': metadata['camera']['intrinsic_matrix'],
                    'pose_matrix': metadata['camera']['pose_matrix'],
                    'true_world_location': cube_data['location']
                })

        print(f"Extracted {len(self.features)} samples from {len(scene_dirs)} scenes")
        return np.array(self.features), np.array(self.targets)

    def extract_cube_features(self, cube_data, metadata):
        """Extract features for a single cube"""
        mask_pixel_count = cube_data['mask_pixel_count']
        mask_coverage_percent = cube_data['mask_coverage_percent']
        object_true_size = cube_data['actual_size']
        img_width, _ = metadata['camera']['resolution']
        focal_length = metadata['camera']['focal_length_mm']
        object_scale = cube_data['scale']

        # Get new bounding box data from metadata
        bbox_width = cube_data['bbox_width']
        bbox_height = cube_data['bbox_height']
        pixel_density = cube_data['pixel_density']
        bbox_aspect_ratio = cube_data['bbox_aspect_ratio']
        
        # New, improved analytical depth calculation
        if bbox_width > 0 and bbox_height > 0:
            analytical_depth = (object_true_size * focal_length * img_width) / (np.sqrt(bbox_width * bbox_height) * 36)
        else:
            analytical_depth = 10.0 # fallback

        # To calculate norm_pixel_x/y, we need to project the 3D point to 2D
        img_width, img_height = metadata['camera']['resolution']
        obj_x, obj_y, obj_z = cube_data['location']
        cam_matrix = np.array(metadata['camera']['pose_matrix'])
        intrinsics = np.array(metadata['camera']['intrinsic_matrix'])
        world_pos = np.array([obj_x, obj_y, obj_z, 1.0])
        cam_pos = np.linalg.inv(cam_matrix) @ world_pos

        if cam_pos[2] > 0:
            pixel_x = (intrinsics[0][0] * cam_pos[0]) / cam_pos[2] + intrinsics[0][2]
            pixel_y = (intrinsics[1][1] * cam_pos[1]) / cam_pos[2] + intrinsics[1][2]
        else:
            pixel_x, pixel_y = img_width / 2, img_height / 2

        norm_pixel_x = pixel_x / img_width
        norm_pixel_y = pixel_y / img_height
        center_x, center_y = img_width / 2, img_height / 2
        distance_from_center = np.sqrt((pixel_x - center_x)**2 + (pixel_y - center_y)**2)
        norm_distance_from_center = distance_from_center / np.sqrt(center_x**2 + center_y**2)

        features = [
            mask_pixel_count, mask_coverage_percent, object_true_size,
            bbox_width, bbox_height, bbox_aspect_ratio, pixel_density,
            object_scale, analytical_depth, norm_pixel_x,
            norm_pixel_y, norm_distance_from_center
        ]
        return features

    def calculate_true_distance(self, cube_data, metadata):
        """Calculate true distance from camera to object center"""
        obj_pos = np.array(cube_data['location'])
        cam_pos_data = metadata['camera']['location']
        cam_pos = np.array([cam_pos_data[0], cam_pos_data[1], cam_pos_data[2]])
        return np.linalg.norm(obj_pos - cam_pos)

    def get_feature_names(self):
        return [
            'mask_pixel_count', 'mask_coverage_percent', 'object_true_size',
            'bbox_width', 'bbox_height', 'bbox_aspect_ratio', 'pixel_density',
            'object_scale', 'analytical_depth', 'norm_pixel_x',
            'norm_pixel_y', 'norm_distance_from_center'
        ]

    def create_dataframe(self):
        """Create a pandas DataFrame with all features and targets"""
        features, targets = self.extract_features_from_metadata()
        feature_names = self.get_feature_names()
        df = pd.DataFrame(features, columns=feature_names)
        df['target_distance'] = targets
        
        scene_df = pd.DataFrame(self.scene_info)
        df = pd.concat([df, scene_df], axis=1)
        return df

class DepthDataset(Dataset):
    def __init__(self, features, analytical_depths, targets):
        self.features = torch.FloatTensor(features)
        self.analytical_depths = torch.FloatTensor(analytical_depths)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.analytical_depths[idx], self.targets[idx]

class ImprovedDepthEstimator(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64):
        super().__init__()
        self.correction_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.BatchNorm1d(hidden_dim), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2), nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x_scaled, analytical_pred):
        correction = self.correction_network(x_scaled).squeeze(-1)
        return analytical_pred + correction

def train_model(features, targets, test_size=0.2, epochs=200, lr=0.001):
    analytical_depth_feature = features[:, 8]
    X_train, X_test, y_train, y_test, analytical_train, analytical_test = train_test_split(
        features, targets, analytical_depth_feature, test_size=test_size, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    train_dataset = DepthDataset(X_train_scaled, analytical_train, y_train)
    test_dataset = DepthDataset(X_test_scaled, analytical_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = ImprovedDepthEstimator(input_dim=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        for batch_features, batch_analytical, batch_targets in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_features, batch_analytical)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
        
    print("Training finished.")
    return model, scaler

# --- NEW FUNCTION START ---

def reconstruct_3d_points(df, model, scaler, feature_names):
    """
    Uses the trained model to predict depth and reconstructs 3D points in camera space.
    """
    print("\n--- Starting 3D Reconstruction ---")
    
    # Prepare features for prediction
    X = df[feature_names].values
    analytical_depths = X[:, 8]
    X_scaled = scaler.transform(X)
    
    # Get depth predictions from the model
    model.eval()
    with torch.no_grad():
        predicted_depths = model(torch.FloatTensor(X_scaled), torch.FloatTensor(analytical_depths)).numpy()
    
    df['predicted_depth'] = predicted_depths
    
    # Store results
    predicted_points_cam = []
    true_points_cam = []
    
    for idx, row in df.iterrows():
        # Unproject predicted point
        depth = row['predicted_depth']
        px, py = row['bbox_center']
        intrinsics = row['intrinsics']
        fx, fy, cx, cy = intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2]
        
        x_cam = (px - cx) * depth / fx
        y_cam = -((py - cy) * depth / fy)

        # The predicted point uses the corrected y_cam and a negative depth
        predicted_points_cam.append([x_cam, y_cam, -depth])
        
        # Transform true world point to camera space for comparison
        pose_matrix = np.array(row['pose_matrix'])
        true_world_pos = np.append(row['true_world_location'], 1)
        true_cam_pos = np.linalg.inv(pose_matrix) @ true_world_pos
        true_points_cam.append(true_cam_pos[:3])

    df[['pred_x_cam', 'pred_y_cam', 'pred_z_cam']] = predicted_points_cam
    df[['true_x_cam', 'true_y_cam', 'true_z_cam']] = true_points_cam
    
    # Calculate final reconstruction error
    pred_points = np.array(predicted_points_cam)
    true_points = np.array(true_points_cam)
    euclidean_errors = np.linalg.norm(pred_points - true_points, axis=1)
    mean_error = np.mean(euclidean_errors)
    
    print(f"Mean 3D Reconstruction Error (Euclidean Distance): {mean_error:.4f}")
    
    return df

# --- NEW FUNCTION END ---

if __name__ == "__main__":
    # Note: Ensure your data is in a subfolder named "outputbb" or change the path
    builder = DepthDatasetBuilder("outputbb") 
    df = builder.create_dataframe()

    feature_cols = builder.get_feature_names()
    X = df[feature_cols].values
    y = df['target_distance'].values
    
    print(f"\nTraining on {len(X)} samples with {X.shape[1]} features")
    
    model, scaler = train_model(X, y)
    
    # Save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_names': builder.get_feature_names()
    }, 'depth_model_final.pth')
    print("\nModel saved as 'depth_model_final.pth'")
    
    # --- ADDED CODE TO RUN RECONSTRUCTION ---
    reconstruction_df = reconstruct_3d_points(df, model, scaler, feature_cols)
    
    print("\nReconstruction Results Overview:")
    print(reconstruction_df[[
        'predicted_depth', 'target_distance', 
        'pred_x_cam', 'true_x_cam',
        'pred_y_cam', 'true_y_cam',
        'pred_z_cam', 'true_z_cam'
    ]].head())