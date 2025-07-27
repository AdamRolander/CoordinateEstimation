import blenderproc as bproc
import os
import shutil
import numpy as np
import sys
import json
import mathutils
import random

# Get command line arguments
if len(sys.argv) >= 3:
    scene_id = int(sys.argv[1])
    output_dir = sys.argv[2]
else:
    scene_id = 0
    output_dir = "output/"

print(f"Generating scene {scene_id:04d} to {output_dir}")

# Initialize BlenderProc
bproc.init()

# Clean output directory for this specific scene
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    print(f"Removed existing output directory: {output_dir}")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Set random seed for scene variation
random.seed(scene_id)
np.random.seed(scene_id)

# Setup camera with specified parameters
image_width, image_height = 640, 480
bproc.camera.set_resolution(image_width, image_height)
focal_length_mm = 35
bproc.camera.set_intrinsics_from_blender_params(lens=focal_length_mm, lens_unit="MILLIMETERS")

# Fixed camera pose
cam_location = [6, -6, 3]
target_location = [0, 0, 3.5]
direction = mathutils.Vector(target_location) - mathutils.Vector(cam_location)
rot_quat = direction.to_track_quat('-Z', 'Y')
cam_pose_matrix = bproc.math.build_transformation_mat(cam_location, rot_quat.to_euler())
bproc.camera.add_camera_pose(cam_pose_matrix)

# --- Manual Matrix Calculations for Full Compatibility ---
sensor_width_mm = 36
fx = (focal_length_mm * image_width) / sensor_width_mm
fy = fx
cx = image_width / 2
cy = image_height / 2
cam_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

view_matrix = np.linalg.inv(cam_pose_matrix)

# Setup lighting
sun_light = bproc.types.Light()
sun_light.set_type("SUN")
sun_light.set_location([2, 2, 3])
sun_light.set_energy(5)
sun_light.set_rotation_euler([0.3, 0.3, 0])

point_light = bproc.types.Light()
point_light.set_type("POINT")
point_light.set_location([-1, -1, 2])
point_light.set_energy(100)

# Add ground plane
ground = bproc.object.create_primitive("PLANE", scale=[10, 10, 1])
ground.set_location([0, 0, -1])
ground.set_cp("category_id", 3)  # Set category_id for ground plane
ground_material = bproc.material.create("ground_material")
ground_material.set_principled_shader_value("Base Color", [0.7, 0.7, 0.7, 1.0])
ground.add_material(ground_material)

# Create cubes with scene-specific variation
cubes_data = []

for cube_idx in range(2):
    cube_scale = 0.5 * random.uniform(0.8, 1.2)
    actual_size = cube_scale * 2
    
    x, y, z = random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(2, 5)
    
    cube = bproc.object.create_primitive("CUBE", scale=[cube_scale, cube_scale, cube_scale])
    cube.set_location([x, y, z])
    category_id = cube_idx + 1  # Use 1 and 2 as IDs
    cube.set_cp("category_id", category_id)
    
    color = random.choice([
        [0.8, 0.2, 0.2, 1.0], [0.2, 0.8, 0.2, 1.0], [0.2, 0.2, 0.8, 1.0],
        [0.8, 0.8, 0.2, 1.0], [0.8, 0.2, 0.8, 1.0], [0.2, 0.8, 0.8, 1.0],
    ])
    material = bproc.material.create(f"cube_{cube_idx}_mat_{scene_id}")
    material.set_principled_shader_value("Base Color", color)
    material.set_principled_shader_value("Roughness", random.uniform(0.3, 0.7))
    cube.add_material(material)
    
    # Store cube data for metadata
    cubes_data.append({
        "location": [x, y, z],
        "scale": cube_scale,
        "actual_size": actual_size,
        "category_id": category_id,
        "color": color[:3],  # RGB without alpha
        "roughness": material.get_principled_shader_value("Roughness")
    })

# Enable outputs
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(map_by=["category_id"])

# Set render engine
bproc.renderer.set_render_devices(use_only_cpu=False)

# Render
print("Starting render...")
data = bproc.renderer.render()
print("Render completed!")

# Debug: Check segmentation
cube_mask_data = {"cube_1": {}, "cube_2": {}}
if "category_id_segmaps" in data:
    seg_data = data["category_id_segmaps"][0]
    unique_ids = np.unique(seg_data)
    print(f"Segmentation contains IDs: {unique_ids}")
    
    # Calculate mask data for each cube
    for i, cube_idx in enumerate([1, 2]):
        mask = (seg_data == cube_idx).astype(np.uint8)
        pixel_count = np.sum(mask)
        cube_key = f"cube_{i+1}"
        cube_mask_data[cube_key] = {
            "mask_pixel_count": int(pixel_count),
            "mask_coverage_percent": float(pixel_count / (image_width * image_height) * 100)
        }

scene_name = f"scene_{scene_id:04d}"

# Create comprehensive metadata
metadata = {
    "scene_id": scene_id,
    "scene_name": scene_name,
    "camera": {
        "location": cam_location,
        "resolution": [image_width, image_height],
        "intrinsic_matrix": cam_intrinsics.tolist(),
        "pose_matrix": cam_pose_matrix.tolist(),
        "focal_length_mm": focal_length_mm,
        "sensor_width_mm": sensor_width_mm,
        "target_location": target_location
    },
    "files": {
        "hdf5_file": "0.hdf5",
        "rgb_image": f"{scene_name}.png",
        "depth_map": f"{scene_name}_depth.npy",
        "semantic_segmentation": f"{scene_name}_semantic.npy",
        "instance_segmentation": f"{scene_name}_instance.npy"
    },
    "cubes": {
        "cube_1": {**cubes_data[0], **cube_mask_data["cube_1"]},
        "cube_2": {**cubes_data[1], **cube_mask_data["cube_2"]}
    },
    "segmentation_info": {
        "method": "blenderproc_builtin",
        "background_category_id": 0,
        "cube_category_ids": [1, 2],
        "ground_category_id": 3,
        "render_keys": list(data.keys()),
        "unique_segmentation_ids": unique_ids.tolist() if "category_id_segmaps" in data else []
    },
    "lighting": {
        "sun_light": {
            "type": "SUN",
            "location": [2, 2, 3],
            "energy": 5,
            "rotation_euler": [0.3, 0.3, 0]
        },
        "point_light": {
            "type": "POINT", 
            "location": [-1, -1, 2],
            "energy": 100
        }
    },
    "ground": {
        "scale": [10, 10, 1],
        "location": [0, 0, -1],
        "color": [0.7, 0.7, 0.7]
    }
}

# Save metadata
metadata_path = os.path.join(output_dir, "metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(metadata, indent=2, fp=f)
print(f"Metadata saved to {metadata_path}")

# Save
bproc.writer.write_hdf5(output_dir, data)
print(f"Data saved to {output_dir}")

# Clean up
bproc.clean_up()

print(f"Scene {scene_id:04d} generation complete!")