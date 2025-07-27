# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2

# # --- CONFIGURATION ---
# # Specify the category ID you want to isolate and view.
# # For example, in pipeline.py, the cube is 1 and the ground is 2.
# TARGET_CATEGORY_ID = 1
# # ---------------------

# # Load the .hdf5 file
# try:
#     file = h5py.File("output/scene_0000.hdf5", "r")
#     print("Successfully loaded HDF5 file")
# except FileNotFoundError:
#     print("Error: output/0.hdf5 not found. Make sure the pipeline ran successfully.")
#     exit()

# # Load render outputs
# rgb = file["colors"][()]
# depth = file["depth"][()]
# seg = file["category_id_segmaps"][()]

# # --- Data Processing ---
# # Check if we have multiple frames and select the first one
# if len(rgb.shape) == 4:
#     rgb = rgb[0]
# if len(depth.shape) == 3:
#     depth = depth[0]
# if len(seg.shape) == 3:
#     seg = seg[0]

# # Normalize RGB for display
# rgb_vis = rgb.copy()
# if rgb.dtype in [np.float32, np.float64]:
#     rgb_vis = np.clip(rgb_vis, 0, 1)
# elif rgb.max() > 1:
#     rgb_vis = rgb_vis / 255.0

# # Normalize depth for display
# depth_vis = depth.copy()
# # Invert depth values for better visualization (closer objects are brighter)
# if depth.max() > 0:
#     depth_vis[depth_vis > 1000] = depth.min() # Set far away points to min depth
#     depth_vis = cv2.normalize(depth_vis, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
#     depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)


# # --- Create the Specific Mask ---
# # Create a binary mask where pixels matching the target ID are 1, and all others are 0.
# print(f"\nOriginal unique segmentation IDs found: {np.unique(seg)}")
# print(f"Isolating mask for Category ID: {TARGET_CATEGORY_ID}")
# mask = (seg == TARGET_CATEGORY_ID).astype(np.uint8)
# pixel_count = np.sum(mask)
# print(f"Total pixels in mask: {pixel_count}")


# # --- Visualization ---
# fig, axs = plt.subplots(1, 3, figsize=(18, 6))
# plt.suptitle(f"Render Output Analysis (Masking ID: {TARGET_CATEGORY_ID})", fontsize=16)

# # RGB Image
# axs[0].imshow(rgb_vis)
# axs[0].set_title(f"RGB Image\nShape: {rgb.shape}")
# axs[0].axis('off')

# # Depth Map
# axs[1].imshow(depth_vis, cmap='plasma')
# axs[1].set_title(f"Depth Map (Normalized)\nNon-zero pixels: {np.count_nonzero(depth)}")
# axs[1].axis('off')

# # Isolated Segmentation Mask
# if pixel_count > 0:
#     axs[2].imshow(mask, cmap='gray')
# else:
#     # Show a black image if the ID was not found
#     axs[2].imshow(np.zeros_like(mask), cmap='gray')
#     print(f"\nWarning: Category ID {TARGET_CATEGORY_ID} not found in the segmentation map.")

# axs[2].set_title(f"Mask for ID: {TARGET_CATEGORY_ID}\nPixel Count: {pixel_count}")
# axs[2].axis('off')

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

# # Close the file
# file.close()


import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2

# --- CONFIGURATION ---
# Specify the category ID you want to isolate and view.
TARGET_CATEGORY_ID = 1
# Specify the path to the HDF5 file you want to render.
HDF5_FILE_PATH = "outputbb/scene_0000/0.hdf5" 
# ---------------------

# Load the .hdf5 file
try:
    file = h5py.File(HDF5_FILE_PATH, "r")
    print(f"Successfully loaded HDF5 file: {HDF5_FILE_PATH}")
except FileNotFoundError:
    print(f"Error: {HDF5_FILE_PATH} not found. Make sure the pipeline ran successfully.")
    exit()

# Load render outputs
rgb = file["colors"][()]
depth = file["depth"][()]
seg = file["category_id_segmaps"][()]

# --- Data Processing ---
# Check if we have multiple frames and select the first one
if len(rgb.shape) == 4:
    rgb = rgb[0]
if len(depth.shape) == 3:
    depth = depth[0]
if len(seg.shape) == 3:
    seg = seg[0]

# Normalize RGB for display
rgb_vis = rgb.copy()
if rgb.dtype in [np.float32, np.float64]:
    rgb_vis = np.clip(rgb_vis, 0, 1)
elif rgb.max() > 1:
    rgb_vis = rgb_vis / 255.0

# Normalize depth for display
depth_vis = depth.copy()
if depth.max() > 0:
    depth_vis[depth_vis > 1000] = depth.min() 
    depth_vis = cv2.normalize(depth_vis, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)


# --- Create the Specific Mask ---
print(f"\nOriginal unique segmentation IDs found: {np.unique(seg)}")
print(f"Isolating mask for Category ID: {TARGET_CATEGORY_ID}")
mask = (seg == TARGET_CATEGORY_ID).astype(np.uint8)
pixel_count = np.sum(mask)
print(f"Total pixels in mask: {pixel_count}")


# --- Visualization ---
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
plt.suptitle(f"Render Output Analysis (Masking ID: {TARGET_CATEGORY_ID})", fontsize=16)

# RGB Image
axs[0].imshow(rgb_vis)
axs[0].set_title(f"RGB Image\nShape: {rgb.shape}")
axs[0].axis('off')

# Depth Map
axs[1].imshow(depth_vis, cmap='plasma')
axs[1].set_title(f"Depth Map (Normalized)\nNon-zero pixels: {np.count_nonzero(depth)}")
axs[1].axis('off')

# Isolated Segmentation Mask
if pixel_count > 0:
    axs[2].imshow(mask, cmap='gray')
else:
    axs[2].imshow(np.zeros_like(mask), cmap='gray')
    print(f"\nWarning: Category ID {TARGET_CATEGORY_ID} not found in the segmentation map.")

axs[2].set_title(f"Mask for ID: {TARGET_CATEGORY_ID}\nPixel Count: {pixel_count}")
axs[2].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Close the file
file.close()
