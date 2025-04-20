import numpy as np
import open3d as o3d
import cv2
import torch

# Camera Intrinsic Parameters
cx, cy = 320, 240
fx, fy = 500, 500

# MiDaS model setup
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()

# Capture a single frame from webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()
if not ret:
    print("Error: Could not read the frame.")
    exit()

# Depth estimation
input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
input_batch = midas_transforms(input_image).to(device)
with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=input_image.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
depth_map = prediction.cpu().numpy()

# Convert depth map to point cloud
def depth_to_point_cloud(depth_map, cx, cy, fx, fy):
    h, w = depth_map.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth_map
    X = (u - cx) * Z / fx
    Y = -(v - cy) * Z / fy
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    return points

points = depth_to_point_cloud(depth_map, cx, cy, fx, fy)

# Create voxel grid from single frame point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
voxel_size = 0.1  # Adjust as needed
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size)

# Visualize
o3d.visualization.draw_geometries([voxel_grid])
