import numpy as np
import open3d as o3d
import cv2
import torch
import time


# Camera Intrinsic Parameters (Adjust according to your camera)
cx, cy = 320, 240  # Principal point (example values)
fx, fy = 500, 500  # Focal length (example values)


# Load MiDaS model
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform


# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()


# Function to convert depth map to 3D point cloud
def depth_to_point_cloud(depth_map, cx, cy, fx, fy):
    h, w = depth_map.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth_map
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    return points


# Function to create voxel grid
def create_voxel_grid(point_cloud, voxel_size=0.1):
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        point_cloud, voxel_size=voxel_size
    )
    return voxel_grid


# Visualization function
def visualize_voxel_map(points, voxel_size=0.1):
    # Convert points to Open3D PointCloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)


    # Create Voxel Grid
    voxel_grid = create_voxel_grid(point_cloud, voxel_size)


    # Visualize
    o3d.visualization.draw_geometries([voxel_grid])


# Process frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the frame.")
        break


    # Convert frame to RGB
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = midas_transforms(input_image).to(device)


    # Perform inference
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=input_image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()


    depth_map = prediction.cpu().numpy()


    # Normalize depth map for visualization
    depth_map_vis = cv2.normalize(
        depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    depth_map_vis_colored = cv2.applyColorMap(depth_map_vis, cv2.COLORMAP_MAGMA)


    # Show the depth map
    cv2.imshow("Depth Map", depth_map_vis_colored)


    # Convert depth map to 3D point cloud
    points = depth_to_point_cloud(depth_map, cx, cy, fx, fy)


    # Visualize the voxel map
    visualize_voxel_map(points, voxel_size=0.1)
     


    # Break on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the video objects and close all windows
cap.release()
cv2.destroyAllWindows()
