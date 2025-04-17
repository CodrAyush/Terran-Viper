import numpy as np
import open3d as o3d
import cv2
import torch
import time
import pygame
import threading

# Camera Intrinsic Parameters
cx, cy = 320, 240
fx, fy = 500, 500

# MiDaS model setup
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

# Open3D Visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Live Voxel Grid", width=640, height=480)
geom_added = False

# Pygame sound init
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.wav")

# Thread control
beep_running = False
beep_thread = None

def depth_to_point_cloud(depth_map, cx, cy, fx, fy):
    h, w = depth_map.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth_map
    X = (u - cx) * Z / fx
    Y = -(v - cy) * Z / fy
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    return points

def create_voxel_grid(points, voxel_size=0.1):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size)
    return voxel_grid

def beep_loop():
    global beep_running
    while beep_running:
        alert_sound.play()
        time.sleep(alert_sound.get_length())

def start_beep():
    global beep_running, beep_thread
    if not beep_running:
        beep_running = True
        beep_thread = threading.Thread(target=beep_loop)
        beep_thread.start()

def stop_beep():
    global beep_running, beep_thread
    if beep_running:
        beep_running = False
        if beep_thread:
            beep_thread.join()
        beep_thread = None

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the frame.")
        break

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

    # Show live depth
    depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

    # Proximity check
    threshold = 0.5  # Tune based on your environment
    valid_depths = depth_map[depth_map > 0.1]
    if valid_depths.size == 0:
        min_distance = float('inf')
    else:
        min_distance = np.percentile(valid_depths, 2)

    if min_distance < threshold:
        cv2.putText(depth_vis_color, "Object Too Close!", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        start_beep()
    else:
        stop_beep()

    cv2.imshow("Live Depth", depth_vis_color)

    # Voxel grid
    points = depth_to_point_cloud(depth_map, cx, cy, fx, fy)
    voxel_grid = create_voxel_grid(points)

    if not geom_added:
        vis.add_geometry(voxel_grid)
        geom_added = True
    else:
        vis.clear_geometries()
        vis.add_geometry(voxel_grid)

    vis.poll_events()
    vis.update_renderer()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
vis.destroy_window()
stop_beep()
