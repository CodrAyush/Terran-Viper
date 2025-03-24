# Real-time 3D Voxel Mapping with MiDaS Depth Estimation

This project implements real-time 3D voxel mapping using a webcam and the MiDaS (Monocular Depth Estimation) model. It captures video from a webcam, estimates depth using MiDaS, converts the depth map to a 3D point cloud, and visualizes it as a voxel grid in real-time.

## Features

- Real-time depth estimation using MiDaS model
- Conversion of depth maps to 3D point clouds
- Voxel grid visualization using Open3D
- Live webcam feed processing
- Interactive visualization

## Requirements

- Python 3.7+
- PyTorch
- Open3D
- OpenCV (cv2)
- NumPy

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required packages:
```bash
pip install torch open3d opencv-python numpy
```

## Usage

1. Run the script:
```bash
python modelxvoxel.py
```

2. The program will:
   - Open your webcam
   - Show the depth map visualization
   - Display the 3D voxel grid visualization
   - Press 'q' to quit the application

## Camera Calibration

The current implementation uses default camera parameters:
- Principal point (cx, cy): (320, 240)
- Focal length (fx, fy): (500, 500)

To use different camera parameters, modify these values in the code:
```python
cx, cy = 320, 240  # Principal point
fx, fy = 500, 500  # Focal length
```

## Notes

- The quality of the 3D reconstruction depends on the accuracy of the depth estimation and camera calibration
- The voxel size can be adjusted by modifying the `voxel_size` parameter in the `visualize_voxel_map` function
- The visualization window can be interacted with using the mouse (rotate, zoom, pan)

## License

[Add your license information here] 