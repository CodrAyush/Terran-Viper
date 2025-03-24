import cv2
import torch
import time
import numpy as np

# Load a MiDaS model for depth estimation
model_type = "DPT_Large"  # Higher accuracy, slower inference speed
midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = (
    midas_transforms.dpt_transform if model_type in ["DPT_Large", "DPT_Hybrid"]
    else midas_transforms.small_transform
)

# Open video capture
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break
    
    start = time.time()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply input transforms
    input_batch = transform(img).to(device)
    
    # Predict depth
    with torch.no_grad():
        prediction = midas(input_batch)
    
    # Debugging: Check prediction output
    if prediction.numel() == 0 or torch.isnan(prediction).any():
        print("Invalid prediction! Skipping frame.")
        continue
    
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False
    ).squeeze()
    
    # Convert to numpy and handle invalid values
    depth_map = prediction.cpu().numpy()
    depth_map[np.isnan(depth_map)] = 0
    depth_map[np.isinf(depth_map)] = 0
    
    # Normalize depth map
    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    
    end = time.time()
    fps = 1 / (end - start)
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Prepare the depth map for display
    depth_map = (depth_map * 255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
    
    # Display FPS on the original image
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    
    # Show original image and depth map
    cv2.imshow('Image', img)
    cv2.imshow('Depth Map', depth_map)
    
    if cv2.waitKey(5) & 0xFF == 27:  # Exit on 'ESC'
        break

cap.release()
cv2.destroyAllWindows()
 