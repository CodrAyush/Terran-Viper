import cv2
import torch
import time
import numpy as np

# Load MiDaS model
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = (
    midas_transforms.dpt_transform if model_type in ["DPT_Large", "DPT_Hybrid"]
    else midas_transforms.small_transform
)

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = 30  # Fixed FPS

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use XVID codec
out = cv2.VideoWriter('depth_output.avi', fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    start = time.time()
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

    if prediction.numel() == 0 or torch.isnan(prediction).any():
        print("Invalid prediction! Skipping frame.")
        continue

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False
    ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_map[np.isnan(depth_map)] = 0
    depth_map[np.isinf(depth_map)] = 0

    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

    end = time.time()
    elapsed_time = end - start
    frame_time = 1.0 / fps  # Desired time per frame

    # Ensure frame size matches video
    resized_frame = cv2.resize(depth_map_colored, (frame_width, frame_height))

    # Write frame
    out.write(resized_frame)
    print("Frame written successfully")

    # Sleep if frame processing was too fast
    if elapsed_time < frame_time:
        time.sleep(frame_time - elapsed_time)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Video saved successfully!")
