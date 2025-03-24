import cv2
import numpy as np
import glob  # For easily reading image files

# 1. Checkerboard Settings (Adjust these!)
CHECKERBOARD = (7, 9)  # Number of INNER corners (rows, columns)
SQUARE_SIZE = 2.5  # Size of each square in centimeters (or your units)

# 2. Image Directory
image_dir = "path/to/your/images"  # Replace with the path to your image folder
images = glob.glob(image_dir + "/*.jpg")  # Or *.png, etc. - adjust file extension

# 3. Calibration Process
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE # Scale by square size

gray = None # Keep track of image shape for calibration

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners (optional, for visualization)
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(100)  # Adjust delay as needed

cv2.destroyAllWindows()

# 4. Camera Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 5. Save Calibration Data (Important!)
calibration_data = {
    "camera_matrix": mtx,
    "dist_coeffs": dist,
    # You can also save rvecs and tvecs if needed
}
np.save("calibration_data.npy", calibration_data)  # Save as a NumPy file


# 6. Print or Display Results
print("Camera Matrix (Intrinsics):\n", mtx)
print("Distortion Coefficients:\n", dist)

# 7. (Optional) Undistort an Image
img = cv2.imread("test_image.jpg") # Replace with a test image
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# Undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibrated_result.png', dst)