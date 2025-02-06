import cv2
import numpy as np

# Baca citra dalam grayscale
image = cv2.imread('Super Mario.jpeg', cv2.IMREAD_GRAYSCALE)

# Terapkan Harris Corner Detector
gray = np.float32(image)
corners = cv2.cornerHarris(gray, 2, 3, 0.04)

# Dilasi untuk menonjolkan sudut yang terdeteksi
corners = cv2.dilate(corners, None)

# Tingkatkan sudut yang terdeteksi
mask = corners > 0.01 * corners.max()

# Tampilkan hasil deteksi sudut
cv2.imshow('Harris Corners Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()