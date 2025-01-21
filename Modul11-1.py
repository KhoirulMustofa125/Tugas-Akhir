import cv2
import numpy as np

# Baca citra grayscale
image = cv2.imread('Landscape_Mount.jpeg', cv2.IMREAD_GRAYSCALE)

# Deteksi tepi menggunakan Sobel
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5) # Sobel x
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5) # Sobel y

# Tampilkan hasil
cv2.imshow('Sobel X', sobelx)
cv2.imshow('Sobel Y', sobely)
cv2.waitKey(0)
cv2.destroyAllWindows()