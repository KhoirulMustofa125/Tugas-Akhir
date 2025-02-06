import cv2

# Baca citra grayscale
image = cv2.imread('Toys_Super Mario.jpeg', cv2.IMREAD_GRAYSCALE)

# Deteksi tepi canny
edges = cv2.Canny(image, 100, 200)

# Tampilkan hasil
cv2.imshow('Edges Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()