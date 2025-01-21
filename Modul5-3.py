import cv2

# Membaca gambar dalam grayscale
image = cv2.imread('Gravity_Among us.jpeg')

# Menerapkan deteksi tepi canny
edges = cv2.Canny(image, 100, 200)

# Menampilkan hasil
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows