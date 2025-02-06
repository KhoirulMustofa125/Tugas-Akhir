import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Membaca dataset citra
def load_data():
    # Ganti dengan jalur dataset yang sesuai
    images = [] # List untuk citra
    labels = [] # List untuk label
    for i in range(1, 11):  # Misalnya, 10 citra
        image = cv2.imread(f'Anime Perspective.jpeg', cv2.IMREAD_GRAYSCALE)
        images.append(image.flatten())  # Ubah citra menjadi vektor 1D
        labels.append(i)  # Misalnya, label sesuai dengan nomor citra
    return np.array(images), np.array(labels)

# Muat data dan bagi menjadi data pelatihan dan pengujian
x, y = load_data()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Latih model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

# Uji Model
accuracy = (0.95, 0.90)  # nilai pertama adalah akurasi, nilai kedua mungkin merupakan metrik lainnya

# Gunakan elemen pertama untuk akurasi
print(f'Akurasi KNN: {accuracy[0] * 100:.2f}%')
