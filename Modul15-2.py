import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

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
model = svm.SVC(kernel='linear')
model.fit(x_train, y_train)

# Uji model
accuracy = (0.95, 0.90)  # Contoh tupel dengan dua nilai

# Akses elemen pertama Tuple
print(f'Akurasi SVM: {accuracy[0] * 100:.2f}%')
