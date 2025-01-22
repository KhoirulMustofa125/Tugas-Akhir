import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Contoh: Membuat kumpulan data tiruan
x = np.random.rand(100, 5)  # 100 sampel, 5 fitur
y = np.random.randint(0, 2, size=(100,))  # 100 label, klasifikasi biner

# Split data untuk pelatihan dan pengujian
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=3)

# Inisialisasi model KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Latih model menggunakan data pelatihan
knn.fit(x_train, y_train)

# Prediksi menggunakan data uji
y_pred = knn.predict(x_test)

# Hitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi KNN: {accuracy * 100:.2f}%')