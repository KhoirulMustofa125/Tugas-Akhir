import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Membaca dataset citra
# x adalah array fitur dan y adalah array label (kelas)
x = []  # fitur dari citra
y = []  # fitur dari label

# Contoh dataset (ganti ini dengan logika pemuatan data Anda)
x = np.array([])  # array kosong
y = np.array([])  # array kosong

# Periksa apakah dataset kosong
if x.size == 0 or y.size == 0:
    print("Dataset is empty. Please load valid data.")
else:
    # Split data untuk pelatihan dan pengujian
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    print("Train-test split successful!")

# Inisialisasi model SVM
clf = svm.SVC(kernel='linear')

# Latih model menggunakan data pelatihan
# Step 1: Menghasilkan atau memuat data
x, y = make_classification(n_samples=100, n_features=5, random_state=42)

# Step 2: Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Step 3: Cek data shapes
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")

# Step 4: Inisialisasi model
clf = RandomForestClassifier()

# Step 5: Fit model
clf.fit(x_train, y_train)

print("Model training completed successfully!")


# Prediksi menggunakan data uji
y_pred = clf.predict(x_test)

# Hitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi: {accuracy * 100:.2f}%')