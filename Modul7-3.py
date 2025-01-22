from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Muat kumpulan data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Data praproses
x_train = x_train.astype('float32') / 255.0  # Normalisasikan ke [0, 1]
x_test = x_test.astype('float32') / 255.0

# Bentuk ulang jika diperlukan (misalnya, untuk Konv2D)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Label enkode one-hot
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


# Inisialisasi model CNN
model = Sequential()

# Tambahkan convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Tambahkan convolutional layer lainnya
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening
model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Latih model
model.fit(x_train, to_categorical(y_train), epochs=10, batch_size=32)

# Evaluasi model
loss, accuracy = model.evaluate(x_test, to_categorical(y_test))
print(f'Akurasi CNN: {accuracy * 100:.2f}%')