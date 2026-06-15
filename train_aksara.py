import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# 1. Tentukan lokasi folder dataset
# Karena folder dataset ada di dalam folder proyek yang sama, kita cukup panggil namanya
DATASET_PATH = "dataset_gabung_fix"

# 2. Load dataset secara otomatis dari folder menggunakan Keras utils
# Kita ubah gambar menjadi Grayscale dan ukuran 64x64 agar training-nya RINGAN di laptop
print("Memuat dataset...")
train_ds = keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,     # 20% data untuk uji/validasi
    subset="training",
    seed=123,
    image_size=(64, 64),      # Mengubah ukuran gambar menjadi 64x64 piksel
    color_mode="grayscale",   # Mengubah menjadi hitam-putih agar komputasi ringan
    batch_size=32
)

val_ds = keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(64, 64),
    color_mode="grayscale",
    batch_size=32
)

# Mengambil nama-nama kelas aksara berdasarkan nama sub-folder
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Daftar Aksara yang dideteksi ({num_classes} kelas): {class_names}")

# 3. Normalisasi nilai piksel gambar dari [0, 255] menjadi [0, 1]
normalization_layer = keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# 4. Membangun Arsitektur Model CNN (Sederhana & Ringan)
model = keras.Sequential([
    # Layer Konvolusi 1
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    
    # Layer Konvolusi 2
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    
    # Flatten & Fully Connected Layer
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3), # Mencegah overfitting
    keras.layers.Dense(num_classes, activation='softmax') # Output dinamis sesuai jumlah aksara
])

# 5. Kompilasi Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 6. Proses Training Model
print("\nMemulai proses training model...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5 # Kita coba 5 kali perulangan dulu agar cepat selesai
)

# 7. Simpan Model Hasil Training (.h5)
# File ini yang nanti akan kita masukkan ke aplikasi Flask web Anda!
model.save("model_aksara_sunda.h5")
print("\nTraining selesai! Model berhasil disimpan dengan nama 'model_aksara_sunda.h5'")