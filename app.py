import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow import keras
import tensorflow as tf  
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Konfigurasi folder tempat menyimpan unggahan gambar sementara
UPLOAD_FOLDER = os.path.join('static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Baris os.makedirs dinonaktifkan untuk menghindari FileExistsError [WinError 183] di Windows
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 1. Muat Model CNN yang sudah dilatih sebelumnya
MODEL_PATH = "model_aksara_sunda.h5"
model = keras.models.load_model(MODEL_PATH)

# 2. Definisikan daftar nama kelas aksara (Sesuai urutan folder saat training)
CLASS_NAMES = ['a', 'ae', 'ba', 'ca', 'da', 'e', 'eu', 'fa', 'ga', 'ha', 
               'i', 'ja', 'ka', 'la', 'ma', 'na', 'nga', 'nya', 'o', 'pa', 
               'qa', 'ra', 'sa', 'ta', 'u', 'va', 'wa', 'xa', 'ya', 'za']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # 3. Preprocessing gambar (64x64, Grayscale)
            img = keras.utils.load_img(filepath, target_size=(64, 64), color_mode="grayscale")
            img_array = keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Dimensi menjadi (1, 64, 64, 1)
            img_array = img_array / 255.0                  # Normalisasi nilai piksel [0, 1]
            
            # 4. Lakukan Prediksi Menggunakan Model CNN
            predictions = model.predict(img_array)
            
            # Menggunakan numpy argmax langsung agar lebih aman dan efisien
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = CLASS_NAMES[predicted_class_idx].upper()
            
            # Hitung persentase keyakinan model
            confidence = float(predictions[0][predicted_class_idx] * 100)
            
            return jsonify({
                'success': True,
                'class_name': predicted_class,
                'confidence': f"{confidence:.2f}%",
                'image_url': filepath.replace('\\', '/')  # Menyelaraskan path format URL di Windows
            })
            
        except Exception as e:
            return jsonify({'error': f"Gagal memproses gambar: {str(e)}"}), 500
            
    return jsonify({'error': 'Format file tidak diizinkan'}), 400

if __name__ == '__main__':
    app.run(debug=True)