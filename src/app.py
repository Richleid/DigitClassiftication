from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Carga el modelo al iniciar la aplicación
model = load_model('cnn/cnn_model.h5')

@app.route('/')
def index():
    return render_template('Index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image = image.resize((28, 28))
        image_array = np.array(image)
        image_array = image_array.reshape(1, 28, 28, 1)
        image_array = image_array.astype('float32')
        image_array /= 255.0
        prediction = model.predict(image_array)
        return jsonify({'prediction': int(prediction.argmax())})

    return jsonify({'error': 'Something went wrong'}), 500

if __name__ == '__main__':
    app.run(debug=True)