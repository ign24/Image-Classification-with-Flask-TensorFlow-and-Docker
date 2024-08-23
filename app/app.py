import os
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from utilities import process_image
from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__)

# Monitoreo de la aplicación con Prometheus
metrics = PrometheusMetrics(app)

# Cargar el modelo en el inicio de la aplicación
MODEL_PATH = os.getenv('MODEL_PATH', '/app/models/fashion_mnist_resnet50.keras')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

model = None
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {str(e)}")

# Verificar si un archivo tiene una extensión permitida
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        filename = secure_filename(file.filename)
        predictions = process_image(file, model)
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)