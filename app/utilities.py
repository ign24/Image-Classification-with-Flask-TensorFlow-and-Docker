from PIL import Image
import numpy as np
from keras.applications.resnet50 import preprocess_input

def process_image(file, model):
    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]
    
    try:
        # Cargar y procesar la imagen
        image = Image.open(file).convert('RGB')
        image = image.resize((32, 32))
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")
    
    try:
        # Convertir la imagen a un array numpy y preprocesarla
        image_array = np.array(image).astype('float32')
        image_array = preprocess_input(image_array)  # Preprocesar para ResNet50
        image_array = np.expand_dims(image_array, axis=0)  # Añadir la dimensión de batch
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

    try:
        # Realizar la predicción
        predictions = model.predict(image_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
    except Exception as e:
        raise ValueError(f"Error during prediction: {str(e)}")

    predicted_class_name = class_names[predicted_class_index]

    return {
        'predicted_class': predicted_class_name,
        'confidence': float(confidence),
    }