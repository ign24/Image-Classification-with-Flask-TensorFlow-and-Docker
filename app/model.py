import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.datasets import fashion_mnist
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# Configurar el logger para mostrar información durante la ejecución
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parámetros configurables a través de variables de entorno para hacer el código más flexible
SUBSET_PERCENTAGE = float(os.getenv('SUBSET_PERCENTAGE', 0.3))  # Porcentaje del conjunto de datos a utilizar
IMAGE_SIZE = int(os.getenv('IMAGE_SIZE', 32))                   # Tamaño de la imagen después del redimensionamiento
EPOCHS = int(os.getenv('EPOCHS', 10))                           # Número de épocas para el entrenamiento
MODEL_FILENAME = os.getenv('MODEL_FILENAME', 'fashion_mnist_resnet50.keras')  # Nombre del archivo para guardar el modelo

def load_data(subset_percentage=SUBSET_PERCENTAGE):
    """Carga y procesa los datos del conjunto de datos Fashion MNIST."""
    logging.info("Loading and processing data...")
    
    # Cargar las imágenes y etiquetas del conjunto de datos Fashion MNIST
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Submuestrear los datos para usar solo un porcentaje de ellos
    total_train_samples = int(len(train_images) * subset_percentage)
    total_test_samples = int(len(test_images) * subset_percentage)
    
    train_images = train_images[:total_train_samples]
    train_labels = train_labels[:total_train_samples]
    test_images = test_images[:total_test_samples]
    test_labels = test_labels[:total_test_samples]

    logging.info("Data loading and processing complete.")
    return (train_images, train_labels), (test_images, test_labels)

def process_data(train_images, train_labels, test_images, test_labels, image_size=IMAGE_SIZE):
    """Preprocesa las imágenes para que estén listas para ser usadas por el modelo."""
    logging.info("Starting data preprocessing...")

    # Expandir la dimensión de las imágenes a (altura, ancho, 1) para añadir un canal de color
    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)

    # Convertir las imágenes de escala de grises a RGB y redimensionarlas al tamaño especificado
    train_images = tf.image.grayscale_to_rgb(tf.image.resize(train_images, [image_size, image_size]))
    test_images = tf.image.grayscale_to_rgb(tf.image.resize(test_images, [image_size, image_size]))

    # Preprocesar las imágenes usando la función de preprocesamiento de ResNet50
    train_images = preprocess_input(train_images.numpy())
    test_images = preprocess_input(test_images.numpy())

    # Dividir los datos de entrenamiento en conjuntos de entrenamiento y validación
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.3, random_state=42, stratify=train_labels)

    logging.info("Data preprocessing complete.")
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)

def create_model(image_size=IMAGE_SIZE):
    """Crea y compila el modelo basado en ResNet50 con capas personalizadas en la parte superior."""
    logging.info("Creating model...")
    
    # Cargar el modelo base ResNet50 sin la capa superior (include_top=False)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

    # Crear nuevas capas en la parte superior del modelo base
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = base_model(inputs, training=False)  # Mantener las capas de ResNet50 congeladas
    x = GlobalAveragePooling2D()(x)         # Añadir una capa de pooling global para reducir dimensionalidad
    x = Dropout(0.2)(x)                     # Añadir una capa de Dropout para regularización
    outputs = Dense(10, activation='softmax')(x)  # Capa final con 10 unidades (una por clase) y activación softmax

    model = Model(inputs, outputs)

    # Congelar las capas del modelo base para que no se entrenen
    base_model.trainable = False

    # Compilar el modelo con optimizador Adam y pérdida de entropía cruzada
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    logging.info("Model created successfully.")
    return model

def train_and_evaluate_model(model, train_images, train_labels, val_images, val_labels, test_images, test_labels, epochs=EPOCHS):
    """Entrena el modelo y lo evalúa en el conjunto de datos de prueba."""
    logging.info("Starting model training...")
    
    # Entrenar el modelo en los datos de entrenamiento y validar en los datos de validación
    history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(val_images, val_labels))
    
    logging.info("Training complete. Evaluating model...")
    
    # Evaluar el modelo en los datos de prueba
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    logging.info(f"Test accuracy: {test_acc}")

    # Generar un informe de clasificación detallado
    logging.info("Generating classification report...")
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    report = classification_report(test_labels, predicted_labels, target_names=[
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ])
    
    logging.info(f"\nClassification Report:\n{report}")

    # Guardar el informe de clasificación en un archivo
    with open('classification_report.txt', 'w') as f:
        f.write(report)
    
    return history, test_loss, test_acc

def save_model(model, filename=MODEL_FILENAME):
    """Guarda el modelo entrenado en un archivo."""
    logging.info("Saving model...")
    model.save(filename)
    logging.info(f"Model saved as {filename}.")

def plot_results(history):
    """Genera y muestra un gráfico con la evolución de la precisión durante el entrenamiento."""
    logging.info("Plotting results...")
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()
    logging.info("Plotting complete.")

if __name__ == '__main__':
    # Inicia la ejecución del pipeline completo
    logging.info("Starting pipeline...")
    
    # Cargar y procesar los datos
    (train_images, train_labels), (test_images, test_labels) = load_data()
    (train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = process_data(
        train_images, train_labels, test_images, test_labels)
    
    # Crear el modelo
    model = create_model()

    # Entrenar y evaluar el modelo
    history, test_loss, test_acc = train_and_evaluate_model(model, train_images, train_labels, val_images, val_labels, test_images, test_labels)

    # Guardar el modelo entrenado
    save_model(model)

    # Graficar los resultados del entrenamiento
    plot_results(history)

    logging.info("Pipeline complete.")