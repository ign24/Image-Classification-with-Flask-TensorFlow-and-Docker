# Usamos la imagen base de TensorFlow
FROM tensorflow/tensorflow:2.17.0

# Establecemos el directorio de trabajo en /app
WORKDIR /app

# Copiamos el archivo de requerimientos desde la carpeta app
COPY app/requirements.txt .

# Instalar las dependencias y prepara el entorno
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Copiar todo el código fuente desde la carpeta app al contenedor
COPY app/ .

# Exponer el puerto 5000 para Flask
EXPOSE 5000

# Ejecutar la aplicación Flask
CMD ["python", "app.py"]