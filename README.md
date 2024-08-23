# Image Classification with Flask, TensorFlow, and Docker

This project implements a web application for image classification using a deep learning model based on the ResNet50 architecture. The project uses Flask as the web framework, TensorFlow/Keras for building and training the model, Docker for containerization, and Prometheus along with Grafana for monitoring and metric visualization.


<p align="center">
  <img src="https://github.com/user-attachments/assets/471212bb-c077-45ca-9e54-843f1aee02f5" alt="Pipeline Diagram" width="600"/>
</p>


## Features

- **Image Classification:** Users can upload images through a web interface and receive predictions for the clothing category.
- **Deep Learning Model:** The model used is based on ResNet50, trained on a subset of the Fashion MNIST dataset.
- **Monitoring:** Prometheus is used to collect metrics from the Flask application, and Grafana is used for visualizing these metrics.
- **Containerization:** Docker and Docker Compose are used to facilitate the deployment of the development and production environments.

## Technologies Used

- **Flask:** Web framework for creating the API and the frontend of the application.
- **TensorFlow and Keras:** Libraries for building, training, and running the image classification model.
- **Prometheus:** Monitoring system for collecting application metrics.
- **Grafana:** Analytics and monitoring platform for visualizing metrics collected by Prometheus.
- **Docker:** Containerization tool that allows packaging the application along with its dependencies.
- **Docker Compose:** Container orchestrator for defining and running multi-container applications.

## Installation

### Prerequisites

Docker and Docker Compose installed on your machine.

## Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/ign24/Image-Classification-with-Flask-TensorFlow-and-Docker.git
   cd Image-Classification-with-Flask-TensorFlow-and-Docker

2. Configure the environment variables in the .env file. Make sure the paths and configurations are correct.

3. Build and bring up the containers using Docker Compose:

          docker-compose up --build

4. Access the web application at http://localhost:5000.

### Usage

1. Navigate to the home page (http://localhost:5000).

2. Upload an image for classification.

3. The system will return the predicted class and the prediction confidence.

### Model Training

If you want to train the model from scratch:

1. Adjust the parameters in model.py according to your needs.
2. Run the script to train and save the model:

         python model.py


## Monitoring

Metrics from the Flask application are exposed at http://localhost:5000/metrics and can be viewed in Grafana by accessing http://localhost:3000.

## Results
The current model achieves approximately 81% accuracy on the Fashion MNIST test set, with the following metrics for different classes:


| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| T-shirt/top  | 0.78      | 0.75   | 0.77     | 302     |
| Trouser      | 0.94      | 0.98   | 0.96     | 308     |
| Pullover     | 0.62      | 0.83   | 0.71     | 310     |
| Dress        | 0.85      | 0.77   | 0.81     | 298     |
| Coat         | 0.73      | 0.53   | 0.62     | 324     |
| Sandal       | 0.97      | 0.91   | 0.93     | 285     |
| Shirt        | 0.55      | 0.55   | 0.55     | 298     |
| Sneaker      | 0.88      | 0.93   | 0.91     | 293     |
| Bag          | 0.95      | 0.96   | 0.96     | 297     |
| Ankle boot   | 0.94      | 0.96   | 0.95     | 285     |
|              |           |        |          |         |
| **Accuracy** |           |        | 0.81     | 3000    |
| **Macro avg**| 0.82      | 0.82   | 0.82     | 3000    |
| **Weighted avg**| 0.82   | 0.81   | 0.81     | 3000    |



## Dataset

This project uses the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist), created by Zalando Research. Fashion MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples—intended to serve as a drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms.

You can read more about it and download the dataset from [here](https://github.com/zalandoresearch/fashion-mnist).

## Contributions

Contributions are welcome. Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
