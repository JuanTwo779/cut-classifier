# Cut Classifier 

## Overview
Backend for machine learning model that classifies haircuts based on images. Users can upload a photo of a haircut, and the model predicts the type of haircut using a Convolutional Neural Network (CNN).  

## Features
- Image upload and classification  
- Backend powered by **Python + TensorFlow**
- REST API for predictions  
- Predictions include a haircut type and accuracy score (100)
- Categories = ['Burst Fade','Buzz Cut','Caesar Cut','Comb Over','Drop Fade','French Crop','High Fade','Ivy League','Low Fade','Mid Fade','Mid Part','Modern Mullet','Quiff','Side Part','Taper Fade']

## Tech Stack  
- **Backend:** Python (Flask)  
- **Machine Learning:** TensorFlow Keras (CNN model)  
- **Deployment:** AWS EC2 (Linux), Gunicorn

## Flow  
1. A `jpg/jpeg` file is passed to the Flask endpoint via the React frontend.  
2. Validation: If the file is valid, proceed; otherwise, return a JSON error.  
3. Convert the image to a `BytesIO` stream.  
4. Resize the image in accordance with the model.  
5. The image is passed to and processed by the model.  
6. A prediction is made, and the result/error is sent to the frontend.

## Neural Network
- Built with TensorFlow Keras
- Datasets: Train, Test, and Validation
    - Training Set: Used to train the model
    - Test Set: Evaluates the model’s performance
    - Validation Set: Simulates real-world inputs, helping improve accuracy by fine-tuning hyperparameters
- Sequential Neural Network Model
    - A linear stack of layers, where each layer processes an input tensor and outputs a transformed tensor

## Dependecies
- `Flask`: lightweight WSGI web application framework; to create REST API
- `tensorflow-cpu`: creates the CNN model and handles image preprocessing  
- `numpy`: used for numerical operations in predictions 
- `io`: converts images to BytesIO stream
- `flask-cors`: allows cross-origin AJAX requests in flask
- `Pillow`: provides extensive file format support
- `gunicorn`: production-ready WSGI HTTP server for running Flask  