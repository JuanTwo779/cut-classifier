# Cut Classifier 

## Overview
A machine learning model that classifies haircuts based on images. Users can upload a photo of a haircut, and the model predicts the type of haircut using a Convolutional Neural Network (CNN).  

## Features
- Image upload and classification  
- Backend powered by **Python + TensorFlow**
- REST API for predictions  
- Predictions include a haircut type and accuracy score (100)
- Categories = ['Burst Fade','Buzz Cut','Caesar Cut','Comb Over','Drop Fade','French Crop','High Fade','Ivy League','Low Fade','Mid Fade','Mid Part','Modern Mullet','Quiff','Side Part','Taper Fade']

## Tech Stack  
- **Backend:** Python (Flask)  
- **Machine Learning:** TensorFlow (CNN model)  
- **Deployment:** AWS....

## Flow
1. jpg/jpeg file passed to flask endpoint via react frontend
2. validation ? valid file, continue : json error returned to frontend 
3. convert image to BytesIO stream
4. resize image in accordance with model
5. image passed to and processed by model
6. prediction is made and result/error is sent to frontend

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
- `tensorflow-cpu`: create CNN model + (backend) preprocess image, import Keras and load the model
- `numpy`: library for working with arrays; prediction and score
- `io`: converts image to BytesIO stream
- `Flask_Limiter`: limit the amount of calls per client
- `flask-cors`: allows cross-origin AJAX requests in flask
- `Pillow`: provides extensive file format support