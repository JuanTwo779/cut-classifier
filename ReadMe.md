# Cut Classification 

## Overview
Flask Rest API, The user uploads a .jpg image of a haircut which is fed to the model returning an output to determine the type of haircut. 

## Flow
1. jpg/jpeg file passed to flask endpoint via react frontend
2. validation ? correct file type, continue : json error returned to frontend 
3. resizing to in accordance with model
4. image passed to and processed by model
5. prediction is made and result/error is send back to frontend

## Neural Network
- Built using TensorFlow Keras
    - Works like API
- Datasets: Train, Test, Validation
    - Validation: Mimics real-world inputs and increases accuracy -> model validates results against this dataset
- Sequential Neural Network Model
    - Linear stack of layers where each layer has one input tensor and one output tensor

## Dependecies
- `Flask`
- `Tensorflow`
- `NumPy`
- `io`