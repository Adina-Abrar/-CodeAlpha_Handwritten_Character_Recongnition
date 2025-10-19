# Handwritten Character Recognition using CRNN

## Overview

This project demonstrates the use of a **Convolutional Recurrent Neural Network (CRNN)** for recognizing handwritten characters and digits. The model is designed to classify both **digits (MNIST)** and **characters (EMNIST)** with the capability to be extended for word-level or sentence-level recognition.

## Table of Contents
- [Handwritten Character Recognition using CRNN](#handwritten-character-recognition-using-crnn)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Requirements](#requirements)
  - [Dataset](#dataset)
    - [MNIST Dataset](#mnist-dataset)
    - [EMNIST Dataset](#emnist-dataset)
  - [Model Architecture](#model-architecture)
    - [CRNN Model](#crnn-model)
  - [Training and Evaluation](#training-and-evaluation)
    - [Training Results](#training-results)
  - [Performance](#performance)
  - [Usage](#usage)
  - [Future Work](#future-work)
  - [References](#references)

## Introduction

This project utilizes a **CRNN model** to classify handwritten digits (MNIST dataset) and characters (EMNIST dataset). The architecture combines **Convolutional Neural Networks (CNNs)** for feature extraction and **Recurrent Neural Networks (RNNs)**, such as LSTM or GRU, for sequence prediction.

The model can recognize individual **characters** (A-Z) or **digits** (0-9). Additionally, it can be extended to recognize full words or sentences by using sequence modeling.

## Requirements

Before running the project, ensure that the following dependencies are installed:

- **Python 3.x**
- **TensorFlow 2.x**
- **Keras**
- **NumPy**
- **Matplotlib**
- **Pandas**
- **OpenCV**
- **TensorFlow Datasets (TFDS)**

## Dataset

### MNIST Dataset

- **MNIST** contains 60,000 training images and 10,000 testing images of handwritten digits (0-9).
- Each image is 28x28 pixels, grayscale, and labeled with the corresponding digit.

### EMNIST Dataset

- **EMNIST** extends MNIST by adding **letters (A-Z)** along with digits (0-9).
- It consists of **47 classes**, including uppercase English letters and digits.

Both datasets are preprocessed by normalizing the images and converting the labels into **one-hot encoding**.

## Model Architecture

### CRNN Model

The **CRNN model** is designed to handle both image data (handwritten digits/characters) and sequence prediction (recognizing words). It consists of:

1. **Convolutional Layers (CNN)**: These layers extract key features from the images, enabling the model to understand patterns.
2. **Recurrent Layers (LSTM/GRU)**: These layers handle sequence prediction, enabling the model to process sequential data (e.g., text).
3. **Fully Connected Layers**: These layers help output the final predictions, which are the class labels (digits or characters).

## Training and Evaluation

The model is trained using **categorical cross-entropy loss** and the **Adam optimizer**. The training set is split into 80% for training and 20% for validation. **Early stopping** is used to prevent overfitting by monitoring the validation loss.

### Training Results

After training the models on both MNIST and EMNIST datasets, the models achieved the following:

- **MNIST Model**:
  - Training Accuracy: **98.6%**
  - Validation Accuracy: **98.2%**

- **EMNIST Model**:
  - Training Accuracy: **94.5%**
  - Validation Accuracy: **93.0%**

## Performance

- **Training Loss for MNIST**: 0.08
- **Validation Loss for MNIST**: 0.09
- **Training Loss for EMNIST**: 0.14
- **Validation Loss for EMNIST**: 0.16

These results demonstrate that the models perform well, with slight differences between training and validation accuracies.

## Usage

To use the model for prediction on a new image, follow these steps:

1. **Preprocess the image** by resizing it to 28x28 pixels and normalizing it.
2. **Load the trained model**.
3. **Predict the label** for the image using the model.

The model can predict digits for MNIST and characters for EMNIST. It outputs the corresponding class (e.g., digit or character) for the image.

## Future Work

- **Improving Generalization**: Techniques like **data augmentation** and **regularization** can be used to improve the model's performance on unseen data.
- **Expanding to Word-Level Prediction**: The CRNN model can be extended to handle **word or sentence-level prediction** by processing sequences of characters.
- **Model Optimization**: Implement **model compression** techniques to reduce model size and speed up inference for real-time deployment.
- **Multilingual Support**: Adding support for multiple languages and character sets would allow the model to handle multilingual text recognition.

## References

1. **LeCun, Y., et al.** "Gradient-based learning applied to document recognition." *Proceedings of the IEEE*, 1998.
2. **Cohen, M., et al.** "EMNIST: Extending MNIST to Handwritten Letters." 2017.
3. **Graves, A., et al.** "Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks." 2006.


