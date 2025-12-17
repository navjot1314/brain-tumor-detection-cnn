# Brain Tumor Detection using Convolutional Neural Networks (CNN)

## Overview
This project focuses on detecting the presence of brain tumors from MRI images
using Convolutional Neural Networks (CNN). The goal is to assist in early
diagnosis by automatically classifying MRI scans as tumor or non-tumor.

## Problem Statement
Manual analysis of MRI images is time-consuming and requires expert knowledge.
An automated image classification system can help in faster preliminary
screening and decision support.

## Dataset Description
The dataset consists of MRI brain images categorized into:
- Tumor
- Non-Tumor

Images are preprocessed and resized before being fed into the CNN model.

> Note: Dataset files are not uploaded to this repository due to size constraints.

## Methodology
1. MRI images are loaded and preprocessed (resizing, normalization).
2. Data is split into training and testing sets.
3. A Convolutional Neural Network is built using Python and deep learning libraries.
4. The model is trained on the training dataset.
5. Performance is evaluated using accuracy and loss metrics.

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib


## Project Status
Model architecture implemented and training workflow defined.
Further training and optimization in progress.

## Future Improvements
- Improve model accuracy using data augmentation
- Experiment with deeper CNN architectures
- Add performance metrics such as precision and recall
- Deploy the model as a simple web application

