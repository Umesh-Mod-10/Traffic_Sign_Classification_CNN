# **Traffic Sign Classification Using CNN**

This repository contains a project for classifying German traffic signs using a Convolutional Neural Network (CNN). The dataset used in this project is the German Traffic Sign Recognition Benchmark (GTSRB). The model was built using Keras with TensorFlow as the backend, and evaluated with metrics such as accuracy, confusion matrix, and classification reports.

## Table of Contents
- Dataset: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
- Model Architecture
- Data Preprocessing and Augmentation
- Training
- Evaluation
- Results

## Dataset
The GTSRB dataset consists of 43 classes representing different traffic signs. The dataset has 39,209 images for training and 7,841 images for testing.

Train images: 39,209
Test images: 7,841
Classes: 43
For a detailed breakdown of the dataset, a bar chart showing class distribution can be seen in the project.

## Model Architecture
The CNN architecture for this task is built using Keras' Sequential API. The network includes:

- Input layer: Rescaling the image values to the range [0, 1].
- Convolutional layers: 4 convolutional layers with ReLU activation and max-pooling.
- Batch Normalization: Applied after each convolutional layer to improve stability and training speed.
- Dense layers: Fully connected layers for classification with softmax activation for the output.

## Summary of the model:
- Input shape: (128, 128, 3)
- Output: 43 classes (softmax)
- Data Preprocessing and Augmentation

Since the dataset has imbalanced class distribution, data augmentation techniques such as:
- Random flipping
- Random rotation
- Random zoom were applied to improve generalization. The training and validation data were split in an 80-20 ratio.

## Training
The model was trained with the following configurations:

- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy
- Metrics: Accuracy
- Epochs: 30 (early stopping applied after epoch 17)

## Callbacks:
- EarlyStopping: Monitors validation accuracy with patience of 10 epochs.
- ReduceLROnPlateau: Reduces learning rate when validation loss plateaus.
- ModelCheckpoint: Saves the best model based on validation accuracy.

## Evaluation
The model was evaluated using:

- Accuracy on the test set
- Confusion matrix
- Classification report (precision, recall, F1-score)

## Key metrics:
- Accuracy: The final accuracy achieved was approximately X%.
- Confusion Matrix: A heatmap visualizing the model's classification performance on each class.
- Classification Report: A detailed analysis of precision, recall, and F1-score for each class.

## Results
The results showed that the CNN model performed well on most classes, but there were some misclassifications in similar-looking signs (e.g., speed limit signs). The accuracy trend during training and validation is visualized in the graphs below.

## Loss and Accuracy Trends:
Training and validation accuracy increased steadily, with early stopping preventing overfitting.
Loss decreased gradually over epochs, as shown in the loss vs. val_loss graph.
