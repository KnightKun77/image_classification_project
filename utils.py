# utils.py

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot_history(history):
    # Accuracy plot
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((32, 32))  # Resize image to match CIFAR-10 input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    return img_array.reshape(1, 32, 32, 3)  # Reshape for the model
