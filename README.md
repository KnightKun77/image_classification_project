# Image Classification Using Python

## Overview

This project is an image classification application that utilizes the CIFAR-10 dataset. It enables users to upload images and classify them using a trained deep learning model. The application features a user-friendly graphical interface that allows for easy interaction and visualization of classification results.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Dependencies](#dependencies)
- [License](#license)
- [Contributing](#contributing)

## Features

- User-friendly GUI for easy image upload and classification.
- Image classification using a pre-trained model on the CIFAR-10 dataset.
- Displays predicted class and confidence scores for classified images.
- Option to upload multiple images for classification.

## Technologies Used

- **Python 3.x**
- **TensorFlow**: For model training and inference.
- **Keras**: For building and training the neural network.
- **NumPy**: For numerical operations.
- **OpenCV**: For image processing.
- **Tkinter**: For creating the graphical user interface (GUI).
- **Matplotlib**: For visualizing results (if applicable).

## Setup Instructions

Follow these steps to set up the project on your local machine:

1. **Clone the Repository:**
   Open your terminal or command prompt and run the following command:

   ```bash
   git clone https://github.com/KnightKun77/image_classification_project.git

2. **Navigate to the Project Directory: Change your working directory to the cloned project folder:**
    cd image_classification_project

3. **Create a Virtual Environment (Optional but Recommended):**
    python -m venv venv
   **Activate the virtual environment:**
    On Windows: venv\Scripts\activate
    On macOS/Linux: source venv/bin/activate

4. **Install Dependencies:**
   Install the required packages using pip. If you have a requirements.txt file, you can install dependencies with: 
   pip install -r requirements.txt
   If you don't have a requirements.txt file, you can manually install the necessary packages:
   pip install tensorflow keras numpy opencv-python matplotlib

## Usage

1. **Run the Application:** After setting up the environment and installing the dependencies, run the application with:
    bash
    python main.py

2. **Upload an Image:** Use the user interface to upload an image you want to classify. The application will process the image and display the predicted class 
    along with the confidence score.

3. **View Results:** The classification result will be displayed on the screen. You can upload another image to classify more images.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The classes are as follows:

Airplane
Automobile
Bird
Cat
Deer
Dog
Frog
Horse
Ship
Truck
The dataset is divided into 50,000 training images and 10,000 test images.

## Model Training

To train the model, ensure you have the CIFAR-10 dataset available. The training script (if provided) will load the dataset, preprocess the images, and build a convolutional neural network (CNN) to classify the images. Once trained, the model will be saved for use in the application.

Training Steps
Load the CIFAR-10 dataset.
Preprocess the images (normalization, resizing, etc.).
Build and compile the CNN model.
Train the model on the training dataset.
Evaluate the model on the test dataset.
Save the trained model for inference.

## Dependencies 
Make sure you have the following dependencies installed:

Python: Ensure you have Python 3.x installed.
TensorFlow: For model training and inference.
Keras: For building and training the neural network.
NumPy: For numerical operations.
OpenCV: For image processing.
Tkinter: For creating the GUI.
Matplotlib: For plotting (if applicable).

## Contributing

Contributions are welcome! If you have suggestions for improvements or want to add features, please fork the repository and submit a pull request.


