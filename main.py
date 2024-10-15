import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Label, Button, Frame
import numpy as np
from PIL import Image, ImageTk  # Import for image handling
from dataset import load_dataset, get_class_names
from model import create_model
from utils import preprocess_image

# Load dataset
(x_train, y_train), (x_test, y_test) = load_dataset()
class_names = get_class_names()

# Create model
model = create_model()

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64)

# GUI setup
class ImageClassifierApp:
    def __init__(self, master):
        self.master = master
        master.title("Image Classifier")

        self.frame = Frame(master)
        self.frame.pack(padx=10, pady=10)

        self.label = Label(self.frame, text="Upload an image to classify:")
        self.label.pack(pady=10)

        self.upload_button = Button(self.frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=5)

        self.result_label = Label(self.frame, text="")
        self.result_label.pack(pady=10)

        self.image_label = Label(self.frame)  # Label for showing the uploaded image
        self.image_label.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.display_image(file_path)
            self.classify_image(file_path)

    def display_image(self, file_path):
        # Load and display the uploaded image in the UI
        img = Image.open(file_path)
        img = img.resize((150, 150))  # Resize image for display in UI
        img_tk = ImageTk.PhotoImage(img)  # Convert image for Tkinter

        # Update the label to show the image
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk  # Keep a reference to avoid garbage collection

    def classify_image(self, file_path):
        img_array = preprocess_image(file_path)
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        self.result_label.config(text=f"Predicted: {predicted_class}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()
