import tensorflow as tf
import numpy as np
import cv2
import os
import sys

# Ensure UTF-8 encoding
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")

# Parameters
MODEL_PATH = "hand_sign_classifier.h5"  # Path to the saved model
IMG_SIZE = 64  # Same size used during training
CLASS_LABELS = [chr(i) for i in range(65, 91)]  # A-Z

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# Function to preprocess the image
def preprocess_image(image_path):
    """
    Preprocesses the image for prediction.

    Args:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray: Preprocessed image ready for prediction.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to make a prediction
def predict_hand_sign(image_path):
    """
    Predicts the hand sign from an image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Predicted label.
    """
    img = preprocess_image(image_path)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)  # Index of the highest probability
    confidence = np.max(predictions)  # Probability of the predicted class
    return CLASS_LABELS[predicted_class], confidence

# Example usage
if __name__ == "__main__":
    # Replace with the path to your test image
    TEST_IMAGE_PATH = r"C:\Users\LOHITH.P.K\Documents\PLANB\dataset\train\A\1.png"
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Test image not found: {TEST_IMAGE_PATH}")
    else:
        try:
            predicted_label, confidence = predict_hand_sign(TEST_IMAGE_PATH)
            print(f"Predicted Label: {predicted_label}")
            print(f"Confidence: {confidence:.2f}")
        except Exception as e:
            print(f"An error occurred: {e}")
