
import os
import sys
import tensorflow as tf
import numpy as np
import cv2
from flask import Flask, request, render_template
from gtts import gTTS
from PIL import Image

# Flask App
app = Flask(__name__)

# Ensure UTF-8 encoding
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")

# Parameters
MODEL_PATH = "hand_sign_classifier.h5"  # Path to the saved TensorFlow model
IMG_SIZE = 64  # Size used during training
CLASS_LABELS = [chr(i) for i in range(65, 91)] + ["_", "RESET"]  # A-Z + space + reset

# Load the TensorFlow model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# Initialize sentence storage
sentence = ""

# Preprocessing Function
def preprocess_image(image_path):
    
   # Preprocesses the image for prediction.
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Prediction Function
def predict_hand_sign(image):
    
   # Predicts the hand sign from an image.
    
    global sentence  # Use the global sentence variable

    temp_image_path = "temp_image.jpg"
    image.save(temp_image_path)

    img = preprocess_image(temp_image_path)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)  # Index of the highest probability
    confidence = np.max(predictions)  # Probability of the predicted class

    os.remove(temp_image_path)

    detected_char = CLASS_LABELS[predicted_class]

    # Handle special cases
    if detected_char == "_":
        sentence += " "  # Add space
    elif detected_char == "RESET":
        sentence = ""  # Reset the sentence
    else:
        sentence += detected_char  # Append letter

    return sentence, confidence

# Text-to-Speech Function
def text_to_audio(text):
    
   # Converts full sentence to speech using gTTS.
    
    tts = gTTS(text=text, lang='en')
    audio_path = 'static/audio/output.mp3'
    tts.save(audio_path)
    return audio_path

# Flask Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            image = Image.open(file.stream)

            # Predict hand signs and form a sentence
            full_sentence, confidence = predict_hand_sign(image)

            # Convert the full sentence to speech
            audio_file = text_to_audio(full_sentence)

            return render_template(
                'result.html',
                text=full_sentence,
                confidence=f"{confidence:.2f}",
                audio=audio_file
            )
        else:
            return "No file uploaded!", 400

    return render_template('index.html')

# Run the App
if __name__ == "__main__":
    app.run(debug=True)

