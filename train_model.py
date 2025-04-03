import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys

# Set UTF-8 encoding
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')

# Disable floating-point round-off warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Dataset and Parameters
DATA_DIR = r"C:\Users\LOHITH.P.K\Documents\PLANB\dataset\train"  # Replace with your dataset path
IMG_SIZE = 64  # Resize all images to 64x64
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 26  # 26 alphabets

# Data Preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True  # Shuffle training data
)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True  # Shuffle validation data
)

# Model Definition
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),  # Explicitly define input shape here
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model Training
if __name__ == "__main__":
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_steps=val_generator.samples // BATCH_SIZE
    )

    # Save the Model
    MODEL_SAVE_PATH = "hand_sign_classifier.h5"
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved at {MODEL_SAVE_PATH}")

    # Evaluate the Model
    loss, accuracy = model.evaluate(val_generator)
    print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")
