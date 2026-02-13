"""
train.py

Handles the training of the model through loading preprocessed data,
building CNN architecture, compiling and training model, and saving 
the trained model to disk.
"""

import os
import tensorflow as tf

from preprocess import preprocess_dataset
from model import build_emotion_model

DATA_PATH = "data"
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001

def main():
    print("Loading and preprocessing the data...")

    X_train, X_val, y_train, y_val, mapping = preprocess_dataset(raw_data_path=DATA_PATH, augment=True, cache=True)

    input_shape = X_train.shape[1:]
    num_classes = len(mapping)

    print("Input shape: ", input_shape)
    print("Number of classes: ", num_classes)

    print("building the model")
    model = build_emotion_model(input_shape=input_shape, num_classes=num_classes)
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    print("training the model")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size = BATCH_SIZE)

    os.makedirs("model", exist_ok=True)
    model_path = "model/emotion_model.h5"
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()