"""
predict.py

Loads a trained emotion classification model and predicts the emotion 
of input images with CLI support.

run "python scripts/predict.py <path to the image>" In order to get a classification for that image using the model
"""
import os
import json
import tensorflow as tf
from preprocess import preprocess_single_image
import cv2
import numpy as np
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "emotion_model.h5")
MAPPING_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "mapping.json")

def load_trained_model(model_path):
    """
    Loads the trained model
    
    :param model_path: the path to the model
    :return model: the actual model is returned
    """
    print("Loading trained model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
    return model

def load_mapping(mapping_path):
    """
    Loads the mapping of the emotion categories
    
    :param mapping_path: the file path to the indexed emotion categories
    :return mapping: the loaded mapping of categories
    :return index_to_emotion: indices mapped to emotions rather than the other way around
    """
    print("Loading label mapping...")
    with open(mapping_path, "r") as f:
        mapping = json.load(f)

    index_to_emotion = {v: k for k, v in mapping.items()}
    return mapping, index_to_emotion

def load_and_preprocess_image(image_path):
    """
    Loads and preprocesses a single image
    
    :param image_path: the file path to a singular image
    :return img: the preprocessed image
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not load image")

    img = preprocess_single_image(img)
    img = np.expand_dims(img, axis=0)  

    return img

def predict_emotion(model, image, index_to_emotion):
    """
    predicts/classifies the emotion of an image along with a confidence level
    using the model that was previously trained using the training dataset
    
    :param model: the model used to create the classification
    :param image: the image that is being classified
    :param index_to_emotion: indices mapped to emotion categories
    :return emotion: the emotion predicted by the model
    :return confidence: the confidence level that the model has with its classification
    """
    preds = model.predict(image)
    class_idx = np.argmax(preds)
    confidence = preds[0][class_idx]

    emotion = index_to_emotion[class_idx]
    return emotion, confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict.py <path_to_image>")
        sys.exit(1)
    IMAGE_PATH = sys.argv[1] 

    model = load_trained_model(MODEL_PATH)
    mapping, index_to_emotion = load_mapping(MAPPING_PATH)

    print("Preprocessing image...")
    image = load_and_preprocess_image(IMAGE_PATH)

    print("Predicting emotion...")
    emotion, confidence = predict_emotion(model, image, index_to_emotion)

    print(f"Predicted emotion: {emotion} ({confidence:.2f})")