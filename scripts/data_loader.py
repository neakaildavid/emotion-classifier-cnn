"""
data_loader.py

Loads the dataset of images, each image's true label, and maps all 
possible emotions to indices.
"""

import os
import cv2
import numpy as np


def load_dataset(dataSetPath):
    """
    Loads the dataset
    
    :param dataSetPath: the path to the dataset that is being loaded
    
    :return images: loaded images
    :return labels: loaded labels for images
    :return indexed emotions: list of potential emotions indexed
    """
    images = []
    labels = []

    trainDirectory = os.path.join(dataSetPath, 'train')

    emotions =  sorted([f for f in os.listdir(trainDirectory) 
                if not f.startswith('.') and os.path.isdir(os.path.join(trainDirectory, f))])
    
    indexedEmotions = {emo: idx for idx, emo in enumerate(emotions)}
    
    for emotion in emotions:
        emotionPath = os.path.join(trainDirectory, emotion)
        for file in os.listdir(emotionPath):
            image_path = os.path.join(emotionPath, file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            images.append(image)
            labels.append(indexedEmotions[emotion])
    images = np.array(images)
    labels = np.array(labels)
    return images, labels, indexedEmotions


