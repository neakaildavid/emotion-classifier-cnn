"""
preprocess.py

Preprocesses the training set images getting it ready to input
into the model and train the model. Normalizes all the images to be
compatable with the input type of the model.
"""

import os
import json
import random
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from data_loader import load_dataset

IMAGE_SIZE = (48, 48)
NORMALIZE = True  

AUGMENTATION = {
    "flip": True,
    "rotation": 10,
}

VAL_SIZE = 0.1
RANDOM_SEED = 42
CACHE_DIR = "data/preprocessed"


def ensure_dir(path):
    """
    Makes sure the path exists.
    
    :param path: the path of the directory we are accessing
    """
    if not os.path.exists(path):
        os.makedirs(path)


def set_random_seed(seed):
    """
    Sets a random seed
    
    :param seed: the seed we want to input to random
    """
    random.seed(seed)
    np.random.seed(seed)


def reshape_add_channel(img):
    """
    reshapes the input image and adds an extra channel
    
    :param img: the image we are reshaping
    :return img: the reshaped image
    """
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    return img


def resize_img(img, size=IMAGE_SIZE):
    """
    Resizes the input image
    
    :return cv2.resize(img, size, interpolation=cv2.INTER_AREA): the input image resized
    """
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def preprocess_dataset(raw_data_path, image_size=IMAGE_SIZE, augment=False, cache=False):
    """
    Fully preprocesses the dataset
    
    :param raw_data_path: the file path to the dataset
    :param image_size: the size of each image
    :param augment: whether to augment or not
    :param cache: whether to cache or not

    :return X_train: preprocessed training dataset
    :return X_val: preprocessed testing dataset
    :return Y_train: preprocessed training dataset's labels
    :return Y_val: preprocessed testing dataset's labels
    :return mapping: emotions mapped to indices
    """
    if cache and os.path.exists(CACHE_DIR):
        try:
            X_train = np.load(os.path.join(CACHE_DIR, "X_train.npy"))
            X_val = np.load(os.path.join(CACHE_DIR, "X_val.npy"))
            y_train = np.load(os.path.join(CACHE_DIR, "y_train.npy"))
            y_val = np.load(os.path.join(CACHE_DIR, "y_val.npy"))

            with open(os.path.join(CACHE_DIR, "mapping.json")) as f:
                mapping = json.load(f)

            print("Loaded cached preprocessed data.")
            return X_train, X_val, y_train, y_val, mapping
        except Exception:
            print("Cache found but failed to load. Reprocessing...")

    set_random_seed(RANDOM_SEED)


    images, labels, mapping = load_dataset(raw_data_path)

    processed_images = []

    for img in images:
        img = resize_img(img, image_size)
        img = reshape_add_channel(img)

        processed_images.append(img)

    X = np.array(processed_images)  
    y = np.array(labels)

    if NORMALIZE:
        X = X.astype("float32") / 255.0


    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VAL_SIZE,
        stratify=y,
        random_state=RANDOM_SEED
    )

    if augment:
        augmented_images = []
        augmented_labels = []

        for img, label in zip(X_train, y_train):
            aug_img = img.copy()

            if AUGMENTATION.get("flip") and random.random() < 0.5:
                aug_img = cv2.flip(aug_img, 1)

            angle = random.uniform(-AUGMENTATION.get("rotation", 0), AUGMENTATION.get("rotation", 0))
            h, w = aug_img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, M, (w, h))

            aug_img = reshape_add_channel(aug_img)

            augmented_images.append(aug_img)
            augmented_labels.append(label)

        X_train = np.concatenate([X_train, np.array(augmented_images)])
        y_train = np.concatenate([y_train, np.array(augmented_labels)])

    print("Train shape:", X_train.shape, y_train.shape)
    print("Val shape:", X_val.shape, y_val.shape)
    print("Pixel range:", X_train.min(), X_train.max())

    if X_train.ndim != 4:
        raise ValueError("Expected input shape (N, H, W, C)")

    if cache:
        ensure_dir(CACHE_DIR)

        np.save(os.path.join(CACHE_DIR, "X_train.npy"), X_train)
        np.save(os.path.join(CACHE_DIR, "X_val.npy"), X_val)
        np.save(os.path.join(CACHE_DIR, "y_train.npy"), y_train)
        np.save(os.path.join(CACHE_DIR, "y_val.npy"), y_val)

        with open(os.path.join(CACHE_DIR, "mapping.json"), "w") as f:
            json.dump(mapping, f)

        print("Saved preprocessed data to cache.")

    return X_train, X_val, y_train, y_val, mapping

def preprocess_single_image(img):
    """
    preprocesses a single image
    
    :param img: the image to preprocess
    :return img: the preprocessed image
    """
    img = resize_img(img, IMAGE_SIZE)
    img = reshape_add_channel(img)

    if NORMALIZE:
        img = img.astype("float32") / 255.0

    return img

if __name__ == "__main__":
    X_train, X_val, y_train, y_val, mapping = preprocess_dataset(
        raw_data_path="data",
        augment=False,
        cache=True
    )

    print("Preprocessing complete.")

