Project Description:

This project implements a deep Convolutional Neural Network for multi-class facial emotion recognition.
The model is trained to classify facial expressions into seven different emotion categories:
- Happy
- Sad
- Fear
- Surprise
- Neutral
- Angry
- Disgust

Dataset:
FER-2013 dataset
Source: Kaggle
License: Public

This system includes a complete machine learning pipeline:
- Dataset loading
- Preprocessing and caching
- Model training
- CLI-based interface
- Label mapping for consistent predictions

Model Architecture:
The CNN architecture includes: 
- 4 convolutional blocks (Conv2D, BatchNorm, LeakyReLU)
- MaxPooling for spacial reduction
- Dropout for regularization
- Global Average Pooling
- Fully connected classifier
- Softmax output layer

Tech Stack:
- Python
- TensorFlow/Keras
- NumPy
- OpenCV

Installation:
Clone the repository:
- git clone 
- cd EmotionClassifier
Create a virtual environment:
- python -m venv venv
- source venv/bin/activate
- pip install -r requirements.txt
Training the model:
- python scripts/train.py
- (This loads and preprocesses the dataset, trains the CNN model, saves the trained model to the disk)
Running the inference:
- python scripts/predict.py path/to/image.jpg
- (EXAMPLE) python scripts/predict.py data/test/happy/PrivateTest_95094.jpg

Results:
Validation Accuracy: ~65%
Input size: 48x48 grayscale
Optimizer: Adam
Loss: Sparse Categorical Crossentropy


Summary:
Designed and trained a deep convolutional neural network for multi-class facial emotion classification.
Built an ML pipeline including preprocessing, model training, and CLI-based interface.
Implemented regularization techniques to improve generalization performance.