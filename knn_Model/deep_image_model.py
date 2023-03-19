import cv2
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set the working directory to the directory where the script is saved
script_dir = os.path.dirname(os.path.abspath('nn_Model'))
os.chdir(script_dir)

# Define the directories where the images are located (EDIT PATHS TO FIT YOUR COMPUTER) 
bald_dir = 'Bald'
not_bald_dir = 'NotBald'

# Define the image size that the model expects
IMG_SIZE = (64, 64)

# Load the bald images
bald_images = []
for filename in os.listdir(bald_dir):
    img = cv2.imread(os.path.join(bald_dir, filename))
    img = cv2.resize(img, IMG_SIZE)
    bald_images.append(np.array(img))

# Load the not bald images
not_bald_images = []
for filename in os.listdir(not_bald_dir):
    img = cv2.imread(os.path.join(not_bald_dir, filename))
    img = cv2.resize(img, IMG_SIZE)
    not_bald_images.append(np.array(img))

# Combine the bald and not bald images into a single dataset
X = np.array(bald_images + not_bald_images)

# Create the corresponding labels for the dataset (1 for bald, 0 for not bald)
y = np.array([1] * len(bald_images) + [0] * len(not_bald_images))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model on the testing data
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
