import cv2
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from image_model import link_detect
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Get the directory where the script is saved
script_dir = os.path.dirname(os.path.abspath('knn_Model'))

# Set the working directory to the script directory
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
    bald_images.append(np.array(img).flatten())

# Load the not bald images
not_bald_images = []
for filename in os.listdir(not_bald_dir):
    img = cv2.imread(os.path.join(not_bald_dir, filename))
    img = cv2.resize(img, IMG_SIZE)
    not_bald_images.append(np.array(img).flatten())

# Combine the bald and not bald images into a single dataset
X = np.array(bald_images + not_bald_images)

# Create the corresponding labels for the dataset (1 for bald, 0 for not bald)
y = np.array([1] * len(bald_images) + [0] * len(not_bald_images))
link_detect()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a Random Forest classifier and fit it to the training data
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model on the testing data
y_pred = rf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Perform cross-validation and print the mean accuracy score
scores = cross_val_score(rf, X, y, cv=5)
print(f"Cross-validation mean accuracy: {np.mean(scores)}")
