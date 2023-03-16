# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 18:47:43 2023

@author: Lyle_
"""

import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Define the directories where the images are located
# bald_dir = 'C:/Users/Lyle_/Desktop/4 Yr Sem 2/DS/Banzai-predicament/data/Validation/Bald'
# not_bald_dir = 'C:/Users/Lyle_/Desktop/4 Yr Sem 2/DS/Banzai-predicament/data/Validation/NotBald'
bald_dir = 'C:/Users/Lyle_/Desktop/4 Yr Sem 2/DS/Validation/Bald'
not_bald_dir = 'C:/Users/Lyle_/Desktop/4 Yr Sem 2/DS/Validation/NotBald'

# Define the image size that the model expects
IMG_SIZE = (64, 64)

# Define the number of neighbors to consider
K = 5

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

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#%%

# Find the optimal k value using cross-validation
k_values = list(range(1, 21))
cv_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='f1')
    cv_scores.append(scores.mean())

optimal_k = k_values[cv_scores.index(max(cv_scores))]
print("Optimal k value:", optimal_k)


# Plot the cross-validation results
plt.plot(k_values, cv_scores)
plt.xlabel('Number of neighbors (k)')
plt.ylabel('F1 score')
plt.title('Cross-validation results')
plt.show()
# Optimal k = 5-6
#%%
# Create the KNN classifier
knn = KNeighborsClassifier(n_neighbors=K)

# Train the classifier
knn.fit(X_train, y_train)

# # Test the classifier
# accuracy = knn.score(X_test, y_test)
# print(f'Test accuracy: {accuracy}')

# Test the classifier
y_pred = knn.predict(X_test)

# Print the confusion matrix and classification report
print(f'Confusion matrix:\n{confusion_matrix(y_test, y_pred)}')
print(f'Classification report:\n{classification_report(y_test, y_pred)}')

