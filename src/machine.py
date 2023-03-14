import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

train_dir = 'path/to/train/directory'
validation_dir = 'path/to/validation/directory'
test_dir = 'path/to/test/directory'

# Define the parameters for the image data generator

batch_size = 16
img_height = 224
img_width = 224

#%%

# We will use the ImageDataGenerator class to preprocess our images and generate batches of augmented data:

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

#%%

# Load our data using the flow_from_directory method:
train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=(img_height, img_width),
                                               batch_size=batch_size,
                                               class_mode='binary')

validation_data = validation_datagen.flow_from_directory(validation_dir,
                                                         target_size=(img_height, img_width),
                                                         batch_size=batch_size,
                                                         class_mode='binary')

test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size=(img_height, img_width),
                                             batch_size=batch_size,
                                             class_mode='binary')

#%%

# Define our model using the Keras API. Here's an example of a simple convolutional neural network (CNN) architecture:

model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

#%%

# Compile the model with binary cross-entropy as the loss function and the Adam optimizer:

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#%%

# Train our model using the fit method:

epochs = 10
history = model.fit(train_data, 
                    epochs=epochs, 
                    validation_data=validation_data, 
                    verbose=1)

#%%

# Evaluate our model on the test set:

test_loss, test_acc = model.evaluate(test_data, verbose=1)
print('Test accuracy:', test_acc)
