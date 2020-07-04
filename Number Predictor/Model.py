# Dependencies for our CNN 
import tensorflow as tf
import numpy as np
# Load in the MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape our data into 4 dimensions because the CNN requires it
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
# Normalize our values from [0, 255] to [0, 1]
X_train = X_train / 255
X_test = X_test / 255
# This converts the class vectors to binary matrices. This means that for class 1 which would represent the number '1', it would be represented as [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Define our model
model = tf.keras.models.Sequential()

#First convolution and maxpooling layer
model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

#Second convolution and maxpooling layer
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Dropout layer to help convergence :)
model.add(tf.keras.layers.Dropout(0.25))
# Flatten our multi dimensional matrix to 1-D to be inputted into the DNN
model.add(tf.keras.layers.Flatten())
# Define the layers and nodes for the DNN
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.5)) 
# Define the output layer
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Define the loss, optimizer and metrics for the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Trains our model for 10 epochs
model.fit(X_train, y_train, epochs=10, shuffle=True)
model.save("digit_recognition.model")
# The model had 99.2% accuracy so I got super lucky with my training
