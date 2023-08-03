# Import the required libraries
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset and split into training and testing sets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the pixel values to range between 0 and 1
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Convert the labels to one-hot encoded format
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Build the neural network model
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  # Flatten the 28x28 image to a 1D vector
model.add(Dense(128, activation='relu'))    # Fully connected layer with 128 units and ReLU activation
model.add(Dense(64, activation='relu'))     # Fully connected layer with 64 units and ReLU activation
model.add(Dense(10, activation='softmax'))  # Output layer with 10 units (one for each digit) and softmax activation

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_accuracy}')

# Save the model for future use
model.save('handwritten_digit_recognition_model.h5')
