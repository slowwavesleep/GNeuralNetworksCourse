import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.datasets import fashion_mnist
from sklearn.metrics import f1_score
from keras.optimizers import SGD

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize the images
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Flatten the images
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

# Build the model
model = Sequential([
  Dense(64, activation='relu', input_shape=(784,)),
  Dense(64, activation='relu'),
  Dropout(0.1, seed=1),
  Dense(64, activation='relu'),
  BatchNormalization(),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax'),
])


# Compile the model
sgd = SGD(lr=0.008, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(
  optimizer=sgd,
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# Train the model.
model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=50,
  batch_size=32,
)

# Evaluate the model
model.evaluate(
  test_images,
  to_categorical(test_labels)
)

# Save the model to disk
# model.save_weights('model.h5')

# Load the model from disk later using:
# model.load_weights('model.h5')

predictions = model.predict(test_images)

print(np.argmax(predictions, axis=1)[:15]) 

print(test_labels[:15]) 

print(f1_score(test_labels, np.argmax(predictions, axis=1), average='weighted'))