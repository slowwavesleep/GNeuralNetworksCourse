from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import models, layers
import tensorflow.keras as keras

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255 
X_test /= 255 

y_train = to_categorical(y_train, 10) 
y_test = to_categorical(y_test, 10) 

X_train = X_train.reshape(X_train.shape[0], 28,28,1)
X_test = X_test.reshape(X_test.shape[0], 28,28,1)


model = Sequential()

model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(28,28,1), padding="same"))
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
model.add(layers.Conv2D(24, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
model.add(layers.Conv2D(128, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
model.add(layers.Conv2D(256, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))

model.add(layers.Flatten())

model.add(layers.Dense(84, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])


hist = model.fit(x=X_train, y=y_train, epochs=1, batch_size=128, validation_data=(X_test, y_test), verbose=1)

test_score = model.evaluate(X_test, y_test)
print(f"Test loss: {test_score[0]:.4f}, accuracy: {(test_score[1] * 100):.2f}")