from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.keras.layers.convolutional import MaxPooling1D


top_words = 3000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)


max_words = 300
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)


model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.001, seed=0))
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=128, verbose=2)

scores = model.evaluate(X_test, y_test, verbose=0)
print(scores[1])