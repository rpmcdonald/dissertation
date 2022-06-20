from keras.models import load_model
import numpy as np
from keras.callbacks import TensorBoard
from keras.callbacks import Callback
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display
import sklearn.preprocessing
import pydub
from keras.utils import to_categorical
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D


length = 64
X_train = np.load('../experiments_data/mozilla/mfccs/X_train_moz.npy').reshape(-1, 16, length, 1)
X_test = np.load('../experiments_data/mozilla/mfccs/X_test_moz.npy').reshape(-1, 16, length, 1)
X_val = np.load('../experiments_data/mozilla/mfccs/X_val_moz.npy').reshape(-1, 16, length, 1)
y_train = np.load('../experiments_data/mozilla/mfccs/y_train_moz.npy')
y_test = np.load('../experiments_data/mozilla/mfccs/y_test_moz.npy')
y_val = np.load('../experiments_data/mozilla/mfccs/y_val_moz.npy')


callbacks = [TensorBoard(log_dir='./logs')]

model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(16, length, 1)))
model.add(Activation('relu'))
model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.15))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
# the model so far outputs 3D feature maps (height, width, features)

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=128, epochs=1000, verbose=1,
            validation_data=(X_val, y_val), callbacks=callbacks)


training_loss = history.history['loss']
test_loss = history.history['val_loss']

training_acc = history.history['acc']
test_acc = history.history['val_acc']

# Create count of the number of epochs
epoch_count = range(1, len(training_acc) + 1)

# Visualize loss history


plt.subplots(1, 2)
plt.subplot(1, 2, 1)
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(epoch_count, training_acc, 'r--')
plt.plot(epoch_count, test_acc, 'b-')
plt.legend(['Training Accuracy', 'Test Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.savefig('binary_plots_newest')
plt.show()

# what really optimized my model: smaller learning rate, larger number of epochs,
#
model.save('final_binary_model.h5')
print(model.summary())
