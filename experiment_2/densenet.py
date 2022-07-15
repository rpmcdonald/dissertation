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

# model = load_model("../models/final_model_3.h5")

data = "moz"
spectrogram_shape = 128
length = 256
classes = 2

# X_train = np.load(f"spectrograms/X_train_{data}.npy").reshape(-1, spectrogram_shape, length, 1)
# X_test = np.load(f"spectrograms/X_test_{data}.npy").reshape(-1, spectrogram_shape, length, 1)
# X_val = np.load(f"spectrograms/X_val_{data}.npy").reshape(-1, spectrogram_shape, length, 1)
# y_train = np.load(f"spectrograms/y_train_{data}.npy")
# y_test = np.load(f"spectrograms/y_test_{data}.npy")
# y_val = np.load(f"spectrograms/y_val_{data}.npy")

X_train = np.load("spectrograms/X_train_moz.npy").reshape(-1, spectrogram_shape, length, 1)
X_test = np.load("spectrograms/X_test_moz.npy").reshape(-1, spectrogram_shape, length, 1)
X_val = np.load("spectrograms/X_val_moz.npy").reshape(-1, spectrogram_shape, length, 1)
y_train = np.load("spectrograms/y_train_moz.npy")
y_test = np.load("spectrograms/y_test_moz.npy")
y_val = np.load("spectrograms/y_val_moz.npy")

print(X_train.shape, X_test.shape, X_val.shape, len(y_train), len(y_test), len(y_val))

y_train_hot = to_categorical(y_train, num_classes=classes)
y_test_hot = to_categorical(y_test, num_classes=classes)
y_val_hot = to_categorical(y_val, num_classes=classes)

callbacks = [TensorBoard(log_dir="./logs")]

model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=(spectrogram_shape, length, 1)))
model.add(Activation("relu"))
model.add(Conv2D(16, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.5))

model.add(Conv2D(16, (3, 3)))
model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.15))

model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# the model so far outputs 3D feature maps (height, width, features)

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(32))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(classes))
model.add(Activation("softmax"))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adagrad(lr=0.01),
              metrics=["accuracy"])

print(X_train.shape, y_train_hot.shape, X_val.shape, y_val_hot.shape)

history = model.fit(X_train, y_train_hot, batch_size=128, epochs=50, verbose=1,
            validation_data=(X_val, y_val_hot), callbacks=callbacks)


training_loss = history.history["loss"]
test_loss = history.history["val_loss"]

training_acc = history.history["accuracy"]
test_acc = history.history["val_accuracy"]

# Create count of the number of epochs
epoch_count = range(1, len(training_acc) + 1)

# Visualize loss history

plt.subplots(1, 2)
plt.subplot(1, 2, 1)
plt.plot(epoch_count, training_loss, "r--")
plt.plot(epoch_count, test_loss, "b-")
plt.legend(["Training Loss", "Test Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(epoch_count, training_acc, "r--")
plt.plot(epoch_count, test_acc, "b-")
plt.legend(["Training Accuracy", "Test Accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.savefig("three_class_plots_DN.png")
plt.show()

# what really optimized my model: smaller learning rate, larger number of epochs,
#
# model.save("final_tert_model.h5")
print(model.summary())
