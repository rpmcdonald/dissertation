from xml.etree.ElementInclude import include
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.applications import DenseNet121, DenseNet169

import numpy as np
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

mfcc_shape = 39
length = 256
classes = 2

X_train = np.load("mfccs/X_train_moz.npy").reshape(-1, mfcc_shape, length, 1)
X_test = np.load("mfccs/X_test_moz.npy").reshape(-1, mfcc_shape, length, 1)
X_val = np.load("mfccs/X_val_moz.npy").reshape(-1, mfcc_shape, length, 1)
y_train = np.load("mfccs/y_train_moz.npy")
y_test = np.load("mfccs/y_test_moz.npy")
y_val = np.load("mfccs/y_val_moz.npy")

y_train_hot = to_categorical(y_train, num_classes=classes)
y_test_hot = to_categorical(y_test, num_classes=classes)
y_val_hot = to_categorical(y_val, num_classes=classes)

callbacks = [TensorBoard(log_dir="./logs")]

model = Sequential()
model.add(Conv2D(16, (7, 7), input_shape=(mfcc_shape, length, 1)))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dense(classes*8, activation="relu"))

model.add(Conv2D(16, (1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dense(classes*4, activation="relu"))

model.add(Conv2D(16, (1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dense(classes*2, activation="relu"))

model.add(Conv2D(16, (1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(classes, activation="relu"))
model.add(Activation('softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adagrad(lr=0.01),
              metrics=["accuracy"])

print(X_train.shape, y_train_hot.shape, X_val.shape, y_val_hot.shape)

history = model.fit(X_train, y_train_hot, batch_size=128, epochs=150, verbose=1,
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
