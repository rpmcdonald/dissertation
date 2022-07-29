import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Add, ReLU, LeakyReLU, GlobalMaxPool2D
from sklearn.metrics import confusion_matrix
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

data = "moz"
#data = "moz_small"
mfcc_shape = 39
length = 192
classes = 2

X_train = np.load(f'mfccs/X_train_{data}.npy').reshape(-1, mfcc_shape, length, 1)
X_test = np.load(f'mfccs/X_test_{data}.npy').reshape(-1, mfcc_shape, length, 1)
X_val = np.load(f'mfccs/X_val_{data}.npy').reshape(-1, mfcc_shape, length, 1)
y_train = np.load(f'mfccs/y_train_{data}.npy')
y_test = np.load(f'mfccs/y_test_{data}.npy')
y_val = np.load(f'mfccs/y_val_{data}.npy')

print(X_train.shape, X_test.shape, X_val.shape, len(y_train), len(y_test), len(y_val))

y_train_hot = to_categorical(y_train, num_classes=classes)
y_test_hot = to_categorical(y_test, num_classes=classes)
y_val_hot = to_categorical(y_val, num_classes=classes)

callbacks = [TensorBoard(log_dir='./logs')]

model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=(mfcc_shape, length, 1)))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Conv2D(16, (3, 3), strides=2))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.5))

model.add(Conv2D(16, (3, 3)))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Conv2D(32, (3, 3), strides=2))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.25))
model.add(GlobalMaxPool2D())

model.add(Flatten())
model.add(Dense(256))
model.add(LeakyReLU())
model.add(Dense(128))
model.add(LeakyReLU())
model.add(Dense(64))
model.add(LeakyReLU())
model.add(Dense(16))
model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Dense(classes))
model.add(Activation('softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adagrad(lr=0.01),
#               metrics=['accuracy'])

print(X_train.shape, y_train_hot.shape, X_val.shape, y_val_hot.shape)

history = model.fit(X_train, y_train_hot, batch_size=128, epochs=500, verbose=1,
            validation_data=(X_val, y_val_hot), callbacks=callbacks)


training_loss = history.history['loss']
test_loss = history.history['val_loss']

training_acc = history.history['accuracy']
test_acc = history.history['val_accuracy']

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

plt.tight_layout()
plt.savefig("cnn.png")
plt.show()

# what really optimized my model: smaller learning rate, larger number of epochs,
#
# model.save('final_tert_model.h5')
print(model.summary())

# Test
results = model.evaluate(X_test, y_test_hot, batch_size=128)
print("test loss, test acc:", results)

y_predict = model.predict(X_test)
y_classes = y_predict.argmax(axis=-1)
y_test = np.ravel(y_test)
cm = confusion_matrix(y_test, y_classes)
print(f'Confusion Matrix: \n{cm}')