import tensorflow as tf
from tensorflow import keras
from tensorflow import Tensor
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Add, ReLU, Input, GlobalMaxPool2D, LeakyReLU
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

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
data = "moz_small"
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

#callbacks = [TensorBoard(log_dir='./logs')]
callbacks = [TensorBoard(log_dir='../../../../../../eddie/scratch/s1126843/logs')]

def relu_bn(inputs: Tensor) -> Tensor:
    bn = BatchNormalization()(inputs)
    relu = ReLU()(bn)
    return relu

def leaky_relu_bn(inputs: Tensor) -> Tensor:
    bn = BatchNormalization()(inputs)
    relu = LeakyReLU()(bn)
    return relu

def residual_stack(input, filters):
    """Convolutional residual stack with weight normalization.

    Args:
        filter: int, determines filter size for the residual stack.

    Returns:
        Residual stack output.
    """
    input_c = Conv2D(filters, 1, dilation_rate=1, padding="same")(input)

    c1 = Conv2D(filters, 3, dilation_rate=1, padding="same")(input)
    lrelu1 = leaky_relu_bn(c1)
    add1 = Add()([lrelu1, input_c])

    c3 = Conv2D(filters, 3, dilation_rate=3, padding="same")(add1)
    lrelu3 = leaky_relu_bn(c3)
    add2 = Add()([add1, lrelu3])

    c5 = Conv2D(filters, 3, dilation_rate=9, padding="same")(add2)
    lrelu5 = leaky_relu_bn(c5)
    add3 = Add()([lrelu5, add2])

    return add3

def create_res_net():
    
    inputs = Input(shape=(mfcc_shape, length, 1))
    num_filters = 32
    
    t = Conv2D(kernel_size=(3, 3),
               strides=1,
               filters=num_filters,
               padding="same")(inputs)
    t = leaky_relu_bn(t)
    #t = Dropout(0.5)(t)
    
    t = residual_stack(t, filters=num_filters)
    # t = Dropout(0.5)(t)
    # t = residual_stack(t, filters=num_filters)
    

    t = GlobalMaxPool2D()(t) # or avg
    #t = Dropout(0.5)(t)
    t = Flatten()(t)
    t = Dense(256)(t)
    t = LeakyReLU()(t)
    t = Dense(128)(t)
    t = LeakyReLU()(t)
    t = Dense(64)(t)
    t = LeakyReLU()(t)
    t = Dense(16)(t)
    t = LeakyReLU()(t)
    t = Dropout(0.5)(t)
    outputs = Dense(classes, activation='softmax')(t)
    
    model = Model(inputs, outputs)

    # model.compile(
    #     optimizer='adam',
    #     loss='sparse_categorical_crossentropy',
    #     metrics=['accuracy']
    # )

    # model.compile(
    #     loss=keras.losses.categorical_crossentropy,
    #     optimizer=keras.optimizers.Adagrad(lr=0.01),
    #     metrics=['accuracy']
    # )

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(lr=0.001),
        metrics=['accuracy']
    )

    return model

model = create_res_net()

print(X_train.shape, y_train_hot.shape, X_val.shape, y_val_hot.shape)
history = model.fit(X_train, y_train_hot, batch_size=64, epochs=50, verbose=1,
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
plt.savefig("cnn_resblock.png")
plt.show()

# what really optimized my model: smaller learning rate, larger number of epochs,
#
# model.save('final_tert_model.h5')
print(model.summary())

# Test
results = model.evaluate(X_test, y_test_hot, batch_size=128)
print("test loss, test acc:", results)

# y_predict = model.predict(X_test)
# y_classes = y_predict.argmax(axis=-1)
# y_test = np.ravel(y_test)
# cm = confusion_matrix(y_test, y_classes)
# print(f'Confusion Matrix: \n{cm}')