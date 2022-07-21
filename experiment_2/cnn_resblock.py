import tensorflow as tf
from tensorflow import keras
from tensorflow import Tensor
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Add, ReLU, Input, AveragePooling2D, LeakyReLU

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

def resblock(x, kernelsize, filters):
    fx = Conv2D(filters, kernelsize, activation='relu', padding='same')(x)
    fx = BatchNormalization()(fx)
    fx = Conv2D(filters, kernelsize, padding='same')(fx)
    out = Add()([x,fx])
    out = ReLU()(out)
    out = BatchNormalization()(out)
    return out

def relu_bn(inputs: Tensor) -> Tensor:
    bn = BatchNormalization()(inputs)
    relu = ReLU()(bn)
    return relu

def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = (3, 3)) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def residual_stack(input, filters):
    """Convolutional residual stack with weight normalization.

    Args:
        filter: int, determines filter size for the residual stack.

    Returns:
        Residual stack output.
    """
    input_c = Conv2D(filters, 1, dilation_rate=1, padding="same")(input)

    c1 = Conv2D(filters, 3, dilation_rate=1, padding="same")(input)
    lrelu1 = LeakyReLU()(c1)
    c2 = Conv2D(filters, 3, dilation_rate=1, padding="same")(lrelu1)
    add1 = Add()([c2, input_c])

    lrelu2 = LeakyReLU()(add1)
    c3 = Conv2D(filters, 3, dilation_rate=3, padding="same")(lrelu2)
    lrelu3 = LeakyReLU()(c3)
    c4 = Conv2D(filters, 3, dilation_rate=1, padding="same")(lrelu3)
    add2 = Add()([add1, c4])

    lrelu4 = LeakyReLU()(add2)
    c5 = Conv2D(filters, 3, dilation_rate=9, padding="same")(lrelu4)
    lrelu5 = LeakyReLU()(c5)
    c6 = Conv2D(filters, 3, dilation_rate=1, padding="same")(lrelu5)
    add3 = Add()([c6, add2])

    return add3

def create_res_net():
    
    inputs = Input(shape=(mfcc_shape, length, 1))
    num_filters = 8
    
    #t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=(3, 3),
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)
    
    #num_blocks_list = [2, 5, 5, 2]
    #num_blocks_list = [2, 4, 2]
    num_blocks_list = [1]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_stack(t, filters=num_filters)
        num_filters *= 2
    
    t = AveragePooling2D(4)(t)
    #t = MaxPooling2D(pool_size=(3, 3))(t)
    t = Flatten()(t)
    outputs = Dense(classes, activation='softmax')(t)
    
    model = Model(inputs, outputs)

    # model.compile(
    #     optimizer='adam',
    #     loss='sparse_categorical_crossentropy',
    #     metrics=['accuracy']
    # )

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adagrad(lr=0.01),
        metrics=['accuracy']
    )

    return model

model = create_res_net()

print(X_train.shape, y_train_hot.shape, X_val.shape, y_val_hot.shape)
print(model.summary())
history = model.fit(X_train, y_train_hot, batch_size=64, epochs=10, verbose=1,
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
