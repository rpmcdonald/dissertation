import tensorflow as tf
from tensorflow import keras
from tensorflow import Tensor
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Add, ReLU, Input, AveragePooling2D
import numpy as np
import matplotlib.pyplot as plt

mfcc_shape = 39
length = 192
classes = 2

def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

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

def create_res_net():
    inputs = Input(shape=(mfcc_shape, length, 1))
    num_filters = 8
    
    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=(3, 3),
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)
    
    num_blocks_list = [2, 4, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
        num_filters *= 2
    
    t = AveragePooling2D(4)(t)
    t = Flatten()(t)
    outputs = Dense(classes, activation='softmax')(t)
    
    model = Model(inputs, outputs)

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adagrad(lr=0.01),
        metrics=['accuracy']
    )

    return model

model = create_res_net()

history = model.fit(X_train, y_train_hot, batch_size=128, epochs=100, verbose=1,
            validation_data=(X_val, y_val_hot), callbacks=callbacks)