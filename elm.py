import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.linalg import pinv2

length = 64
X_train = np.load('mfccs/X_train_moz.npy').reshape(-1, 16, length, 1)
X_test = np.load('mfccs/X_test_moz.npy').reshape(-1, 16, length, 1)
X_val = np.load('mfccs/X_val_moz.npy').reshape(-1, 16, length, 1)
y_train = np.load('mfccs/y_train_moz.npy')
y_test = np.load('mfccs/y_test_moz.npy')
y_val = np.load('mfccs/y_val_moz.npy')

input_size = X_train.shape[1]
hidden_size = 1000

input_weights = np.random.normal(size=[input_size,hidden_size])
biases = np.random.normal(size=[hidden_size])

def relu(x):
   return np.maximum(x, 0, x)

def hidden_nodes(X):
    G = np.dot(X, input_weights)
    G = G + biases
    H = relu(G)
    return H

output_weights = np.dot(pinv2(hidden_nodes(X_train)), y_train)

def predict(X):
    out = hidden_nodes(X)
    out = np.dot(out, output_weights)
    return out

prediction = predict(X_test)
correct = 0
total = X_test.shape[0]


for i in range(total):
    predicted = np.argmax(prediction[i])
    actual = np.argmax(y_test[i])
    correct += 1 if predicted == actual else 0
accuracy = correct/total
print('Accuracy for ', hidden_size, ' hidden nodes: ', accuracy)
