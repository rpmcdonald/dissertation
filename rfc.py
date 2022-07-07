from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

data = "moz"
mfcc_shape = 39
length = 128
n_components = 32
pca = True

if pca:
    X_train = np.load(f'mfccs/X_train_{data}.npy').reshape(-1, n_components)
    X_test = np.load(f'mfccs/X_test_{data}.npy').reshape(-1, n_components)
else:
    X_train = np.load(f'mfccs/X_train_{data}.npy').reshape(-1, mfcc_shape, length)
    X_test = np.load(f'mfccs/X_test_{data}.npy').reshape(-1, mfcc_shape, length)
y_train = np.load(f'mfccs/y_train_{data}.npy')
y_test = np.load(f'mfccs/y_test_{data}.npy')

# Initialise and fit
model = RandomForestClassifier(n_estimators = 150)

if not pca:
    nsamples, nx, ny = X_train.shape
    X_train_reshape = X_train.reshape((nsamples,nx*ny))
else:
    X_train_reshape = X_train

y_train = np.ravel(y_train)

model.fit(X_train_reshape, y_train)

# Predict
if not pca:
    nsamples, nx, ny = X_test.shape
    X_test_reshape = X_test.reshape((nsamples,nx*ny))
else:
    X_test_reshape = X_test

y_predict = model.predict(X_test_reshape)
y_test = np.ravel(y_test)
print(f'Model Score: {model.score(X_test_reshape, y_test)}')
cm = confusion_matrix(y_test, y_predict)
print(f'Confusion Matrix: \n{cm}')

