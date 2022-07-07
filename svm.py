from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

data = "moz"
mfcc_shape = 39
length = 32
n_components = 10
pca = True

if pca:
    X_train = np.load(f'mfccs/X_train_{data}.npy').reshape(-1, n_components)
    X_test = np.load(f'mfccs/X_test_{data}.npy').reshape(-1, n_components)
else:
    X_train = np.load(f'mfccs/X_train_{data}.npy').reshape(-1, mfcc_shape, length)
    X_test = np.load(f'mfccs/X_test_{data}.npy').reshape(-1, mfcc_shape, length)
y_train = np.load(f'mfccs/y_train_{data}.npy')
y_test = np.load(f'mfccs/y_test_{data}.npy')

print(X_train.shape, X_test.shape, len(y_train), len(y_test))

# Set up model parameters
#parameters = [{'C' : [0.1,1,10,100,1000], 'kernel' : ['linear']},{'C' : [0.1,1,10,100,1000], 'kernel' : ['rbf'], 'gamma' : [0.5,0.1,0.01,0.001]}]
parameters = [{'C' : [0.1,1,10,100,1000], 'kernel' : ['rbf'], 'gamma' : [0.5,0.1,0.01,0.001]}]
classifier = SVC(kernel = 'rbf', random_state=0,  C=1)

model = GridSearchCV(estimator=classifier,
	param_grid=parameters,
	scoring='accuracy',
	cv=5,
	n_jobs=-1, 
    verbose=2)

if not pca:
    nsamples, nx, ny = X_train.shape
    X_train_reshape = X_train.reshape((nsamples,nx*ny))
else:
    X_train_reshape = X_train

y_train = np.ravel(y_train)

# Find optimal model
grid_search = model.fit(X_train_reshape,y_train)

if not pca:
    nsamples, nx, ny = X_test.shape
    X_test_reshape = X_test.reshape((nsamples,nx*ny))
else:
    X_test_reshape = X_test


y_predict = model.predict(X_test_reshape)
y_test = np.ravel(y_test)
print(f'Model Score: {model.score(X_test_reshape, y_test)}')
print(f"Best params: {model.best_params_}")
cm = confusion_matrix(y_test, y_predict)
print(f'Confusion Matrix: \n{cm}')
