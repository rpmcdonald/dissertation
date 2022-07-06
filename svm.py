from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

data = "moz"
mfcc_shape = 39
length = 128
n_components = 3
pca = False

if pca:
    X_train = np.load(f'mfccs/X_train_{data}.npy').reshape(-1, n_components)
    X_test = np.load(f'mfccs/X_test_{data}.npy').reshape(-1, n_components)
else:
    X_train = np.load(f'mfccs/X_train_{data}.npy').reshape(-1, mfcc_shape, length)
    X_test = np.load(f'mfccs/X_test_{data}.npy').reshape(-1, mfcc_shape, length)
y_train = np.load(f'mfccs/y_train_{data}.npy')
y_test = np.load(f'mfccs/y_test_{data}.npy')

print(X_train.shape, X_test.shape, len(y_train), len(y_test))

# parameters = [{'C' : [1,10,100,1000], 'kernel' : ['linear']},{'C' : [1,10,100,1000], 'kernel' : ['rbf'], 'gamma' : [0.5,0.1,0.01,0.001]}]
# classifier = SVC(kernel = 'linear', random_state=0,  C=1)
# grid_search = GridSearchCV(estimator=classifier,
# 	param_grid=parameters,
# 	scoring='accuracy',
# 	cv=10,
# 	n_jobs=-1)
# grid_search = grid_search.fit(X_train,y_train)
# best_accuracy = grid_search.best_score_
# print('\nBest Accuracy : \n{}'.format(best_accuracy))
# best_parameters = grid_search.best_params_
# print('\nBest Parameters : \n{}'.format(best_parameters))

model = SVC(kernel = 'rbf', probability=True)

nsamples, nx, ny = X_train.shape
X_train_reshape = X_train.reshape((nsamples,nx*ny))

nsamples, nx, ny = X_test.shape
X_test_reshape = X_test.reshape((nsamples,nx*ny))

model.fit(X_train_reshape, y_train)

y_predict = model.predict(X_test_reshape)
y_test = np.ravel(y_test)

print(f'Model Score: {model.score(X_test_reshape, y_test)}')
#print(f"Best params: {model.best_params_}")
cm = confusion_matrix(y_test, y_predict)
print(f'Confusion Matrix: \n{cm}')