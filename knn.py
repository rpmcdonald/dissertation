from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

length = 64
X_train = np.load('mfccs/X_train_moz.npy').reshape(-1, 16, length, 1)
X_test = np.load('mfccs/X_test_moz.npy').reshape(-1, 16, length, 1)
X_val = np.load('mfccs/X_val_moz.npy').reshape(-1, 16, length, 1)
y_train = np.load('mfccs/y_train_moz.npy')
y_test = np.load('mfccs/y_test_moz.npy')
y_val = np.load('mfccs/y_val_moz.npy')


grid_params = {
    'n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
y_train = np.ravel(y_train)
#print(y_train.shape)

nsamples, nx, ny, _ = X_train.shape
X_train_reshape = X_train.reshape((nsamples,nx*ny))

#print(X_train_reshape.shape)
model = GridSearchCV(KNeighborsClassifier(), grid_params, cv=5, n_jobs=-1)
model.fit(X_train_reshape,y_train)

nsamples, nx, ny, _ = X_test.shape
X_test_reshape = X_test.reshape((nsamples,nx*ny))

y_predict = model.predict(X_test_reshape)
y_test = np.ravel(y_test)
print(f'Model Score: {model.score(X_test_reshape, y_test)}')
cm = confusion_matrix(y_predict, y_test)
print(f'Confusion Matrix: \n{cm}')

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()