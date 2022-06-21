from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import numpy as np

length = 64
X_train = np.load('mfccs/X_train_moz.npy').reshape(-1, 16, length, 1)
X_test = np.load('mfccs/X_test_moz.npy').reshape(-1, 16, length, 1)
X_val = np.load('mfccs/X_val_moz.npy').reshape(-1, 16, length, 1)
y_train = np.load('mfccs/y_train_moz.npy')
y_test = np.load('mfccs/y_test_moz.npy')
y_val = np.load('mfccs/y_val_moz.npy')

#model = KNeighborsClassifier(n_neighbors=3)

grid_params = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
print(y_train.shape)
print(y_train)
nsamples, nx, ny, _ = X_train.shape
print(nsamples)
d2_train_dataset = X_train.reshape((nsamples,nx*ny))
print(d2_train_dataset.shape)
model = GridSearchCV(KNeighborsClassifier(), grid_params, cv=5, n_jobs=-1)
#model.fit(d2_train_dataset,y_train)