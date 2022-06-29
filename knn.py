from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

#data = "openslr83"
data = "ssa"
#data = "moz"
mfcc_shape = 39
length = 1024
n_components = 39

X_train = np.load(f'mfccs/X_train_{data}.npy').reshape(-1, mfcc_shape, n_components)
X_test = np.load(f'mfccs/X_test_{data}.npy').reshape(-1, mfcc_shape, n_components)
y_train = np.load(f'mfccs/y_train_{data}.npy')
y_test = np.load(f'mfccs/y_test_{data}.npy')

print(X_train.shape, X_test.shape, len(y_train), len(y_test))

grid_params = {
    'n_neighbors': list(range(1, 15)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
y_train = np.ravel(y_train)
#print(y_train.shape)

nsamples, nx, ny = X_train.shape
X_train_reshape = X_train.reshape((nsamples,nx*ny))

#print(X_train_reshape.shape)
model = GridSearchCV(KNeighborsClassifier(), grid_params, cv=5, n_jobs=-1, verbose=1)
print(X_train_reshape.shape, y_train.shape)
model.fit(X_train_reshape,y_train)

nsamples, nx, ny = X_test.shape
X_test_reshape = X_test.reshape((nsamples,nx*ny))

y_predict = model.predict(X_test_reshape)
y_test = np.ravel(y_test)
print(f'Model Score: {model.score(X_test_reshape, y_test)}')
print(f"Best params: {model.best_params_}")
cm = confusion_matrix(y_test, y_predict)
print(f'Confusion Matrix: \n{cm}')

if data == "openslr83":
    labels = ["we", "mi", "no", "sc", "so"]
if data == "ssa":
    #labels = ["usa", "ch", "uk", "ind", "can", "kor"]
    #labels = ["usa", "ch"]
    labels = ["sa", "aus", "ch"]
else:
    labels = model.classes_

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.show()