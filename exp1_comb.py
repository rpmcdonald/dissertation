from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

data = "moz"
mfcc_shape = 39
length = 128
n_components = 8
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

if not pca:
    nsamples, nx, ny = X_train.shape
    X_train_reshape = X_train.reshape((nsamples,nx*ny))
    nsamples, nx, ny = X_test.shape
    X_test_reshape = X_test.reshape((nsamples,nx*ny))
else:
    X_train_reshape = X_train
    X_test_reshape = X_test

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# KNN

parameters = {
    'n_neighbors': list(range(1, 15)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
}

model = GridSearchCV(KNeighborsClassifier(), parameters, cv=5, n_jobs=-1, verbose=1)
model.fit(X_train_reshape,y_train)
y_predict = model.predict(X_test_reshape)
print(f'KNN Model Score: {model.score(X_test_reshape, y_test)}')

# SVM
parameters = [{'C' : [0.1,1,10,100,1000], 'kernel' : ['rbf'], 'gamma' : [0.5,0.1,0.01,0.001]}]
classifier = SVC(kernel = 'rbf', random_state=0,  C=1)

model = GridSearchCV(estimator=classifier,
    param_grid=parameters,
    scoring='accuracy',
    cv=5,
    n_jobs=-1, 
    verbose=1)

grid_search = model.fit(X_train_reshape,y_train)
y_predict = model.predict(X_test_reshape)
print(f'SVM Model Score: {model.score(X_test_reshape, y_test)}')

# RFC
parameters = { 
    'n_estimators': [100, 200, 500],
    'max_features': ['sqrt', 'log2'],
    #'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
classifier = RandomForestClassifier()

model = GridSearchCV(estimator=classifier,
	param_grid=parameters,
	scoring='accuracy',
	cv=5,
	n_jobs=-1, 
    verbose=1)

model.fit(X_train_reshape, y_train)
y_predict = model.predict(X_test_reshape)
print(f'RFC Model Score: {model.score(X_test_reshape, y_test)}')