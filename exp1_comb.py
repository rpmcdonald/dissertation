from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

data = "moz"
#data = "moz_small"
mfcc_shape = 39
length = 192
n_components = 192
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

# parameters = {
#     'n_neighbors': list(range(1, 15)),
#     'weights': ['uniform', 'distance'],
#     'metric': ['euclidean', 'manhattan', 'minkowski'],
# }

parameters_euc = {
    'n_neighbors': [11],
    'weights': ['distance'],
    'metric': ['euclidean'],
}

parameters_man = {
    'n_neighbors': [11],
    'weights': ['distance'],
    'metric': ['manhattan'],
}

model = KNeighborsClassifier(n_neighbors=11, weights="uniform", metric="euclidean")
model.fit(X_train_reshape,y_train)
print(f'KNN Euclidean Model Score: {model.score(X_test_reshape, y_test)}')

model = KNeighborsClassifier(n_neighbors=11, weights="uniform", metric="manhattan")
model.fit(X_train_reshape,y_train)
print(f'KNN Manhattan Model Score: {model.score(X_test_reshape, y_test)}')

# SVM
parameters = {
    'C' : [1], 
    'kernel' : ['rbf'], 
    'gamma' : ["scale"]}

# parameters = {
#     'C' : [0.1,1,10,100,1000], 
#     'kernel' : ['rbf'], 
#     'gamma' : ["scale"]}

model = SVC(kernel = 'rbf', gamma="scale",  C=1)
model.fit(X_train_reshape,y_train)
print(f'SVM Model Score: {model.score(X_test_reshape, y_test)}')

# RFC
parameters_gini = { 
    'n_estimators': [300],
    'max_features': ['sqrt'],
    'criterion' :['gini']
}

parameters_ent = { 
    'n_estimators': [300],
    'max_features': ['sqrt'],
    'criterion' :['entropy']
}

model = RandomForestClassifier(n_estimators=300, max_features="sqrt", criterion='gini')
model.fit(X_train_reshape, y_train)
print(f'RFC Gini Model Score: {model.score(X_test_reshape, y_test)}')

model = RandomForestClassifier(n_estimators=300, max_features="sqrt", criterion='entropy')
model.fit(X_train_reshape, y_train)
print(f'RFC Entropy Model Score: {model.score(X_test_reshape, y_test)}')