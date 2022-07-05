from sklearn.svm import SVC

data = "moz"
mfcc_shape = 39
length = 64
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

parameters = [{'C' : [1,10,100,1000], 'kernel' : ['linear']},{'C' : [1,10,100,1000], 'kernel' : ['rbf'], 'gamma' : [0.5,0.1,0.01,0.001]}]
classifier = SVC(kernel = 'linear', random_state=0,  C=1)
grid_search = GridSearchCV(estimator=classifier,
	param_grid=parameters,
	scoring='accuracy',
	cv=10,
	n_jobs=-1)
grid_search = grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
print('\nBest Accuracy : \n{}'.format(best_accuracy))
best_parameters = grid_search.best_params_
print('\nBest Parameters : \n{}'.format(best_parameters))