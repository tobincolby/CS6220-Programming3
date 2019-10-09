from sklearn import model_selection
from sklearn.svm import SVC
import numpy as np
from sklearn import preprocessing
import pickle

car_file = open("data/car_quality_train.csv")
car_data = np.loadtxt(car_file, delimiter=",")
data_X = car_data[:, :-1]
data_Y = car_data[:, -1]

data_X = preprocessing.scale(data_X)

parameters = {'C': np.arange(1,100), 'degree': np.arange(1,10)}
clf = model_selection.GridSearchCV(SVC(kernel="poly", random_state=1, decision_function_shape='ovo'),
                                    parameters,
                                    cv=5,
                                    verbose=1,
                                    n_jobs=4)

clf.fit(data_X, data_Y)

best_poly = clf.best_estimator_

pickle.dump(clf.cv_results_, open("cv_outputs/poly_data.data", "wb"))

parameters = {'C': np.arange(0.1, 1.5, 0.1), 'gamma': np.arange(0.01, 2, 0.1)}
clf = model_selection.GridSearchCV(SVC(kernel="rbf", random_state=1, decision_function_shape='ovo'),
                                    parameters,
                                    cv=5,
                                    verbose=1,
                                    n_jobs=4)
clf.fit(data_X, data_Y)

best_rbf = clf.best_estimator_

pickle.dump(clf.cv_results_, open("cv_outputs/rbf_data.data", "wb"))

parameters = {'C': np.arange(1, 100)}
clf = model_selection.GridSearchCV(SVC(kernel="linear", random_state=1, decision_function_shape='ovo'),
                                    parameters,
                                    cv=5,
                                    verbose=1,
                                    n_jobs=4)
clf.fit(data_X, data_Y)

best_linear = clf.best_estimator_

pickle.dump(clf.cv_results_, open("cv_outputs/linear_data.data", "wb"))

parameters = {'C': np.arange(0.1, 1.5, 0.1), 'gamma': np.arange(0.01, 2, 0.1)}
clf = model_selection.GridSearchCV(SVC(kernel="sigmoid", random_state=1, decision_function_shape='ovo'),
                                    parameters,
                                    cv=5,
                                    verbose=1,
                                    n_jobs=4)
clf.fit(data_X, data_Y)

best_sigmoid = clf.best_estimator_

pickle.dump(clf.cv_results_, open("cv_outputs/sigmoid_data.data", "wb"))

print(best_poly.C, best_poly.degree)
print(best_sigmoid.C, best_sigmoid.gamma)
print(best_linear.C)
print(best_rbf.C, best_rbf.gamma)


pickle.dump(best_linear, open("cv_outputs/svm_linear.clf", "wb"))
pickle.dump(best_rbf, open("cv_outputs/svm_rbf.clf", "wb"))
pickle.dump(best_sigmoid, open("cv_outputs/svm_sigmoid.clf", "wb"))
pickle.dump(best_poly, open("cv_outputs/svm_poly.clf", "wb"))


