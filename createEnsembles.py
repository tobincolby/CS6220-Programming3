from sklearn.ensemble import AdaBoostClassifier
import pickle
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.svm import SVC


best_linear = pickle.load(open("cv_outputs/svm_linear.clf", "rb"))
best_rbf = pickle.load(open("cv_outputs/svm_rbf.clf", "rb"))
best_sigmoid = pickle.load(open("cv_outputs/svm_sigmoid.clf", "rb"))
best_poly = pickle.load(open("cv_outputs/svm_poly.clf", "rb"))

car_file = open("data/car_quality_train.csv")
car_data = np.loadtxt(car_file, delimiter=",")
data_X = car_data[:, :-1]
data_Y = car_data[:, -1]

data_X = preprocessing.scale(data_X)


parameters = {'n_estimators': np.arange(2, 75, 2), 'learning_rate': np.arange(0.1, 2., 0.12)}
# clf = model_selection.GridSearchCV(AdaBoostClassifier(best_linear, algorithm="SAMME.R", random_state=1),
#                                    parameters,
#                                    n_jobs=8,
#                                    cv=5,
#                                    verbose=1)
# clf.fit(data_X, data_Y)
#
# linear_clf = clf.best_estimator_
# pickle.dump(linear_clf, open("cv_outputs/svm_linear_boost.clf", "wb"))
# pickle.dump(clf.cv_results_, open("cv_outputs/svm_linear_boost_data.data", "wb"))

best_rbf = SVC(kernel="rbf", C=1.4, gamma=0.41, max_iter=1300)
clf = model_selection.GridSearchCV(AdaBoostClassifier(best_rbf, algorithm="SAMME"),
                                   parameters,
                                   n_jobs=4,
                                   cv=5,
                                   verbose=1)
clf.fit(data_X, data_Y)

rbf_clf = clf.best_estimator_
pickle.dump(rbf_clf, open("cv_outputs/svm_rbf_boost.clf", "wb"))
pickle.dump(clf.cv_results_, open("cv_outputs/svm_rbf_boost_data.data", "wb"))

best_sigmoid = SVC(kernel="sigmoid", C=0.8, gamma=0.11, max_iter=1300)
clf = model_selection.GridSearchCV(AdaBoostClassifier(best_sigmoid, algorithm="SAMME"),
                                   parameters,
                                   n_jobs=4,
                                   cv=5,
                                   verbose=1)
clf.fit(data_X, data_Y)

sigmoid_clf = clf.best_estimator_
pickle.dump(sigmoid_clf, open("cv_outputs/svm_sigmoid_boost.clf", "wb"))
pickle.dump(clf.cv_results_, open("cv_outputs/svm_sigmoid_boost_data.data", "wb"))

# best_poly = SVC(kernel="poly", C=27, degree=3, max_iter=2000, probability=True)
# clf = model_selection.GridSearchCV(AdaBoostClassifier(best_poly, algorithm="SAMME.R", random_state=1),
#                                    parameters,
#                                    n_jobs=8,
#                                    cv=5,
#                                    verbose=1)
# clf.fit(data_X, data_Y)
#
# poly_clf = clf.best_estimator_
#
# pickle.dump(poly_clf, open("cv_outputs/svm_poly_boost.clf", "wb"))
# pickle.dump(clf.cv_results_, open("cv_outputs/svm_poly_boost_data.data", "wb"))
