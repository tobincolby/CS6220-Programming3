import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pickle

car_file = open("data/car_quality_train.csv")
car_data = np.loadtxt(car_file, delimiter=",")
data_X = car_data[:, :-1]
data_Y = car_data[:, -1]


parameters = {'max_depth': np.arange(2, 20), 'min_samples_leaf': np.arange(1, len(data_Y) - 2)}
clf = GridSearchCV(RandomForestClassifier(n_estimators=5, random_state=1),
                   parameters,
                   n_jobs=4,
                   verbose=1)

clf.fit(data_X, data_Y)

best_forest = clf.best_estimator_
data = clf.cv_results_

pickle.dump(best_forest, open("cv_outputs/random_forest.clf", "wb"))
pickle.dump(data, open("cv_outputs/random_forest_data.data", "wb"))


