import numpy as np
import pickle
from sklearn import preprocessing

car_file = open("data/car_quality_test.csv")
car_data = np.loadtxt(car_file, delimiter=",")
test_X = car_data[:, :-1]
test_Y = car_data[:, -1]
test_X = preprocessing.scale(test_X)


best_linear = pickle.load(open("cv_outputs/svm_linear.clf", "rb"))
best_rbf = pickle.load(open("cv_outputs/svm_rbf.clf", "rb"))
best_sigmoid = pickle.load(open("cv_outputs/svm_sigmoid.clf", "rb"))
best_poly = pickle.load(open("cv_outputs/svm_poly.clf", "rb"))
boost_linear = pickle.load(open("cv_outputs/svm_linear_boost.clf", "rb"))
boost_rbf = pickle.load(open("cv_outputs/svm_rbf_boost.clf", "rb"))
boost_sigmoid = pickle.load(open("cv_outputs/svm_sigmoid_boost.clf", "rb"))
boost_poly = pickle.load(open("cv_outputs/svm_poly_boost.clf", "rb"))


test_poly = best_poly.predict(test_X)
test_rbf = best_rbf.predict(test_X)
test_linear = best_linear.predict(test_X)
test_sigmoid = best_sigmoid.predict(test_X)

test_poly_boost = boost_poly.predict(test_X)
test_rbf_boost = boost_rbf.predict(test_X)
test_linear_boost = boost_linear.predict(test_X)
test_sigmoid_boost = boost_sigmoid.predict(test_X)

correct_poly = 0
correct_rbf = 0
correct_sigmoid = 0
correct_linear = 0

correct_poly_boost = 0
correct_rbf_boost = 0
correct_sigmoid_boost = 0
correct_linear_boost = 0
for i in range(len(test_Y)):
    if test_Y[i] == test_poly[i]:
        correct_poly += 1

    if test_Y[i] == test_linear[i]:
        correct_linear += 1

    if test_Y[i] == test_rbf[i]:
        correct_rbf += 1

    if test_Y[i] == test_sigmoid[i]:
        correct_sigmoid += 1

    if test_Y[i] == test_poly_boost[i]:
        correct_poly_boost += 1

    if test_Y[i] == test_linear_boost[i]:
        correct_linear_boost += 1

    if test_Y[i] == test_rbf_boost[i]:
        correct_rbf_boost += 1

    if test_Y[i] == test_sigmoid_boost[i]:
        correct_sigmoid_boost += 1


poly_accuracy = float(correct_poly) / len(test_Y)
sigmoid_accuracy = float(correct_sigmoid) / len(test_Y)
rbf_accuracy = float(correct_rbf) / len(test_Y)
linear_accuracy = float(correct_linear) / len(test_Y)

poly_accuracy_boost = float(correct_poly_boost) / len(test_Y)
sigmoid_accuracy_boost = float(correct_sigmoid_boost) / len(test_Y)
rbf_accuracy_boost = float(correct_rbf_boost) / len(test_Y)
linear_accuracy_boost = float(correct_linear_boost) / len(test_Y)

print(poly_accuracy)
print(sigmoid_accuracy)
print(rbf_accuracy)
print(linear_accuracy)

print(poly_accuracy_boost)
print(sigmoid_accuracy_boost)
print(rbf_accuracy_boost)
print(linear_accuracy_boost)