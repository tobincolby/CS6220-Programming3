import numpy as np
import pickle
from sklearn import preprocessing
from datetime import datetime
import csv

car_file = open("data/car_quality_test.csv")
car_data = np.loadtxt(car_file, delimiter=",")
test_X = car_data[:, :-1]
test_Y = car_data[:, -1]
test_X = preprocessing.scale(test_X)

car_file = open("data/car_quality_train.csv")
car_data = np.loadtxt(car_file, delimiter=",")
data_X = car_data[:, :-1]
data_Y = car_data[:, -1]
data_X = preprocessing.scale(data_X)



best_linear = pickle.load(open("cv_outputs/svm_linear.clf", "rb"))
best_rbf = pickle.load(open("cv_outputs/svm_rbf.clf", "rb"))
best_sigmoid = pickle.load(open("cv_outputs/svm_sigmoid.clf", "rb"))
best_poly = pickle.load(open("cv_outputs/svm_poly.clf", "rb"))
boost_linear = pickle.load(open("cv_outputs/svm_linear_boost.clf", "rb"))
boost_rbf = pickle.load(open("cv_outputs/svm_rbf_boost.clf", "rb"))
boost_sigmoid = pickle.load(open("cv_outputs/svm_sigmoid_boost.clf", "rb"))
boost_poly = pickle.load(open("cv_outputs/svm_poly_boost.clf", "rb"))

random_forest = pickle.load(open("cv_outputs/random_forest.clf", "rb"))

best_linear.fit(data_X, data_Y)
best_rbf.fit(data_X, data_Y)
best_sigmoid.fit(data_X, data_Y)
best_poly.fit(data_X, data_Y)
boost_linear.fit(data_X, data_Y)
boost_sigmoid.fit(data_X, data_Y)
boost_rbf.fit(data_X, data_Y)
boost_poly.fit(data_X, data_Y)
random_forest.fit(data_X, data_Y)



starttime = datetime.now()
test_poly = best_poly.predict(test_X)
poly_time = (datetime.now() - starttime).total_seconds()

starttime = datetime.now()
test_rbf = best_rbf.predict(test_X)
rbf_time = (datetime.now() - starttime).total_seconds()

starttime = datetime.now()
test_linear = best_linear.predict(test_X)
linear_time = (datetime.now() - starttime).total_seconds()

starttime = datetime.now()
test_sigmoid = best_sigmoid.predict(test_X)
sigmoid_time = (datetime.now() - starttime).total_seconds()



starttime = datetime.now()
test_poly_boost = boost_poly.predict(test_X)
poly_boost_time = (datetime.now() - starttime).total_seconds()

starttime = datetime.now()
test_rbf_boost = boost_rbf.predict(test_X)
rbf_boost_time = (datetime.now() - starttime).total_seconds()

starttime = datetime.now()
test_linear_boost = boost_linear.predict(test_X)
linear_boost_time = (datetime.now() - starttime).total_seconds()

starttime = datetime.now()
test_sigmoid_boost = boost_sigmoid.predict(test_X)
sigmoid_boost_time = (datetime.now() - starttime).total_seconds()

starttime = datetime.now()
test_random_forest = random_forest.predict(test_X)
random_forest_time = (datetime.now() - starttime).total_seconds()

correct_poly = 0
correct_rbf = 0
correct_sigmoid = 0
correct_linear = 0

correct_poly_boost = 0
correct_rbf_boost = 0
correct_sigmoid_boost = 0
correct_linear_boost = 0

correct_random_forest = 0

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

    if test_Y[i] == test_random_forest[i]:
        correct_random_forest += 1


poly_accuracy = float(correct_poly) / len(test_Y)
sigmoid_accuracy = float(correct_sigmoid) / len(test_Y)
rbf_accuracy = float(correct_rbf) / len(test_Y)
linear_accuracy = float(correct_linear) / len(test_Y)

poly_accuracy_boost = float(correct_poly_boost) / len(test_Y)
sigmoid_accuracy_boost = float(correct_sigmoid_boost) / len(test_Y)
rbf_accuracy_boost = float(correct_rbf_boost) / len(test_Y)
linear_accuracy_boost = float(correct_linear_boost) / len(test_Y)

random_forest_accuracy = float(correct_random_forest) / len(test_Y)

write_dict = dict()

write_dict["poly_accuracy"] = poly_accuracy
write_dict["sigmoid_accuracy"] = sigmoid_accuracy
write_dict["rbf_accuracy"] = rbf_accuracy
write_dict["linear_accuracy"] = linear_accuracy

write_dict["poly_boost_accuracy"] = poly_accuracy_boost
write_dict["sigmoid_boost_accuracy"] = sigmoid_accuracy_boost
write_dict["rbf_boost_accuracy"] = rbf_accuracy_boost
write_dict["linear_boost_accuracy"] = linear_accuracy_boost

write_dict["random_forest"] = random_forest_accuracy

writer = csv.writer(open("test_accuracy.csv", "w"))
for key, value in write_dict.items():
    writer.writerow([key, value])

print(poly_accuracy)
print(sigmoid_accuracy)
print(rbf_accuracy)
print(linear_accuracy)

print(poly_accuracy_boost)
print(sigmoid_accuracy_boost)
print(rbf_accuracy_boost)
print(linear_accuracy_boost)

print(random_forest_accuracy)

print(poly_time)
print(rbf_time)
print(linear_time)
print(sigmoid_time)

print(poly_boost_time)
print(rbf_boost_time)
print(linear_boost_time)
print(sigmoid_boost_time)

print(random_forest_time)