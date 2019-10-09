import numpy as np

car_file = open("data/original_car_quality.csv")
car_data = np.loadtxt(car_file, delimiter=",")

np.random.shuffle(car_data)
car_training, car_test = car_data[:int(len(car_data)*0.7),:], car_data[int(len(car_data)*0.7):,:]

np.savetxt("data/car_quality_train.csv", car_training, delimiter=",")
np.savetxt("data/car_quality_test.csv", car_test, delimiter=",")