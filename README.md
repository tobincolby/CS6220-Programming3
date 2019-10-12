split.py
- Splits the overall data set into a train and test set

createSVM.py
- Uses GridSearchCV to tune hyperparameters for the various SVM's and saves the models and GridSearch results

createEnsembles.py
- Uses GridSearchCV to tune hyperparameters for the boosted SVM's and saves the models and GridSearch results

visualize.py
- Uses the GridSearch results to generate 3D graphs to show how different combinations of parameters result in different perforance

testSVM.py
- Takes the saved models and runs them against the test data to get their performance

# How to Run

1. python3 split.py to get get a new split of the data
2. python3 createSVM.py to get the best SVM's for the various kernels
3. python3 createEnsembles.py to get the best boosted SVM's for various kernels
4. python3 visualize.py to get the graphs for visualizing results
5. python3 testSVM.py to get the test performance of the SVM's regular/boosted