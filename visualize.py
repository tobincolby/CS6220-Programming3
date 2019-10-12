import pickle
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

linear_data = pickle.load(open("cv_outputs/linear_data.data", "rb"))
rbf_data = pickle.load(open("cv_outputs/rbf_data.data", "rb"))
sigmoid_data = pickle.load(open("cv_outputs/sigmoid_data.data", "rb"))
poly_data = pickle.load(open("cv_outputs/poly_data.data", "rb"))

linear_boost_data = pickle.load(open("cv_outputs/svm_linear_boost_data.data", "rb"))
rbf_boost_data = pickle.load(open("cv_outputs/svm_rbf_boost_data.data", "rb"))
sigmoid_boost_data = pickle.load(open("cv_outputs/svm_sigmoid_boost_data.data", "rb"))
poly_boost_data = pickle.load(open("cv_outputs/svm_poly_boost_data.data", "rb"))


random_forest_data = pickle.load(open("cv_outputs/random_forest_data.data", "rb"))
random_forest_graph = pd.DataFrame({'x': [item for item in random_forest_data['param_max_depth']],
                              'y': [item for item in random_forest_data['param_min_samples_leaf']],
                              'z': [1- item for item in random_forest_data['mean_test_score']]})
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(random_forest_graph.x, random_forest_graph.y, random_forest_graph.z, cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Random Forest')
ax.set_xlabel('Max Depth')
ax.set_ylabel('Min Samples Leaf')
ax.set_zlabel('Error')
plt.savefig('car-evaluation-random-forest.png')
plt.show()



linear_boost_graph = pd.DataFrame({'x': [item for item in linear_boost_data['param_learning_rate']],
                              'y': [item for item in linear_boost_data['param_n_estimators']],
                              'z': [1- item for item in linear_boost_data['mean_test_score']]})
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(linear_boost_graph.x, linear_boost_graph.y, linear_boost_graph.z, cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Car Evaluation SVM Boosted (Linear)')
ax.set_xlabel('Learning Rate')
ax.set_ylabel('# Estimators')
ax.set_zlabel('Error')
plt.savefig('car-evaluation-svm-linear-boosted.png')
plt.show()



poly_boost_graph = pd.DataFrame({'x': [item for item in poly_boost_data['param_learning_rate']],
                              'y': [item for item in poly_boost_data['param_n_estimators']],
                              'z': [1- item for item in poly_boost_data['mean_test_score']]})
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(poly_boost_graph.x, poly_boost_graph.y, poly_boost_graph.z, cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Car Evaluation SVM Boosted (Poly)')
ax.set_xlabel('Learning Rate')
ax.set_ylabel('# Estimators')
ax.set_zlabel('Error')
plt.savefig('car-evaluation-svm-poly-boosted.png')
plt.show()

rbf_boost_graph = pd.DataFrame({'x': [item for item in rbf_boost_data['param_learning_rate']],
                              'y': [item for item in rbf_boost_data['param_n_estimators']],
                              'z': [1- item for item in rbf_boost_data['mean_test_score']]})
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(rbf_boost_graph.x, rbf_boost_graph.y, rbf_boost_graph.z, cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Car Evaluation SVM Boosted (RBF)')
ax.set_xlabel('Learning Rate')
ax.set_ylabel('# Estimators')
ax.set_zlabel('Error')
plt.savefig('car-evaluation-svm-rbf-boosted.png')
plt.show()


sigmoid_boost_graph = pd.DataFrame({'x': [item for item in sigmoid_boost_data['param_learning_rate']],
                              'y': [item for item in sigmoid_boost_data['param_n_estimators']],
                              'z': [1- item for item in sigmoid_boost_data['mean_test_score']]})
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(sigmoid_boost_graph.x, sigmoid_boost_graph.y, sigmoid_boost_graph.z, cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Car Evaluation SVM Boosted (Sigmoid)')
ax.set_xlabel('Learning Rate')
ax.set_ylabel('# Estimators')
ax.set_zlabel('Error')
plt.savefig('car-evaluation-svm-sigmoid-boosted.png')
plt.show()




poly_graph = pd.DataFrame({'x': [item for item in poly_data['param_C']],
                              'y': [item for item in poly_data['param_degree']],
                              'z': [1- item for item in poly_data['mean_test_score']]})
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(poly_graph.x, poly_graph.y, poly_graph.z, cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Car Evaluation SVM (Poly)')
ax.set_xlabel('C')
ax.set_ylabel('Degree')
ax.set_zlabel('Error')
plt.savefig('car-evaluation-svm-poly.png')
plt.show()

rbf_graph = pd.DataFrame({'x': [item for item in rbf_data['param_C']],
                              'y': [item for item in rbf_data['param_gamma']],
                              'z': [1- item for item in rbf_data['mean_test_score']]})
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(rbf_graph.x, rbf_graph.y, rbf_graph.z, cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Car Evaluation SVM (RBF)')
ax.set_xlabel('C')
ax.set_ylabel('Gamma')
ax.set_zlabel('Error')
plt.savefig('car-evaluation-svm-rbf.png')
plt.show()

sigmoid_graph = pd.DataFrame({'x': [item for item in sigmoid_data['param_C']],
                              'y': [item for item in sigmoid_data['param_gamma']],
                              'z': [1- item for item in sigmoid_data['mean_test_score']]})
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(sigmoid_graph.x, sigmoid_graph.y, sigmoid_graph.z, cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Car Evaluation SVM (Sigmoid)')
ax.set_xlabel('C')
ax.set_ylabel('Gamma')
ax.set_zlabel('Error')
plt.savefig('car-evaluation-svm-sigmoid.png')
plt.show()

linear_graph = pd.DataFrame({'x': [item for item in linear_data['param_C']],
                              'z': [1- item for item in linear_data['mean_test_score']]})
plt.plot(linear_graph.x, linear_graph.z)

plt.title('Car Evaluation SVM (Linear)')
plt.ylabel('Error')
plt.xlabel('C')
plt.savefig('car-evaluation-svm-linear.png')
plt.show()