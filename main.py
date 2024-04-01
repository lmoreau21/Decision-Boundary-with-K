# Importing necessary libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# colors
scatter_colors = ['purple', 'turquoise', 'gold']
background_colors = ['#CBC3E3', '#b6dcdd', '#ebd197']

k = 7

# Loading the iris dataset
iris_dataset = datasets.load_iris()
feature_matrix = iris_dataset.data[:, :2] 
target_vector = iris_dataset.target  


knn_classifier = KNeighborsClassifier(n_neighbors=k)

knn_classifier.fit(feature_matrix, target_vector)

mesh_step_size = .02  

feature1_min, feature1_max = feature_matrix[:, 0].min() - 1, feature_matrix[:, 0].max() + 1
feature2_min, feature2_max = feature_matrix[:, 1].min() - 1, feature_matrix[:, 1].max() + 1


mesh_x, mesh_y = np.meshgrid(np.arange(feature1_min, feature1_max, mesh_step_size),
                             np.arange(feature2_min, feature2_max, mesh_step_size))


class_prediction = knn_classifier.predict(np.c_[mesh_x.ravel(), mesh_y.ravel()])
class_prediction = class_prediction.reshape(mesh_x.shape)


plt.figure(figsize=(8, 6))
plt.contourf(mesh_x, mesh_y, class_prediction, cmap=matplotlib.colors.ListedColormap(background_colors))


scatter = plt.scatter(feature_matrix[:, 0], feature_matrix[:, 1], c=target_vector, cmap=matplotlib.colors.ListedColormap(scatter_colors), edgecolor='black', s=50)


class_labels = iris_dataset.target_names
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter_colors[i], markersize=10, markeredgewidth=2, markeredgecolor='black') for i in range(len(class_labels))]


plt.title("3-Class Classification with K-NN (k=7)")
plt.xlabel(iris_dataset.feature_names[0])
plt.ylabel(iris_dataset.feature_names[1])
plt.xlim(mesh_x.min(), mesh_x.max())
plt.ylim(mesh_y.min(), mesh_y.max())
plt.legend(handles, class_labels, title="Classes",  loc='lower right')


plt.show()
