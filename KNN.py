#   Import the class you want to use.
#   Instantiate the "Estimator".
#   "Estimator" is scikit-learn's term for model.
#   "Instaniate" means making an instance.


from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


iris = load_iris()
 
# store the feature matrix (X) and response vector (y)
X = iris.data
y = iris.target


# Split iris data in train and test data
# A random permutation, to split the data randomly
np.random.seed(0)
indices = np.random.permutation(len(X))
iris_X_train = X[indices[:-10]]
iris_y_train = y[indices[:-10]]
iris_X_test  = X[indices[-10:]]
iris_y_test  = y[indices[-10:]]

knn=KNeighborsClassifier(n_neighbors=1)
#knn=instance of the KNeighborsClassifier
#parameters: n_neighbors=1

knn.fit(iris_X_train, iris_y_train) 

print("predicted classification with KNN algorithm: ",knn.predict(iris_X_test))
print("actual classification: ",iris_y_test)



#KNN CLASSIFICATION

# 1.Pick a value for K.
# 2.Search for the K observations in the training data that
#   are nearest to the measuement of unnown iris.

