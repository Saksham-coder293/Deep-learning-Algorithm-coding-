import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

#predicts true or false for these 2 flowers 
iris = load_iris(as_frame = True)

X = iris.data[["Petal lenght cm", "Sepal lenght cm"]]
y = (iris.target == 0)



per_clf = Perceptron()
per_clf.fit(X,y)


X_new = [[1,2], [5,8]]
y_pred = per_clf.predict(X_new)   #predict true and false for these 2 flowers 