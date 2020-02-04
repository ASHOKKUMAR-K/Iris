# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import pickle

# Loading the Data
iris = sb.load_dataset('iris')


# Summarizing the Data

iris.groupby('species').size()


# ## Classification


#from sklearn.xgboost import XGBClassifier
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix



# Selecting Dependent and Independent Variables
X = iris.iloc[:, :-1]
y = iris.iloc[:, -1]


# Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 45)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


classifiers = []

model1 = svm.SVC()
classifiers.append(model1)

model2 = tree.DecisionTreeClassifier()
classifiers.append(model2)

model3 = RandomForestClassifier()
classifiers.append(model3)

model1.fit(X_train, y_train)

model1.predict([[4.5, 3.3, 1.25, 0.2]])

pickle.dump(model1, open("classifier.pkl", "wb"))

