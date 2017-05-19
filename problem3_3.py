import numpy as np
import sys
import csv
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

X = []
Y = []

with open(sys.argv[1]) as input3:
    for row in csv.reader(input3):
        X.append([row[0], row[1]])
        Y.append(row[2])
input3.close()
X = preprocessing.scale(np.array(X[1:]).astype(np.float64))
y = np.array(Y[1:]).astype(np.float64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, stratify = y, random_state = 42)

estimators = [SVC(),
              SVC(),
              SVC(),
              LogisticRegression(),
              KNeighborsClassifier(),
              DecisionTreeClassifier(),
              RandomForestClassifier()]
tuned_parameters = [{'kernel': ['linear'], 'C': [0.1, 0.5, 1, 5, 10, 50, 100]},
                    {'kernel': ['poly'], 'C': [0.1, 1., 3.], 'degree': [4, 5, 6], 'gamma': [0.1, 1.]},
                    {'kernel': ['rbf'], 'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'gamma': [0.1, 0.5, 1, 3, 6, 10]},
                    {'C': [0.1, 0.5, 1, 5, 10, 50, 100]},
                    {'n_neighbors': np.linspace(1,50,50).astype(np.int8), 'leaf_size': np.linspace(5,60,12).astype(np.int8)},
                    {'max_depth': np.linspace(1,50,50).astype(np.int8), 'min_samples_split': [2,3,4,5,6,7,8,9,10]},
                    {'max_depth': np.linspace(1,50,50).astype(np.int8), 'min_samples_split': [2,3,4,5,6,7,8,9,10]}]

O = []
col1s = ['svm_linear', 'svm_polynomial', 'svm_rbf', 'logistic', 'knn', 'decision_tree', 'random_forest']
for col1, estimator, tuned_parameter in zip(col1s, estimators, tuned_parameters):
    clf = GridSearchCV(estimator, tuned_parameter, cv = 5, scoring = 'accuracy')
    clf.fit(X_train, y_train)
    o = [col1, clf.best_score_, accuracy_score(y_test, clf.predict(X_test))]
    O.append(o)

with open(sys.argv[2], 'w') as output3:
    for row in O:
        csv.writer(output3).writerow(row)
output3.close()