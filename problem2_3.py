import numpy as np
import sys
import csv
from sklearn import preprocessing

X = []
Y = []

with open(sys.argv[1]) as input2:
    for row in csv.reader(input2):
        X.append([row[0], row[1]])
        Y.append(row[2])
input2.close()

X = np.array(X).astype(np.float64)
Y = np.array(Y).astype(np.float64)
X_scaling = preprocessing.scale(X)
X_scaled = np.hstack((np.ones((X.shape[0],1)), X_scaling))
A = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.09]
O = []

for a in A:
    B = np.zeros(3)
    for i in range(100):
        error = np.dot(X_scaled, B) - Y
        for d in range(X_scaled.shape[1]):
            B[d] -= (a/X_scaled.shape[0]) * np.sum(error * X_scaled[:, d])
    o = [a, 100] + list(B.copy())
    O.append(o)

with open(sys.argv[2], 'w') as output2:
    for row in O:
        csv.writer(output2).writerow(row)
output2.close()
