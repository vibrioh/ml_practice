import numpy as np
import sys
import csv

X = []
Y = []

with open(sys.argv[1]) as input1:
    for row in csv.reader(input1):
        X.append([row[0], row[1], 1])
        Y.append(row[2])
input1.close()

X = np.array(X).astype(np.int64)
Y = np.array(Y).astype(np.int64)
w = np.zeros(3).astype(np.int64)
W = []
flag = True

while flag:
    flag = False
    for x, y in zip(X, Y):
        if y != np.where(np.dot(x, w) <= 0, -1, 1):
            flag = True
            w += y * x
    o = w.copy()
    W.append(o)

with open(sys.argv[2], 'w') as output1:
    for row in W:
        csv.writer(output1).writerow(row)
output1.close()
