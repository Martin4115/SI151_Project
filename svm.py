import numpy as np
import csv
import sklearn
from sklearn import svm
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split

c_index = 0
run_proportion = 0.3  # To accelerate the computation
total_size = 8675

features = []
i = 0

with open('./feature_dim200_sg0.csv', 'r', encoding='UTF-8') as f:
	reader = csv.reader(f)
	for row in reader:
		i += 1
		features.append(row)
		if i > run_proportion*total_size:
			break

# print(type(float(features[0][1])))

data = []
for row in features:
	row_data = []
	for i in range(len(row)):
		if i == 0:
			row_data.append(int(row[0][c_index]))
		else:
			row_data.append(float(row[i]))
	data.append(row_data)
# print(type(data[0][1]))

data = np.asarray(data)
y, x = np.split(data, (1,), axis=1)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, random_state=2, train_size=0.8)

'''
clf = svm.SVC(C=1, kernel='rbf', gamma=30, decision_function_shape='ovo')
clf.fit(x_train, y_train.ravel())
print(clf.score(x_train, y_train))
print(clf.score(x_test, y_test))'''

def svm_c(x_train, x_test, y_train, y_test):
	svc = SVC(kernel='rbf', class_weight='balanced',)
	c_range = np.logspace(-5, 15, 11, base=2)
	gamma_range = np.logspace(-9, 3, 13, base=2)
	param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
	grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
	clf = grid.fit(x_train, y_train.ravel())
	score = grid.score(x_test, y_test)
	print("精度为", score)

svm_c(x_train, x_test, y_train, y_test)

# print(data)