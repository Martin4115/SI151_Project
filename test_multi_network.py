import numpy as np
import csv
import sklearn
from sklearn import svm
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split

c_index = 3
run_proportion = 1  # To accelerate the computation
total_size = 8675

features = []
i = 0

with open('./feature_dim200_sg1.csv', 'r', encoding='UTF-8') as f:
	reader = csv.reader(f)
	for row in reader:
		i += 1
		features.append(row)
		if i > run_proportion*total_size:
			break

# print(type(float(features[0][1])))

labels = []
def one_hot(label):
	global labels
	result = []
	for i in range(16):
		result.append(0)
	if label in labels:
		result[labels.index(label)] = 1
		return result
	else:
		labels.append(label)
		result[labels.index(label)] = 1
		return result

x = []
y = []
for row in features:
	row_data = []
	for i in range(len(row)):
		if i == 0:
			y.append(one_hot(int(row[0])))
		else:
			row_data.append(float(row[i]))
	x.append(row_data)
# print(type(data[0][1]))

# data = np.asarray(data)
# y, x = np.split(data, (1,), axis=1)

x = np.asarray(x)
y = np.asarray(y)
# print(y[3456])
# exit()


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, 
	random_state=3, train_size=0.8)

# print(y_test.shape)
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape = (200,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(16, activation='sigmoid'))

model.compile(optimizer='rmsprop',
	loss = 'categorical_crossentropy',
	metrics = ['accuracy'])

history = model.fit(x_train, y_train,
	epochs = 6, batch_size = 256)

result = model.evaluate(x_test, y_test)
print(result)