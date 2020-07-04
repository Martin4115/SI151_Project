import numpy as np
import csv
import sklearn
from sklearn import svm
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split

from keras.layers import LSTM

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

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, 
	random_state=3, train_size=0.8)
x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x_train, y_train,
	random_state=1, train_size=0.7)

# print(y_test.shape)
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape = (200,)))
model.add(layers.Dense(LSTM(32)))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
	loss = 'binary_crossentropy',
	metrics = ['accuracy'])

history = model.fit(x_train, y_train,
	epochs = 20, batch_size = 256, validation_data=(x_val, y_val))

import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss_values)+1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label = 'validation loss')
plt.title("Training and validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# print(history_dict.keys())

plt.clf()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label = 'validation accuracy')
plt.title("Training and validation accuracy")
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
