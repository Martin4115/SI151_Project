import numpy as np
import pandas as pd
import csv
import sklearn
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn import cross_validation
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


# c_index = 0
run_proportion = 1
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


label = []
data = []
for row in tqdm(features):
	row_data = []
	for i in range(len(row)):
		if i == 0:
			row_data.append(int(row[0]))
			if int(row[0]) not in label:
				label.append(int(row[0]))
		else:
			row_data.append(float(row[i]))
	data.append(row_data)

data = np.asarray(data)
y, x = np.split(data, (1,), axis=1)

seed = 7
train_size = 0.8
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, random_state= seed, train_size= train_size)
 
model = xgb.XGBClassifier(max_depth=3,
                      learning_rate=0.3,
                      n_estimators=2000,
                      min_child_weight=5,
                      max_delta_step=0,
                      subsample=0.8,
                      colsample_bytree=0.7,
                      reg_alpha= 0.2 ,
                      reg_lambda=0.5,
                      objective='binary:logistic',
                      missing=None,
                      eval_metric='auc',
                      seed=140,
                      gamma=0)
model.fit( x_train ,  y_train )  
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
# test_score = model.score(x_test, y_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# print('test_score: {0}'.format(test_score))



# num_round = 2
# param = {'max_depth':2, 'eta':1, 'silent':0, 'objective':'binary:logistic' }
# def XGBoost(x_train, x_test, y_train, y_test):
#     xgboost = xgb.train(param, x_train, num_round)
#     # xgboost = XGBClassifier()
#     xgboost.fit(x_train,y_train)

#     c_range = np.logspace(-5, 15, 11, base=2)
#     param_grid = [{'C': c_range}]
#     grid = GridSearchCV(xgboost, param_grid, cv=3, n_jobs=-1)
#     clf = grid.fit(x_train, y_train.ravel())
#     score = grid.score(x_test, y_test)
#     print(" Acc", score)


# XGBoost(x_train, x_test, y_train, y_test)