import numpy as np
import csv
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler
#import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


c_index = 0
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
for row in features:
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


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, random_state=2, train_size=0.8)

'''def LogisticRegression(x_train, x_test, y_train, y_test):
    lr = LR(multi_class = 'multinomial',class_weight='balanced',solver = 'lbfgs',max_iter = 5000)
    c_range = np.logspace(-5, 15, 11, base=2)
    param_grid = [{'C': c_range}]
    grid = GridSearchCV(lr, param_grid, cv=3, n_jobs=-1)
    clf = grid.fit(x_train, y_train.ravel())
    score = grid.score(x_test, y_test)
    print("", score)'''

num_round = 2
param = {'max_depth':2, 'eta':1, 'silent':0, 'objective':'binary:logistic' }
def XGBoost(x_train, x_test, y_train, y_test):
    #xgboost = xgb(param, x_train, num_round)
    xgboost = XGBClassifier()
    xgboost.fit(x_train,y_train)

    c_range = np.logspace(-5, 15, 11, base=2)
    param_grid = [{'C': c_range}]
    grid = GridSearchCV(xgboost, param_grid, cv=3, n_jobs=-1)
    clf = grid.fit(x_train, y_train.ravel())
    score = grid.score(x_test, y_test)
    print(" Acc", score)


XGBoost(x_train, x_test, y_train, y_test)