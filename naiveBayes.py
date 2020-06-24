import numpy as np
import csv
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB as MNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tqdm import tqdm

c_index = 0
run_proportion = 0.3
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

data = []
for row in tqdm(features):
	row_data = []
	for i in range(len(row)):
		if i == 0:
			row_data.append(int(row[0][c_index]))
		else:
			row_data.append(float(row[i]))
	data.append(row_data)

data = np.asarray(data)
y, x = np.split(data, (1,), axis=1)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, random_state=2, train_size=0.8)

def NaiveBayes(x_train, x_test, y_train, y_test):
    gnb = GNB()
    clf = gnb.fit(x_train, y_train.ravel())
    score = clf.score(x_test, y_test)
    print("c_index = ",c_index ," 精度为", score)

NaiveBayes(x_train, x_test, y_train, y_test)
'''‘lbfgs’ ‘newton-cg’ ‘sag’ ‘saga’'''