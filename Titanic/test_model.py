import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

def test_method(method, a_train, a_test, b_train, b_test):
	print method.__class__
	method.fit(a_train, b_train)
	print method.score(a_test, b_test)
	print method.score(a_train, b_train)

def test_methods(methods, a_train, a_test, b_train, b_test):
	for method in methods:
		test_method(method, a_train, a_test, b_train, b_test)

def test_models(data):
	target = data.Survived
	train = data.drop('Survived', axis = 1)
	a_train, a_test, b_train, b_test = train_test_split(train, target, test_size = 0.4, random_state = 42)

	methods = [GaussianNB(), LogisticRegression(), RandomForestClassifier(n_estimators = 100, max_features = 4)]
	test_methods(methods, a_train, a_test, b_train, b_test)

def main():
	data = pd.read_csv('Data/train_processed.csv')
	test_models(data)

if __name__=="__main__":
    main()