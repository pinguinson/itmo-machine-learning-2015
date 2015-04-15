import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

def test_model(data):
	target = data.Survived
	train = data.drop('Survived', axis = 1)
	a_train, a_test, b_train, b_test = train_test_split(train, target, test_size=0.25, random_state = 42)
	method = LogisticRegression()
	method.fit(a_train, b_train)
	print method.score(a_test, b_test)
	print method.score(a_train, b_train)

def main():
	data = pd.read_csv('Data/train_processed.csv')
	test_model(data)

if __name__=="__main__":
    main()