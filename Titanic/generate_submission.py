import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def main():
	train = pd.read_csv('Data/train_processed.csv')
	test = pd.read_csv('Data/test_processed.csv')
	train_targets = train['Survived'].values
	train = train.drop(['Survived'], axis = 1)
	train_data = train.values
	clf = RandomForestClassifier()
	logReg = clf.fit(train_data, train_targets)
	predicted = logReg.predict(test)
	test['Survived'] = predicted
	submission = pd.concat([test['PassengerId'], test['Survived']], axis=1, keys=['PassengerId', 'Survived'])
	submission.to_csv('Data/submission_rfc.csv', index = False)

if __name__=="__main__":
    main()