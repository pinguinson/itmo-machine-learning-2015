import pandas as pd
import numpy as np

fields_to_drop = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age', 'Fare']

def convert_sex(df):
	df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
	return df

def fill_missing_ages(df):
	median_ages = np.zeros((2,3))#Rows correspond to females, males, columns correspond to the class

	for i in range(0, 2):
	    for j in range(0, 3):
	        median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j + 1)]['Age'].dropna().median()

	df['AgeFill'] = df['Age']

	for i in range(0, 2):
	    for j in range(0, 3):
	        df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j + 1), 'AgeFill'] = median_ages[i,j]
	return df

def get_family_size(df):
	df['FamilySize'] = df['SibSp'] + df['Parch']
	return df

def mul_age_class(df):
	df['Age*Class'] = df.AgeFill * df.Pclass
	return df

def drop(df):
	df = df.drop(fields_to_drop, axis = 1)
	return df

def process_data_frame(df):
	df = convert_sex(df)
	df = fill_missing_ages(df)
	df = get_family_size(df)
	df = mul_age_class(df)
	df = drop(df)
	return df

def process_file(name):
	df = pd.read_csv(name)
	process_data_frame(df).to_csv(name[:-4] + '_processed.csv', index = False)

def main():
	process_file('Data/train.csv')
	process_file('Data/test.csv')

if __name__=="__main__":
    main()