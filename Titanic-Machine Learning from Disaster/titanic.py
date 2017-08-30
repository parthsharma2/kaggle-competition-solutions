# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 22:06:28 2017

@author: parthsharma2
"""
#Importing numpy, pandas and RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#Reading data
data = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
full_data = [train, test]

#Cleaning the datasets
for dataset in full_data:
    #Creating a new feature 'FamilySize'
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    #Adding missing ages
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size = age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    #Adding missing fares
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
    #Mapping Sex values to numeric codes/values
    dataset['Sex'] = dataset['Sex'].map({'female':0, 'male':1}).astype(int)
    #Mapping Embarked values to numeric coed/values
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

#Dropping unecessary features/columns
train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
#Creating the features and target
features_forest = train[['Pclass', 'Age', 'Sex', 'Fare', 'SibSp', 'Parch', 'Embarked']].values
target = train['Survived'].values
#Creating a RandomForestClassifier Model
forest = RandomForestClassifier(max_depth=10, min_samples_split=2, n_estimators=100, random_state=1)
my_forest = forest.fit(features_forest, target)
#Predicting the test data
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features)
#Creating a dataframe containing the predicted data
PassengerId = np.array(data["PassengerId"]).astype(int)
my_solution = pd.DataFrame(pred_forest, PassengerId, columns=["Survived"])
#Saving the dataframe as a csv file
my_solution.to_csv("my_solution.csv", index_label=["PassengerId"])

print(my_solution)
