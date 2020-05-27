# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 13:30:06 2020

@author: ManavChordia
"""

# Importing the libraries
import pandas as pd
import numpy as np
# Importing the dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
all_data = [train_data,test_data]
#X = dat.iloc[:, [2,3]].values
#y = dataset.iloc[:, 4].values
print( train_data[["Pclass","Survived"]].groupby(["Pclass"], as_index = 0).mean() )

#Feature : Family size
for data in all_data:
    data['Family_size'] = data['SibSp'] + data['Parch']+1
print( train_data[["Family_size","Survived"]].groupby(["Family_size"], as_index = 0).mean() )

"""
#Feature : isAlone
for data in all_data:
    data['is_Alone'] = 0
    data.loc[data['Family_size'] ==1 ,'is_Alone'] = 1

print( train_data[["isAlone","Survived"]].groupby(["isAlone"], as_index = 0).mean() )
"""

for data in all_data:
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Age'] = data['Age'].astype(int)
train_data['category_age'] = pd.cut(train_data['Age'], 5)
print( train_data[["category_age","Survived"]].groupby(["category_age"], as_index = False).mean() )

for data in all_data:

    #Mapping Sex
    data['Sex'] = data['Sex'].fillna('male')
    sex_map = { 'female':0 , 'male':1 }
    data['Sex'] = data['Sex'].map(sex_map).astype(int)

for data in all_data:
        #Mapping Embarked
    data['Embarked'] = data['Embarked'].fillna('S')
    embark_map = {'S':0, 'C':1, 'Q':2}
    data['Embarked'] = data['Embarked'].map(embark_map).astype(int)

for data in all_data:
    #Mapping Fare
    data['Fare'] = data['Fare'].fillna(7.91)
    data.loc[ data['Fare'] <= 7.91, 'Fare']                            = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2
    data.loc[ data['Fare'] > 31, 'Fare']                               = 3
    data['Fare'] = data['Fare'].astype(int)

for data in all_data:
    #Mapping Age
    data['Age'] = data['Age'].fillna(40)
    data.loc[ data['Age'] <= 16, 'Age']                       = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[ data['Age'] > 64, 'Age'] = 4

#Feature Selection
#Create list of columns to drop
    

drop_elements = ["Name", "Ticket", "Cabin", "SibSp", "Parch", "Family_size"]
#Drop columns from both data sets
train_data = train_data.drop(drop_elements, axis = 1)
train_data = train_data.drop(['PassengerId', 'category_age'], axis = 1)
test_data = test_data.drop(drop_elements, axis = 1)

#Print ready to use data
print(train_data.head(10))

x_train = train_data.drop("Survived", axis=1)
y_train = train_data.iloc[:, 0:1]
x_test = test_data.drop("PassengerId", axis=1)

from sklearn import preprocessing

data = preprocessing.normalize(x_train.iloc[:,0:3])

c1 = np.array([0, 0, 0])[np.newaxis]
c2 = np.array([1, 1, 1])[np.newaxis]

n = 2

c1_o = c2_n = c1
c2_o = c1_n = c2



#kmeans main algo
while c1_o[0][0] != c1_n[0][0] and c2_o[0][0] != c2_n[0][0]:
    mid = (c1_o + c2_o)/2
    c1_o = c1_n
    c2_o = c2_n
    
    print(1)
    
    c1_l = []
    c2_l = []
    
    for i in range(0,891):
        
        
        if np.sum(mid*(data[i]-mid)) < 0:
            c1_l.append(data[i])
            
        else:
            c2_l.append(data[i])
            
        c1_a = np.array(c1_l)
        c2_a = np.array(c2_l)
    print(2) 
    c1_n = np.sum(c1_a, axis = 0)[np.newaxis]
    c2_n = np.sum(c2_a, axis = 0)[np.newaxis]
    
    
    
    
"""
import matplotlib.pyplot as plt

plt.scatter(x_train['Embarked'], x_train['Fare'])
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()   
"""


