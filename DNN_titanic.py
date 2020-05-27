

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

# Model Preparation

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

def tanh_der(x):
    dt=1-np.tanh(x)**2
    return dt

def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

class NN_2hd:
    def __init__(self, x, y):
        self.x = x
        n_h = 12
        self.lr = 0.000005
        n_x = x.shape[1]
        n_y = 1

        
        self.W1 = np.random.randn(n_x, n_h) 
        self.b1 = np.zeros(shape=(1, n_h))
        self.W2 = np.random.randn(n_h, n_h)
        self.b2 = np.zeros(shape=(1, n_h))
        self.W3 = np.random.randn(n_h, n_h)
        self.b3 = np.zeros(shape=(1, n_h))
        self.W4 = np.random.randn(n_h, n_y)
        self.b4 = np.zeros(shape=(1, n_y))
        self.y = y
        
    def feedforward(self):
        self.Z1 = np.dot(self.x, self.W1) + self.b1
        self.A1 = np.tanh(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = np.tanh(self.Z2)
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = np.tanh(self.Z3)
        self.Z4 = np.dot(self.A3, self.W4) + self.b4
        self.A4 = sigmoid(self.Z4)
        self.cache = {"Z1": self.Z1,"A1": self.A1,"Z2": self.Z2,"A2": self.A2 ,"Z3": self.Z3,"A3": self.A3, "Z4": self.Z4,"A4": self.A4}
    
        return self.A4, self.cache
        
    def backprop(self):
        #backward propagation
        #output layer
        dZ4 = (self.A4 - self.y)
        #print(dZ3[0:10,0])
        dW4 = np.dot(self.A3.T, dZ4)
        db4 = dZ4
        #print(self.A4[0:20,0])
        
        #hidden layer3
        dA3 = np.dot(dZ4, self.W4.T)
        dZ3_ = tanh_der(self.Z3)
        dZ3 = dZ3_ * dA3
        dW3 = np.dot(self.A2.T, dZ3)
        db3 = dZ3
    
        #hidden layer2
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2_ = tanh_der(self.Z2)
        dZ2 = dZ2_ * dA2
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = dZ2
    
        #hidden layer 1
        dA1 = np.dot(self.W2, dZ2.T)
        dZ1_ = tanh_der(self.Z1)
        dZ1 = dZ1_ * self.A1
        dW1 = np.dot(self.x.T, dZ1)
        db1 = dZ1

        #update the weights
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * np.sum(db1, axis=0, keepdims = True)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * np.sum(db2, axis=0, keepdims = True)

        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * np.sum(db3, axis=0, keepdims = True)
        
        self.W4 -= self.lr * dW4
        self.b4 -= self.lr * np.sum(db4, axis=0, keepdims = True)
        
        temp = (self.A4 - self.y)*(self.A4 - self.y)
        print(np.sum(temp))
        
    def predict(self):
        self.A4, self.cache = self.feedforward()
        predictions = np.round(self.A4)
        print(predictions)

model = NN_2hd(x_train, np.array(y_train))

for i in range(2500):
    model.feedforward()
    model.backprop()
    model.predict()
    if i%100 == 0:
        print(i)
    

"""
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,Y_train)

y_pred = classifier.predict(X_test)

accuracy = round(classifier.score(X_train, Y_train) * 100, 2)
print("Model Accuracy: ",accuracy)

#Create a CSV with results
submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": y_pred
})
submission.to_csv('submission.csv', index = False)
"""