# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 22:54:42 2020

@author: ManavChordia
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import norm
import numpy as np


iris = sns.load_dataset("iris")
iris = iris.rename(index = str, columns = {'sepal_length':'1_sepal_length','sepal_width':'2_sepal_width', 'petal_length':'3_petal_length', 'petal_width':'4_petal_width'})

df = iris[["1_sepal_length", "2_sepal_width",'species']]
df = df.sample(frac = 1)
for i in range(150):
    if df.iloc[i, 2] == 'setosa':
        df.iloc[i,2] = 0
    elif df.iloc[i,2] == 'versicolor':
        df.iloc[i,2] = 1
    else:
        df.iloc[i,2] = 2
#x = iris.iloc[:, :2]
#y = iris['species']

plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c = df.iloc[:, 2])
plt.xlabel('Sepa1 Length')
plt.ylabel('Sepal Width')



def predict_NB_gaussian_class(X,mu_list,std_list,pi_list): 
    #Returns the class for which the Gaussian Naive Bayes has greatest value
    scores_list = []
    classes = len(mu_list)
    
    for p in range(classes):
        score = (norm.pdf(x = X[0], loc = mu_list[p][0][0], scale = std_list[p][0][0] )  
                * norm.pdf(x = X[1], loc = mu_list[p][0][1], scale = std_list[p][0][1] ) 
                * pi_list[p])
        scores_list.append(score)
             
    return np.argmax(scores_list)


mu_list = np.split(df[:120].groupby('species').mean().values,[1,2])
std_list = np.split(df[:120].groupby('species').std().values,[1,2], axis = 0)
pi_list = df[:120].iloc[:,2].value_counts().values / len(df)

def make_prediction(X):
    predictions = []
    for i in range(X.shape[0]):
        prediction = np.array(  [predict_NB_gaussian_class( np.array(X.iloc[i]), mu_list, std_list, pi_list)] )
        #print(prediction[0])
        predictions.append(prediction[0])
    return predictions

predictions = make_prediction(df[120:])

from sklearn.metrics import confusion_matrix
confusion_matrix(np.array(df[120:].iloc[:, -1]), np.array(predictions))