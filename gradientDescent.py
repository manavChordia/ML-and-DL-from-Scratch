# -*- coding: utf-8 -*-
"""

@author: ManavChordia
"""

import numpy as np
import matplotlib.pyplot as plt

#to find the cost for a particular value of theta0 and theta1
def findCost(x,y,theta0,theta1):
    sum = 0
    for i in range(0, len(x)-1):
        sum = sum + (((theta0 + theta1*x[i]) - y[i]))**2
    return sum/(2*len(x))


def error(x,y,theta0,theta1):
    sum = 0
    for i in range(0, len(x)-1):
        sum = sum + (((theta0 + theta1*x[i]) - y[i]))
        #print( " " + str(sum/len(x)) + " ")
    return sum/(len(x))

def gradientDescent(x,y,alpha):
    theta0 = theta1 = 0
    iter=0
    cost = findCost(x,y,theta0,theta1)
    while iter<4:
        for i in range(0,len(x)-1):
            theta1 = theta1 - alpha*error(x,y,theta0,theta1)*x[i]
            theta0 = theta1 - alpha*error(x,y,theta0,theta1)
            cost = findCost(x,y,theta0,theta1)
        
            print("cost is : " + str(cost) + " theta0 is : " + str(theta0) + " theta1 is : " + str(theta1))
        iter=iter+1
        
    return theta0,theta1

x = np.array([0,1,3,6,7,8,4])
y = np.array([1,3,4,8,8,9,6])
alpha = 0.05
theta0,theta1 = gradientDescent(x,y,alpha)

yopt = theta0 + theta1*5

plt.scatter(x,y)

yval = [theta1 * i + theta0 for i in x]

# Plot the best fit line over the actual values
plt.plot(x, yval)
plt.show()






