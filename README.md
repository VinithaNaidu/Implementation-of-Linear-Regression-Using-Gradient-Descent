# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.
2.Write a function computeCost to generate the cost function.
3.Perform iterations og gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required graph. 


## Program:
```

Program to implement the linear regression using gradient descent.
Developed by: D.Vinitha Naidu
RegisterNumber:  212222230175

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("/content/ex1.txt", header = None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  m=len(y)
  h=X.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions-y))
    descent=alpha* 1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))
  return theta,J_history

theta,J_history=gradientDescent(X,y,theta,0.01,1500)
print("h(X)="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000 ,we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000,we predict a profit of $"+str(round(predict2,0)))
```


## Output:
### Profit Prediction Graph
![image](https://github.com/VinithaNaidu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121166004/6b78bc3f-c31e-4ad2-a8db-0a31b189a85d)
![image](https://github.com/VinithaNaidu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121166004/bbce1f7e-dcf5-4dbd-b387-4c261551e2d5)

### Compute Cost Value
![image](https://github.com/VinithaNaidu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121166004/722dc497-74dc-48d7-9e37-6a5d25249df1)


### h(x) value
![image](https://github.com/VinithaNaidu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121166004/5d8e78a6-04fe-4ea7-8827-bb5cc3e0a8e5)


### Cost function using Gradient Descent Graph


### Profit Prediction Graph
![image](https://github.com/VinithaNaidu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121166004/33048b97-a244-41b3-b41c-c4fc5daface5)

### Profit for the Population 35,000
![image](https://github.com/VinithaNaidu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121166004/e8abb537-7215-4686-8145-7b1c63cbc42d)

### Profit for the Population 70,000
![image](https://github.com/VinithaNaidu/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121166004/097530d1-a526-413c-8f5b-97986faa3b7f)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
