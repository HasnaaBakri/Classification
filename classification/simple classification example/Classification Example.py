#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
#==============================================functions we need
#sigmoid func
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#cost func
def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))

#gradient func
def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(X * theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)
    
    return grad
#predict func
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


#===============================================================#implementation
#read data
path = 'data2.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
#prepare data
data.insert(0 , 'Ones' , 1)
cols = data.shape[1]
X=np.matrix(data.iloc[: , 0: cols-1])
Y=np.matrix(data.iloc[: , cols-1:])
theta= np.matrix(np.zeros(3))
print()
print('cost befor optimize = ' , cost(theta , X,Y))
print()
# minimize cost func
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, Y))
costafteroptimize = cost(result[0], X,Y)
print()
print('cost after optimize = ' , costafteroptimize)
print()

theta_min = np.matrix(result[0])
print(theta_min)
predictions = predict(theta_min, X)
#print (predictions)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, Y)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))
#=============================================================#ploting
#plot data
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
#plot sigmoid
nums = np.arange(-10, 10, step=1)
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(nums, sigmoid(nums), 'r')








    
    

