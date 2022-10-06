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
def costReg(theta, X, y, lr):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = lr/2*len(X) *np.sum(np.power(theta[: , 1:theta.shape[1]], 2))
    return np.sum(first - second) / (len(X))+reg

#gradient func
def gradientReg(theta, X, y, lr):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(X * theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        if (i==0) :
            grad[i] = np.sum(term) / len(X)
        else :
            grad[i]=np.sum(term)/len(X)+((lr/len(X) )*theta[:,i])
    
    return grad
#predict func
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


#===============================================================#implementation
#read data
path = 'data.txt'
data = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])
#before regularization
degree=10
x1=data['Test 1']
x2=data['Test 2']
data.insert(3, 'Ones', 1)   # adding x0 ''' Remember adding x0 '''

for i in range(1, degree):
    for j in range (0,i):
        data['F'+str(i)+str(j)]=np.power(x1 , i-j)*np.power(x2, j)
        
data.drop('Test 1' , axis=1 , inplace=True)
data.drop('Test 2' , axis=1 , inplace=True)

#prepare data
cols =data.shape[1]
X=np.matrix(data.iloc[: , 1:cols])
Y=np.matrix(data.iloc[: , 0:1])
theta=np.matrix(np.zeros(X.shape[1]))

# work
lr= 0.00001
cost = costReg(theta, X, Y, lr)
result =opt.fmin_tnc(func =costReg ,x0 =theta ,fprime=gradientReg , args=(X ,Y, lr)  )
theta_min=np.matrix(result[0] )
prediction = predict(theta_min , X)
correct =[1 if ((a==1 and b==1)or (a==0 and b==0)) else 0 for (a , b) in zip(prediction,Y)]
accuracy = (sum(map(int , correct))% len (correct))
print ('accuracy ={0}%' .format(accuracy))







    
    

