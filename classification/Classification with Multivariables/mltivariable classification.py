import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
#====================================
def sigmoid (z):
    return (1/(1+np.exp(-z)))
def cost(theta , X,y,lr):
    theta =np.matrix(theta)
    X=np.matrix(X)
    y=np.matrix(y)
    first=np.multiply(-y ,np.log(sigmoid(X*theta.T)))
    second =np.multiply((1-y), np.log(1-sigmoid(X*theta.T)))
    reg=np.multiply(lr/2,np.sum(np.power(theta[: , 1:],2)))
    return np.sum(first-second)/(len(X))+reg
def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    # parameters = int(theta.ravel().shape[1])
    error = sigmoid(X * theta.T) - y
    
    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)
    
    # intercept gradient is not regularized
    grad[0, 0] = np.sum(np.multiply(error, X[:,0])) / len(X)
    
    return np.array(grad).ravel()
def one_vs_all(X, y, num_labels, learning_rate):
    rows = X.shape[0] #5000
    params = X.shape[1] #400

    # k X (n + 1) array for the parameters of each of the k classifiers
    all_theta = np.zeros((num_labels, params ))
    # print('all_theta shape ' , all_theta.shape)
    # labels are 1-indexed instead of 0-indexed
    for i in range(1, num_labels + 1):
        theta = np.zeros(params )
        y_i = np.array([1 if label == i else 0 for label in y])
        # print('=============')
        # print (y_i)
        y_i = np.reshape(y_i, (rows, 1))
        # print('==============')
        # print (y_i)
        
        # minimize the objective function
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
        all_theta[i-1,:] = fmin.x
    
    return all_theta

def predict_all (X , all_theta):
    X=np.matrix(X)
    all_theta=np.matrix(all_theta)
    h_argmax=np.argmax((sigmoid(X*all_theta.T)) ,axis=1 )+1
    return h_argmax
#=====================================
data =loadmat('ex3data1.mat')
X=np.matrix(data['X'])
np.insert(X , 0, values=np.ones(X.shape[0]),axis=1)
rows = X.shape[0]
params = X.shape[1]
all_theta = np.zeros((10, params + 1))
theta=np.zeros(params)
all_theta = one_vs_all(X, data['y'], 10, 0.099)
y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
y_0 = np.reshape(y_0, (rows, 1))
y_pred = predict_all(X, all_theta)
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('accuracy = {0}%'.format(accuracy * 100))
























