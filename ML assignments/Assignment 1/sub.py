"""Contains all the functions I wrote for the assignment"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn import datasets
from sklearn.model_selection import train_test_split

def compute_cost(yhat,y):
    m = len(y)
    return 1/(2*m)*np.sum((yhat-y)**2)

def initialize_model(X_train,y_train,with_momentum=False):
    global model
    model = {}
    w = np.zeros((X_train.shape[1],1))
    b = np.zeros((1))

    model['w'] = w
    model['b'] = b
    if (with_momentum==True):
        v = np.zeros((X_train.shape[1],1))
        vb = np.zeros((1))
        model['v'] = v
        model['vb'] = vb
    return model

def train_model_final(model,X_train,y_train,num_iterations=10,learning_rate=0.01,beta = 0,gamma=0.9,grad_type = 'vanilla'):
    costs = []
    slopes = []
    intercepts = []
    m = len(y_train)
    y_train = y_train.reshape(y_train.shape[0],1)
    count = 0
    if (grad_type == 'vanilla'):
        model = initialize_model(X_train,y_train)
        w = model['w']
        b = model['b']
        yhat = np.dot(X_train,w) + b
        for i in range(num_iterations):
            w = w - 1/m*np.dot(X_train.T,(yhat-y_train))*learning_rate
            b = b - 1/m*np.sum(yhat-y_train)*learning_rate
            yhat = np.dot(X_train,w) + b
            cost = compute_cost(y_train,yhat)
            costs.append(cost)
        model['w'] = w
        model['b'] = b
    elif (grad_type == 'stochastic'):
        model = initialize_model(X_train,y_train,with_momentum=True)
        w = model['w']
        b = model['b']
        v = model['v']
        vb = model['vb']
        yhat = np.dot(X_train,w) + b
        for i in range(num_iterations):
            for j in range(m):
                yhat = np.dot(X_train,w) + b
                arr = X_train[j,:].reshape(X_train.shape[1],1)
                grad = np.dot(arr,(yhat[j]-y_train[j]))
                grad = grad.reshape(X_train.shape[1],1)
                v = beta*v + (1-beta)*grad
                vb = beta*vb + (1-beta)*(yhat[j]-y_train[j])
                w = w - learning_rate*v
                b = b - learning_rate*vb
                cost = compute_cost(y_train,yhat)
                if (count%1==0):
                    costs.append(cost)
                count += 1
        model['w'] = w
        model['b'] = b
        model['v'] = v
        model['vb'] = vb
    elif (grad_type == 'adagrad'):
        model = initialize_model(X_train,y_train)
        w = model['w']
        b = model['b']
        eps = 10**(-8)
        yhat = np.dot(X_train,w) + b
        G = np.zeros(w.shape)
        Gb = np.zeros(b.shape)
        for i in range(num_iterations):
            for j in range(m):
                yhat = np.dot(X_train,w) + b
                arr = X_train[j,:].reshape(X_train.shape[1],1)
                grad = np.dot(arr,(yhat[j]-y_train[j]))
                grad = grad.reshape(X_train.shape[1],1)
                G += np.square(grad)
                Gb += np.square((yhat[j] - y_train[j]))
                grad_b_corr = (yhat[j] - y_train[j])/np.sqrt(Gb + eps)
                grad_corr = grad/np.sqrt(G + eps)
                w = w - learning_rate*grad_corr
                b = b - learning_rate*grad_b_corr
                cost = compute_cost(y_train,yhat)
                costs.append(cost)
        model['w'] = w
        model['b'] = b
    elif (grad_type == 'rms'):
        model = initialize_model(X_train,y_train,with_momentum=True)
        w = model['w']
        b = model['b']
        v = model['v']
        vb = model['vb']
        eps = 10**(-8)
        for i in range(num_iterations):
            for j in range(m):
                yhat = np.dot(X_train,w) + b
                arr = X_train[j,:].reshape(X_train.shape[1],1)
                grad = np.dot(arr,(yhat[j]-y_train[j]))
                grad = grad.reshape(X_train.shape[1],1)
                v = gamma*v + (1-gamma)*np.square(grad)
                vb = gamma*vb + (1-gamma)*np.square(yhat[j]-y_train[j])
                w = w - learning_rate*grad/np.sqrt(v+eps)
                b = b - learning_rate*(yhat[j]-y_train[j])/np.sqrt(vb+eps)
                cost = compute_cost(y_train,yhat)
                costs.append(cost)
        model['w'] = w
        model['b'] = b
        model['v'] = v
        model['vb'] = vb


    return (costs,model)

""" This part of the code automatically reads the csv file and runs a model on it"""
"""For the first part of the assignment"""

df = pd.read_csv('assignment1.csv')
df.drop(['Sl.No'],axis=1,inplace=True)
df.dropna(how='any',inplace=True)
Y = df[df.columns[0]]
X = df[df.columns[1:]]
X = X.as_matrix()
Y = Y.as_matrix()
Y = Y.reshape(Y.shape[0],1)
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
initialize_model(X_train,y_train)
costs,model = train_model_final(model,X_train,y_train,num_iterations=1000,learning_rate=0.01,grad_type='adagrad')
