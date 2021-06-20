## importing all the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import sklearn
from sklearn.model_selection import train_test_split


def normalize_data(data):
    n_features = data.shape[1]
    new_data = np.zeros((data.shape))
    for i in range(n_features):
        new_data[:,i] = data[:,i]/abs(max(data[:,i],key=abs))
    return new_data

def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return (mean_error**0.5)

def predict(x,coeff):
    y_pred = []
    for i in x:
        summ = 0
        for j in range(len(coeff)):
            summ = summ + coeff[j]*i[j]
        y_pred.append(summ)
    y_pred = np.array(y_pred)
    return y_pred
        
def predict_optimized(x,coeff):
    pred_matrix = coeff*x
    y_pred= np.sum(pred_matrix, axis=-1)
    return y_pred
    
def grad_desc(x_train, y_train, l_rate, n_epoch):
    coeff = np.random.rand((n_features))
    err = []
    for epoch in range(n_epoch):
        y_pred = predict_optimized(x_train, coeff)
        error = y_pred - y_train
        rmse = rmse_metric(y_train,y_pred)
        for i in range(len(coeff)):
            coeff[i] = coeff[i] - (l_rate)*(2/len(x_train))*(np.sum(error*x_train[:,i]))
        err.append(rmse)
    return(coeff,err)
    
def batch_grad_desc(x_train, y_train, l_rate, n_epoch, batch_size = 1):
    coeff = np.random.rand((n_features))
    err = []
    n_batch = int(len(x_train)/batch_size)
    for epoch in range(n_epoch):
        for batch in range(n_batch):
            x = x_train[batch*batch_size:(batch+1)*batch_size]
            y = y_train[batch*batch_size:(batch+1)*batch_size]
            y_pred = predict_optimized(x, coeff)
            error = y_pred - y
            for i in range(len(coeff)):
                coeff[i] = coeff[i] - (l_rate)*(2/len(x))*(np.sum(error*x[:,i]))
        
        x = x_train[batch*batch_size:]
        y = y_train[batch*batch_size:]
        y_pred = predict_optimized(x, coeff)
        error = y_pred - y
        for i in range(len(coeff)):
            coeff[i] = coeff[i] - (l_rate)*(2/len(x))*(np.sum(error*x[:,i]))
        
        y_pred = predict_optimized(x_train, coeff)
        rmse = rmse_metric(y_train,y_pred)
        err.append(rmse)
    
    return(coeff,err)
    
def evaluate(x_test, y_test, coeff):
    y_pred = predict_optimized(x_test, coeff)
    rmse = rmse_metric(y_test,y_pred)
    print(rmse)
    
    
df = pd.read_csv("winequality-red.csv",delimiter=";")
#corrMatrix = df.corr()
#sn.heatmap(corrMatrix, annot=True)
#plt.show()

data = df.to_numpy()
X = data[:,:-1]
Y = data[:,-1]
XX = X

temp = np.ones((X.shape[0],X.shape[1]+1))
temp[:,1:] = X
X = normalize_data(temp)

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=42)
n_features = X.shape[1]

coe, err = batch_grad_desc(x_train,y_train,0.01,200)
plt.plot(err)
evaluate(x_test,y_test,coe)
y_pred = predict_optimized(x_test, coe)