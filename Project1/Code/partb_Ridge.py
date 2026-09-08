import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split


#Part b


#Define Runge function
def Runge_function(x):
    return 1 / (1 + 25*x**2)

#MSE
def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


# R2
def R2(y_true, y_pred):return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

#Generate data and parameteres

n=100
sigma = 0.1
rng = np.random.default_rng(2026)
x = np.linspace(-1,1,n)
y_true = Runge_function(x)
noise = sigma * rng.standard_normal(n)
y = y_true + noise
degree = 15

#Split 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2026)

X_train = np.vander(x_train.ravel(),degree + 1,increasing=True)

X_test = np.vander(x_test.ravel(),degree + 1,increasing=True)

# Scaling
mean_X = np.mean(X_train[:, 1:], axis=0)
std_X = np.std(X_train[:, 1:], axis=0)

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[:, 1:] = (X_train[:, 1:] - mean_X) / std_X
X_test_scaled[:, 1:] = (X_test[:, 1:] - mean_X) / std_X


# Remove intercept column
Xtr_scaled = X_train_scaled[:, 1:]
Xtest_scaled = X_test_scaled[:, 1:]


# Center y
y_mean = np.mean(y_train)
y_train_centered = y_train - y_mean


# Ridge using SVD
def ridge_svd(X, y, lmbda):
    y = y.ravel()
    U, s, Vt = np.linalg.svd(X,full_matrices=False)
    theta = Vt.T @ ((s / (s**2 + lmbda))* (U.T @ y))
    return theta 




lambdas = np.logspace(-8, 4, 100)
train_mse_ridge = []
test_mse_ridge = []


for lmbda in lambdas:
    theta = ridge_svd(Xtr_scaled,y_train_centered,lmbda)
    y_train_pred = (Xtr_scaled @ theta + y_mean)
    y_test_pred = (Xtest_scaled @ theta + y_mean)
    train_mse_ridge.append(MSE(y_train, y_train_pred))
    test_mse_ridge.append(MSE(y_test, y_test_pred))


plt.plot(lambdas,train_mse_ridge,label="Training MSE")
plt.plot(lambdas,test_mse_ridge,label="Test MSE")
plt.xscale("log")
plt.xlabel("Lambda")
plt.ylabel("MSE")
plt.legend()
plt.show()