import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from Rungefunction import Runge_function, MSE , R2



#Part b

# Ridge using SVD
def ridge_svd(X, y, lmbda):

    U, s, Vt = np.linalg.svd(X,full_matrices=False)

    return Vt.T @ (s / (s**2 + lmbda) * (U.T @ y))


#Lambda grid


lambdas = np.logspace(-8,4,100)

train_mse_ridge = []
test_mse_ridge = []


for lmbda in lambdas:

    theta = ridge_svd(Xtr_scaled,y_train_centered,lmbda)

    y_train_pred = Xtr_scaled @ theta + y_mean
    y_test_pred = Xtest_scaled @ theta + y_mean

    train_mse_ridge.append(MSE(y_train.ravel(), y_train_pred))

    test_mse_ridge.append(MSE(y_test.ravel(), y_test_pred))


plt.plot(lambdas,train_mse_ridge,label="Training MSE")

plt.plot(lambdas,test_mse_ridge,label="Test MSE")

plt.xscale("log")
plt.xlabel("Lambda")
plt.ylabel("MSE")
plt.legend()
plt.show()
