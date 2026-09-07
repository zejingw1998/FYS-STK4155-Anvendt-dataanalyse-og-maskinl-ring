import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split


rng = np.random.default_rng(2026)


#Generate x values

n = 100
x = np.linspace(-1,1,n) 

#Define the Runge fuction 
def Runge_function(x):
    return 1 / (1+25*x**2) 


#True function values
y_true = Runge_function(x)


#Add noise
sigma = 0.1
noise = sigma* rng.standard_normal(n)

y = y_true +noise 


# Plot
plt.scatter(x, y, s=10, label="Noisy data")
plt.plot(x, y_true, label="Runge function")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

#Split the data

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state= 2026)

#Degree of polynomial
degree = 15 


#Design matrix
X_train = np.vander(x_train.ravel(), degree +1, increasing=True) #Train
X_test = np.vander(x_test.ravel(), degree +1, increasing=True) #Test


#Find the OLS parameters

theta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

#Predictions

y_train_pred =X_train @ theta
y_test_pred = X_test @ theta 


#Compute the MSE 

def MSE(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)


mse_train= MSE(y_train,y_train_pred)
mse_test=MSE(y_test,y_test_pred)

print("Traing MSE",mse_train)
print("Tset MSE",mse_test)

#The traing MSE is 0.008422895082031664
#The test MSE is 0.06116904521970146


#Compute the R^2

def R2(y_true,y_pred):
    return 1-np.sum((y_true-y_pred)**2)/ np.sum((y_true-np.mean(y_true))**2)


R2_train = R2(y_train,y_train_pred)
R2_test = R2(y_test,y_test_pred)

print("Train R2",R2_train)
print("Test R2",R2_test)


#The train R2 is 0.9237913217985219
#The test R2 is 0.3335873280577112

#Part b

# Ridge using SVD
def ridge_svd(X, y, lmbda):

    U, s, Vt = np.linalg.svd(X,full_matrices=False)

    return Vt.T @ (s / (s**2 + lmbda) * (U.T @ y))