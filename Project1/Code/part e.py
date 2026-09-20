import torch 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from Rungefunction import Runge_function, MSE, R2


#Gradient descent



#Analytical gradients

n = 100

sigma = 0.1

x = torch.linspace(-1,1,n,dtype=torch.float64)
y = Runge_function(x) + sigma*torch.randn(n,dtype=torch.float64)



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=2026)

degree_first_test = 5

X_train = torch.vander(x_train,degree_first_test+1,increasing=True)
X_test = torch.vander(x_test,degree_first_test+1,increasing=True)


X_train = torch.vander(x_train,degree_first_test+1,increasing=True)
X_test = torch.vander(x_test,degree_first_test+1,increasing=True)


def OLS_Gradient(X,y,theta):


    n = len(y)



    gradient_1E = (2/n)*X.T @(X@theta-y)


    return gradient_1E





theta_1E = torch.zeros(X_train.shape[1],dtype=torch.float64)


learning_rate= 0.01

iterations = 10000


history = []

for i in range(iterations):
    gradient = OLS_Gradient(X_train,y_train,theta_1E)
    theta_1E = theta_1E-learning_rate*gradient
    history.append(MSE(y_train,X_train@theta_1E).item())


print("GD theta =",theta_1E)

theta_closed = torch.linalg.pinv(X_train)@y_train

print("Closed form theta =",theta_closed)
print("Difference =",torch.norm(theta_1E-theta_closed))
print("GD theta =",theta_1E)



theta_closed = torch.linalg.pinv(X_train)@y_train

print("Closed form theta =",theta_closed)
print("Difference =",torch.norm(theta_1E-theta_closed))