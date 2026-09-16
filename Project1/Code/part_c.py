import torch
from Rungefunction import Runge_function, MSE, R2
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

#The data

torch.manual_seed(2026)
n = 100
sigma = 0.1
x= torch.linspace(-1,1,n,dtype=torch.float64)

y_true = Runge_function(x)

noise = sigma*torch.randn(n,dtype=torch.float64)

y = y_true + noise




#Split 

x_train,x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state = 2026)

train_MSE = []
test_MSE = []

degree_test = 30

for degree in range(1,degree_test +1):

#Design the matrix.

    X_train = torch.vander(x_train,degree +1,increasing= True)
    X_test = torch.vander(x_test,degree +1 , increasing=True)


#Compute the parameter.

thetha_1C = torch.linalg.pinv(X_train)@ y_train


#Prediction

y_train_PRED_1C = x_train @ thetha_1C
y_test_PRED_1C = x_test @thetha_1C

MSE_1C_train = MSE(y_train,y_train_PRED_1C)
MSE_1C_test =  MSE(y_test, y_test_PRED_1C)

print("Train MSE", MSE_1C_train)
print("Test MSE", MSE_1C_test)



