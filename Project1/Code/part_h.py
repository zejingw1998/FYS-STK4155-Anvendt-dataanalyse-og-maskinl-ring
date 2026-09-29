import torch 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from Rungefunction import Runge_function, MSE, R2


torch.manual_seed(2026)


def OLS_Gradient(X, y, theta):
    n = len(y)
    gradient = (2/n) * X.T @ (X @ theta - y)
    return gradient


#Stochastic Gradicent Descent
n = 100

sigma = 0.1


x = torch.linspace(-1,1,n,dtype = torch.float64)
y = Runge_function(x) + sigma * torch.randn(n,dtype= torch.float64)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=2026)
degree_first_test = 5
X_train = torch.vander(x_train,degree_first_test+1,increasing=True)
X_test = torch.vander(x_test,degree_first_test+1,increasing=True)
Thetha_Stochastic_Gradicent = torch.zeros(X_train.shape[1],dtype= torch.float64)


#The parameters


epochs = 100

learning_Rate_H = 0.01

Batch_Size = 20

for epoch in range(epochs):


    indices = torch.randperm(len(y_train))

    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    for start in range(0, len(y_train), Batch_Size):
        end = start + Batch_Size
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]
        gradient = OLS_Gradient(X_batch, y_batch, Thetha_Stochastic_Gradicent)
        Thetha_Stochastic_Gradicent = Thetha_Stochastic_Gradicent - learning_Rate_H * gradient
y_pred_SGD = X_test @ Thetha_Stochastic_Gradicent

print("SGD MSE =", MSE(y_test, y_pred_SGD))
print("SGD R2 =", R2(y_test, y_pred_SGD))


sorted_index = torch.argsort(x_test)

plt.scatter(x_test,y_test,label="Test data")
plt.plot(x_test[sorted_index],y_pred_SGD[sorted_index],label="SGD")
plt.xlabel("x")
plt.ylabel("y")
plt.title("SGD R2 MSE" )
plt.legend()
plt.show()


"""
SGD MSE = tensor(0.1495, dtype=torch.float64)
SGD R2 = tensor(-0.7125, dtype=torch.float64)
"""

# Try several values of the epoch value

epochs_serval = [10, 50, 100, 500, 1000, 5000]

MSE_epochs = []

for epochs in epochs_serval:

    Thetha_Stochastic_Gradicent = torch.zeros(X_train.shape[1],dtype=torch.float64)

    for epoch in range(epochs):

        indices = torch.randperm(len(y_train))

        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        for start in range(0,len(y_train),Batch_Size):

            end = start + Batch_Size

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            gradient = OLS_Gradient(X_batch,y_batch,Thetha_Stochastic_Gradicent)

            Thetha_Stochastic_Gradicent = Thetha_Stochastic_Gradicent - learning_Rate_H*gradient

    y_pred_SGD = X_test @ Thetha_Stochastic_Gradicent

    mse = MSE(y_test,y_pred_SGD)

    MSE_epochs.append(mse.item())

    print("Epochs =",epochs,"MSE =",mse.item())
plt.plot(epochs_serval,MSE_epochs,marker="o")
plt.xscale("log")
plt.xlabel("Epochs")
plt.ylabel("Test MSE")
plt.title("SGD: Test MSE vs  Several Epochs")
plt.show()


#The several values of the batch size 


Batch_Size_serval= [1,5,10,20,35,70]

epochs = 1000
MSE_Batch_Size = []

for Batch_Size in Batch_Size_serval:

    Thetha_Stochastic_Gradicent = torch.zeros(X_train.shape[1],dtype=torch.float64)

    for epoch in range(epochs):

        indices = torch.randperm(len(y_train))

        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        for start in range(0,len(y_train),Batch_Size):

            end = start + Batch_Size

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            gradient = OLS_Gradient(X_batch,y_batch,Thetha_Stochastic_Gradicent)

            Thetha_Stochastic_Gradicent = Thetha_Stochastic_Gradicent - learning_Rate_H*gradient

    y_pred_SGD = X_test @ Thetha_Stochastic_Gradicent

    mse = MSE(y_test,y_pred_SGD)

    MSE_Batch_Size.append(mse.item())

    print("Batch Size =",Batch_Size,"MSE =",mse.item())


plt.plot(Batch_Size_serval,MSE_Batch_Size,marker="o")
plt.xlabel("Batch Size")
plt.ylabel("Test MSE")
plt.title("SGD: Test MSE vs Several Batch Sizes")
plt.show()

# Learning rate schedule

epochs = 5000
Batch_Size = 20

learning_rate_start = 0.01
decay = 0.001

Thetha_SGD_Schedule = torch.zeros(X_train.shape[1],dtype=torch.float64)

MSE_Schedule = []

for epoch in range(epochs):

    learning_rate = learning_rate_start / (1 + decay*epoch)

    indices = torch.randperm(len(y_train))

    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    for start in range(0,len(y_train),Batch_Size):

        end = start + Batch_Size

        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        gradient = OLS_Gradient(X_batch,y_batch,Thetha_SGD_Schedule)

        Thetha_SGD_Schedule = Thetha_SGD_Schedule - learning_rate*gradient

    y_pred_epoch = X_test @ Thetha_SGD_Schedule
    mse_epoch = MSE(y_test,y_pred_epoch)

    MSE_Schedule.append(mse_epoch.item())


y_pred_Schedule = X_test @ Thetha_SGD_Schedule

MSE_schedule = MSE(y_test,y_pred_Schedule)
R2_schedule = R2(y_test,y_pred_Schedule)

print("SGD with schedule MSE =",MSE_schedule)
print("SGD with schedule R2 =",R2_schedule)


plt.plot(range(epochs),MSE_Schedule)
plt.xlabel("Epoch")
plt.ylabel("Test MSE")
plt.title("SGD with Learning Rate Schedule")
plt.show()


# Full GD

epochs_compare = 5000
learning_rate_compare = 0.01

Theta_Full_GD = torch.zeros(X_train.shape[1],dtype=torch.float64)

MSE_Full_GD = []

for epoch in range(epochs_compare):

    gradient = OLS_Gradient(X_train,y_train,Theta_Full_GD)

    Theta_Full_GD = Theta_Full_GD - learning_rate_compare*gradient

    y_pred_Full_GD = X_test @ Theta_Full_GD

    MSE_Full_GD.append(MSE(y_test,y_pred_Full_GD).item())


print("Full GD MSE =",MSE(y_test,y_pred_Full_GD))
print("Full GD R2 =",R2(y_test,y_pred_Full_GD))


# Closed-form OLS

Theta_Closed = torch.linalg.pinv(X_train) @ y_train

y_pred_Closed = X_test @ Theta_Closed

print("Closed Form MSE =",MSE(y_test,y_pred_Closed))
print("Closed Form R2 =",R2(y_test,y_pred_Closed))


# Compare convergence

plt.plot(range(epochs_compare),MSE_Full_GD,label="Full GD")
plt.plot(range(epochs),MSE_Schedule,label="Mini-Batch SGD")
plt.xlabel("Epoch")
plt.ylabel("Test MSE")
plt.title("Full-Batch GD vs Mini-Batch SGD")
plt.legend()
plt.show()



import time

# Computational cost comparison

epochs_time = 5000
Batch_Size_time = 20
learning_rate_time = 0.01


# Mini-Batch SGD

Theta_SGD_Time = torch.zeros(X_train.shape[1],dtype=torch.float64)

start_time = time.perf_counter()

for epoch in range(epochs_time):
    indices = torch.randperm(len(y_train))
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    for start in range(0,len(y_train),Batch_Size_time):
        end = start + Batch_Size_time
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]
        gradient = OLS_Gradient(X_batch,y_batch,Theta_SGD_Time)

        Theta_SGD_Time = Theta_SGD_Time - learning_rate_time*gradient

end_time = time.perf_counter()

SGD_time = end_time-start_time


# Full-Batch GD

Theta_GD_Time = torch.zeros(X_train.shape[1],dtype=torch.float64)

start_time = time.perf_counter()

for epoch in range(epochs_time):
    gradient = OLS_Gradient(X_train,y_train,Theta_GD_Time)
    Theta_GD_Time = Theta_GD_Time - learning_rate_time*gradient
end_time = time.perf_counter()
GD_time = end_time-start_time


# Closed-form
start_time = time.perf_counter()
Theta_Closed_Time = torch.linalg.pinv(X_train) @ y_train
end_time = time.perf_counter()
Closed_time = end_time-start_time
print("Mini-Batch SGD time =",SGD_time)
print("Full GD time =",GD_time)
print("Closed Form time =",Closed_time)





"""
SGD MSE = tensor(0.0549, dtype=torch.float64)
SGD R2 = tensor(0.3710, dtype=torch.float64)
Epochs = 10 MSE = 0.0956072772301189
Epochs = 50 MSE = 0.06915689383084173
Epochs = 100 MSE = 0.05495076883207279
Epochs = 500 MSE = 0.04518959796170568
Epochs = 1000 MSE = 0.039589061456543825
Epochs = 5000 MSE = 0.023252531928110737
Batch Size = 1 MSE = 0.03428235998720144
Batch Size = 5 MSE = 0.024399103668019388
Batch Size = 10 MSE = 0.032321488062090634
Batch Size = 20 MSE = 0.039501148271023565
Batch Size = 35 MSE = 0.04521989774502706
Batch Size = 70 MSE = 0.04739937754415114
SGD with schedule MSE = tensor(0.0320, dtype=torch.float64)
SGD with schedule R2 = tensor(0.6336, dtype=torch.float64)
Full GD MSE = tensor(0.0368, dtype=torch.float64)
Full GD R2 = tensor(0.5780, dtype=torch.float64)
Closed Form MSE = tensor(0.0430, dtype=torch.float64)
Closed Form R2 = tensor(0.5077, dtype=torch.float64)
Mini-Batch SGD time = 0.3867867999942973
Full GD time = 0.05700959998648614
Closed Form time = 0.0004070000140927732
"""