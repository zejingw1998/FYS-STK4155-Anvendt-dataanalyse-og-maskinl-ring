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
plt.plot(range(iterations),history)
plt.xlabel("Iteration")
plt.ylabel("Training MSE")
plt.yscale("log")
plt.title("OLS Gradient Descent Convergence")
plt.show()


y_pred_GD = X_test@theta_1E
y_pred_closed = X_test@theta_closed

sorted_index = torch.argsort(x_test)

plt.scatter(x_test,y_test,label="Test data")
plt.plot(x_test[sorted_index],y_pred_GD[sorted_index],label="Gradient Descent")
plt.plot(x_test[sorted_index],y_pred_closed[sorted_index],label="Closed form")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()


plt.plot(history)
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.yscale("log")
plt.show()



#Hessian

H = (2/len(y_train))*X_train.T@X_train
eigenvalues = torch.linalg.eigvalsh(H)
lambda_max = torch.max(eigenvalues)

print("Largest eigenvalue =",lambda_max.item())
print("Approx maximum learning rate =",2/lambda_max.item())


#Autograd


theta_1E_Autograd_differation = torch.zeros(X_train.shape[1],dtype=torch.float64,requires_grad=True)
y_pred_auto = X_train@theta_1E_Autograd_differation
cost_auto = torch.mean((y_train-y_pred_auto)**2)
cost_auto.backward()

print("Automatic gradient =",theta_1E_Autograd_differation.grad)



gradient_analytical = OLS_Gradient(X_train,y_train,theta_1E_Autograd_differation.detach())

print("Analytical gradient =",gradient_analytical)
print("Automatic gradient =",theta_1E_Autograd_differation.grad)
print("Gradient difference =",torch.norm(gradient_analytical-theta_1E_Autograd_differation.grad))


plt.plot(gradient_analytical.detach(),label="Analytical gradient")
plt.plot(theta_1E_Autograd_differation.grad.detach(),label="Autograd gradient")

plt.xlabel("Parameter index")
plt.ylabel("Gradient")

plt.legend()
plt.show()


history_Auto=[]


for i in range(iterations):
    cost = torch.mean((y_train - X_train@theta_1E_Autograd_differation)**2)

    cost.backward()


    with torch.no_grad():
        theta_1E_Autograd_differation -= learning_rate*theta_1E_Autograd_differation.grad


        theta_1E_Autograd_differation.grad.zero_()

        history_Auto.append(cost.item())

plt.plot(history,label="Analytical Gradient")
plt.plot(history_Auto,label="Autograd")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.yscale("log")
plt.legend()
plt.show()







# Ridge

lambda_1E_Ridge = 1e-8

def Ridge_AU_Gradient(X,y,theta,lmbda):
    n = len(y)
    gradient = (2/n)*(X.T@(X@theta-y)+lmbda*theta)
    return gradient

def ridge_svd_1E(X,y,lmbda):
    y = y.ravel()
    U,s,Vt = torch.linalg.svd(X,full_matrices=False)
    theta = Vt.T@((s/(s**2+lmbda))*(U.T@y))
    return theta

degrees_1E_AUTO_Ridge = [1,5,20,25,30,50]

Ridge_Difference = []

for degree in degrees_1E_AUTO_Ridge:

    X_train = torch.vander(x_train,degree+1,increasing=True)
    X_test = torch.vander(x_test,degree+1,increasing=True)

    mean_X = torch.mean(X_train[:,1:],dim=0)
    std_X = torch.std(X_train[:,1:],dim=0,correction=0)

    Xtr_scaled = (X_train[:,1:]-mean_X)/std_X
    Xtest_scaled = (X_test[:,1:]-mean_X)/std_X

    y_mean = torch.mean(y_train)
    y_train_centered = y_train-y_mean

    theta_Ridge_GD = torch.zeros(Xtr_scaled.shape[1],dtype=torch.float64)

    history_Ridge = []

    for i in range(iterations):
        gradient = Ridge_AU_Gradient(Xtr_scaled,y_train_centered,theta_Ridge_GD,lambda_1E_Ridge)
        theta_Ridge_GD = theta_Ridge_GD-learning_rate*gradient
        cost = MSE(y_train_centered,Xtr_scaled@theta_Ridge_GD)+(lambda_1E_Ridge/len(y_train_centered))*torch.sum(theta_Ridge_GD**2)
        history_Ridge.append(cost.item())

    theta_Ridge_closed = ridge_svd_1E(Xtr_scaled,y_train_centered,lambda_1E_Ridge)

    Difference = torch.norm(theta_Ridge_GD-theta_Ridge_closed)

    print("Degree =",degree)
    print("Ridge GD =",theta_Ridge_GD)
    print("Ridge closed form =",theta_Ridge_closed)
    print("Difference =",Difference)

    Ridge_Difference.append(Difference.item())

plt.plot(degrees_1E_AUTO_Ridge,Ridge_Difference,marker="o")
plt.xlabel("Polynomial degree")
plt.ylabel("Difference")
plt.yscale("log")
plt.title("Ridge GD vs Closed-form")
plt.show()


I = torch.eye(Xtr_scaled.shape[1],dtype=torch.float64)

H_Ridge = (2/len(y_train_centered))*(Xtr_scaled.T@Xtr_scaled+lambda_1E_Ridge*I)

eigenvalues_Ridge = torch.linalg.eigvalsh(H_Ridge)

lambda_max_Ridge = torch.max(eigenvalues_Ridge)

eta_max_Ridge = 2/lambda_max_Ridge

print("Ridge largest eigenvalue =",lambda_max_Ridge.item())
print("Ridge maximum learning rate =",eta_max_Ridge.item())


# Autograd gradient Ridge

theta_1E_Autograd_Ridge_differation = torch.randn(Xtr_scaled.shape[1],dtype=torch.float64,requires_grad=True)

y_pred_auto_Ridge = Xtr_scaled@theta_1E_Autograd_Ridge_differation

cost_auto_Ridge = torch.mean((y_train_centered-y_pred_auto_Ridge)**2)+(lambda_1E_Ridge/len(y_train_centered))*torch.sum(theta_1E_Autograd_Ridge_differation**2)

cost_auto_Ridge.backward()

gradient_Ridge_analytical = Ridge_AU_Gradient(Xtr_scaled,y_train_centered,theta_1E_Autograd_Ridge_differation.detach(),lambda_1E_Ridge)

Difference_Auto_Norm = torch.norm(gradient_Ridge_analytical-theta_1E_Autograd_Ridge_differation.grad)

print("Ridge analytical gradient =",gradient_Ridge_analytical)
print("Ridge Autograd gradient =",theta_1E_Autograd_Ridge_differation.grad)
print("Ridge gradient difference =",Difference_Auto_Norm)