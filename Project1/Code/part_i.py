import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from Rungefunction import Runge_function, MSE, R2



print("I")


torch.manual_seed(2026)

n = 100
sigma = 0.1

x = torch.linspace(-1,1,n,dtype=torch.float64)
y = Runge_function(x)+sigma*torch.randn(n,dtype=torch.float64)
max_degree = 12
degrees = range(1,max_degree+1)
k_values = [5,10]


#OLS
def OLS_Cross_validation_I(x,y,k):

    CV_MSE_I_OLS = []

    for degree in range(1,max_degree+1):

        fold_MSE = []
        kfold = KFold(n_splits=k,shuffle=True,random_state=2026)

        for train_index,test_index in kfold.split(x):

            x_train = x[train_index]
            x_test = x[test_index]
            y_train = y[train_index]
            y_test = y[test_index]

            X_train = torch.vander(x_train,degree+1,increasing=True)
            X_test = torch.vander(x_test,degree+1,increasing=True)

            theta = torch.linalg.pinv(X_train)@y_train
            y_pred = X_test@theta

            fold_MSE.append(MSE(y_test,y_pred).item())

        CV_MSE_I_OLS.append(sum(fold_MSE)/len(fold_MSE))

    return CV_MSE_I_OLS


OLS_results = {}

for k in k_values:
    OLS_MSE = OLS_Cross_validation_I(x,y,k)
    OLS_results[k] = OLS_MSE

for k in k_values:
    best_index = torch.tensor(OLS_results[k]).argmin()
    print("OLS",k,"fold: Best degree =",best_index.item()+1,"Best MSE =",OLS_results[k][best_index])



for k in k_values:
    plt.plot(degrees,OLS_results[k],label=f"{k}-fold OLS")

plt.xlabel("Polynomial degree")
plt.ylabel("CV MSE")
plt.yscale("log")
plt.title("OLS Cross Validation")
plt.legend()
plt.show()
#Ridge 


def RIdge_SVD(X,y,lambda_I):

    y = y.ravel()

    U,s,Vh = torch.linalg.svd(X,full_matrices= False)

    theta_I_RIDGE = Vh.T @ ((s/(s**2+lambda_I))*(U.T@y))

    return theta_I_RIDGE



def Ridge_Croos_Validation_I(x,y,k,lambda_I):

    Ridge_Croos_Validation_I_MSE = []

    for degree in range(1,max_degree+1):
        lambda_I_MSE = []
        for lmbda in lambda_I:
            fold_MSE = []
            kfold = KFold(n_splits=k,shuffle=True,random_state=2026)
            for train_index,test_index in kfold.split(x):
                x_train = x[train_index]
                x_test = x[test_index]
                y_train = y[train_index]
                y_test = y[test_index]
            
                X_train = torch.vander(x_train,degree+1,increasing=True)
                X_test = torch.vander(x_test,degree+1,increasing=True)
            
                mean_X = torch.mean(X_train[:,1:],dim=0)
                std_X = torch.std(X_train[:,1:],dim=0,correction=0)
            
                X_train_scaled = (X_train[:,1:]-mean_X)/std_X
                X_test_scaled = (X_test[:,1:]-mean_X)/std_X
            
                y_mean = torch.mean(y_train)
                y_train_centered = y_train-y_mean
            
                theta = RIdge_SVD(X_train_scaled,y_train_centered,lmbda)
                y_pred = X_test_scaled@theta+y_mean
            
                fold_MSE.append(MSE(y_test,y_pred).item())
            
            lambda_I_MSE.append(sum(fold_MSE)/len(fold_MSE))
            
        Ridge_Croos_Validation_I_MSE.append(lambda_I_MSE)
            
    return torch.tensor(Ridge_Croos_Validation_I_MSE,dtype=torch.float64)

lambda_I = torch.logspace(-8,4,50,dtype=torch.float64)

Ridge_results = {}
Ridge_best_lambda = {}

for k in k_values:
    Ridge_MSE = Ridge_Croos_Validation_I(x,y,k,lambda_I)
    best_MSE,best_lambda_index = torch.min(Ridge_MSE,dim=1)
    best_lambdas = lambda_I[best_lambda_index]

    Ridge_results[k] = best_MSE
    Ridge_best_lambda[k] = best_lambdas

    plt.plot(degrees,best_MSE,label=f"{k}-fold Ridge")

    for degree in range(1,max_degree+1):
        print("k =",k,"Degree =",degree,"Best lambda =",best_lambdas[degree-1].item(),"Best MSE =",best_MSE[degree-1].item())

plt.xlabel("Polynomial degree")
plt.ylabel("Best CV MSE")
plt.yscale("log")
plt.title("Ridge Cross Validation")
plt.legend()
plt.show()


for k in k_values:
    best_degree_index = torch.argmin(Ridge_results[k])
    print("Ridge",k,"fold: Best degree =",best_degree_index.item()+1,"Best lambda =",Ridge_best_lambda[k][best_degree_index].item(),"Best MSE =",Ridge_results[k][best_degree_index].item())



# Lasso




def Lasso_Gradient_I(X,y,theta,lmbda):

    n = len(y)
    gradient = (2/n)*X.T@(X@theta-y)+(lmbda/n)*torch.sign(theta)

    return gradient

def optimise_Lasso_I(grad_func,theta,learning_rate,iterations,tolerance=1e-5):

    G = torch.zeros_like(theta)
    epsilon = 1e-8

    for i in range(iterations):
        gradient = grad_func(theta)
        G = G+gradient**2
        theta_new = theta-learning_rate*gradient/(torch.sqrt(G)+epsilon)

        if torch.max(torch.abs(theta_new-theta))<tolerance:
            return theta_new,i+1

        theta = theta_new

    return theta,iterations

def Lasso_Cross_Validation_I(x,y,k,lambda_I):

    Lasso_Cross_Validation_I_MSE = []

    for degree in range(1,max_degree+1):
        lambda_I_MSE = []
        print("Lasso k =",k,"degree =",degree)

        for lmbda in lambda_I:
            fold_MSE = []
            kfold = KFold(n_splits=k,shuffle=True,random_state=2026)

            for train_index,test_index in kfold.split(x):
                x_train = x[train_index]
                x_test = x[test_index]
                y_train = y[train_index]
                y_test = y[test_index]

                X_train = torch.vander(x_train,degree+1,increasing=True)
                X_test = torch.vander(x_test,degree+1,increasing=True)

                mean_X = torch.mean(X_train[:,1:],dim=0)
                std_X = torch.std(X_train[:,1:],dim=0,correction=0)

                X_train_scaled = (X_train[:,1:]-mean_X)/std_X
                X_test_scaled = (X_test[:,1:]-mean_X)/std_X

                y_mean = torch.mean(y_train)
                y_train_centered = y_train-y_mean

                grad_Lasso = lambda theta: Lasso_Gradient_I(X_train_scaled,y_train_centered,theta,lmbda)

                theta0 = torch.zeros(X_train_scaled.shape[1],dtype=torch.float64)

                theta,steps = optimise_Lasso_I(grad_Lasso,theta0,0.01,10000)

                print("steps =",steps)
                y_pred = X_test_scaled@theta+y_mean
                fold_MSE.append(MSE(y_test,y_pred).item())

            lambda_I_MSE.append(sum(fold_MSE)/len(fold_MSE))

        Lasso_Cross_Validation_I_MSE.append(lambda_I_MSE)

    return torch.tensor(Lasso_Cross_Validation_I_MSE,dtype=torch.float64)


lambda_Lasso_I = torch.logspace(-4,0,10,dtype=torch.float64)

Lasso_results = {}
Lasso_best_lambda = {}

for k in k_values:
    Lasso_MSE = Lasso_Cross_Validation_I(x,y,k,lambda_Lasso_I)
    best_MSE,best_lambda_index = torch.min(Lasso_MSE,dim=1)
    best_lambdas = lambda_Lasso_I[best_lambda_index]

    Lasso_results[k] = best_MSE
    Lasso_best_lambda[k] = best_lambdas

    plt.plot(degrees,best_MSE,label=f"{k}-fold Lasso")

plt.xlabel("Polynomial degree")
plt.ylabel("Best CV MSE")
plt.yscale("log")
plt.title("Lasso Cross Validation")
plt.legend()
plt.show()


