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
max_degree_OLS = 30
max_degree_Ridge = 30
max_degree_Lasso = 12

degrees_OLS = range(1,max_degree_OLS+1)
degrees_Ridge = range(1,max_degree_Ridge+1)
degrees_Lasso = range(1,max_degree_Lasso+1)

k_values = [5,10]


#OLS
def OLS_Cross_validation_I(x,y,k):

    CV_MSE_I_OLS = []
    CV_SE_I_OLS = []

    for degree in range(1,max_degree_OLS+1):

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

        fold_MSE_tensor = torch.tensor(fold_MSE,dtype=torch.float64)

        mean_MSE = torch.mean(fold_MSE_tensor).item()
        SE_MSE = (torch.std(fold_MSE_tensor,correction=1)/torch.sqrt(torch.tensor(float(k)))).item()

        CV_MSE_I_OLS.append(mean_MSE)
        CV_SE_I_OLS.append(SE_MSE)

    return CV_MSE_I_OLS,CV_SE_I_OLS


OLS_results = {}
OLS_SE = {}

for k in k_values:
    OLS_MSE,OLS_standard_error = OLS_Cross_validation_I(x,y,k)

    OLS_results[k] = OLS_MSE
    OLS_SE[k] = OLS_standard_error

for k in k_values:

    best_index = torch.tensor(OLS_results[k]).argmin()

    print( "OLS",k,"fold:", "Best degree =",best_index.item()+1, "Best MSE =",OLS_results[k][best_index], "SE =",OLS_SE[k][best_index] )


for k in k_values:
    plt.plot(degrees_OLS,OLS_results[k],label=f"{k}-fold OLS")

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

    Ridge_MSE_all = []
    Ridge_SE_all = []

    for degree in range(1,max_degree_Ridge+1):

        lambda_I_MSE = []
        lambda_I_SE = []

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

            fold_MSE_tensor = torch.tensor(fold_MSE,dtype=torch.float64)

            mean_MSE = torch.mean(fold_MSE_tensor)
            SE_MSE = torch.std(fold_MSE_tensor,correction=1)/torch.sqrt(torch.tensor(float(k)))

            lambda_I_MSE.append(mean_MSE.item())
            lambda_I_SE.append(SE_MSE.item())

        Ridge_MSE_all.append(lambda_I_MSE)
        Ridge_SE_all.append(lambda_I_SE)

    return torch.tensor(Ridge_MSE_all,dtype=torch.float64),torch.tensor(Ridge_SE_all,dtype=torch.float64)

lambda_I = torch.logspace(-8,4,50,dtype=torch.float64)

Ridge_results = {}
Ridge_best_lambda = {}
Ridge_best_SE = {}

for k in k_values:

    Ridge_MSE,Ridge_SE = Ridge_Croos_Validation_I(x,y,k,lambda_I)

    best_MSE,best_lambda_index = torch.min(Ridge_MSE,dim=1)
    best_lambdas = lambda_I[best_lambda_index]

    degree_indices = torch.arange(max_degree_Ridge)
    best_SE = Ridge_SE[degree_indices,best_lambda_index]

    Ridge_results[k] = best_MSE
    Ridge_best_lambda[k] = best_lambdas
    Ridge_best_SE[k] = best_SE

    plt.plot(degrees_Ridge,best_MSE,label=f"{k}-fold Ridge")

for k in k_values:

    best_degree_index = torch.argmin(Ridge_results[k])

    print(
        "Ridge",k,"fold:",
        "Best degree =",best_degree_index.item()+1,
        "Best lambda =",Ridge_best_lambda[k][best_degree_index].item(),
        "Best MSE =",Ridge_results[k][best_degree_index].item(),
        "SE =",Ridge_best_SE[k][best_degree_index].item()
    )

plt.xlabel("Polynomial degree")
plt.ylabel("Best CV MSE")
plt.yscale("log")
plt.title("Ridge Cross Validation")
plt.legend()
plt.show()





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

    Lasso_MSE_all = []
    Lasso_SE_all = []

    for degree in range(1,max_degree_Lasso+1):

        lambda_I_MSE = []
        lambda_I_SE = []

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

                grad_Lasso = lambda theta: Lasso_Gradient_I(
                    X_train_scaled,
                    y_train_centered,
                    theta,
                    lmbda
                )

                theta0 = torch.zeros(
                    X_train_scaled.shape[1],
                    dtype=torch.float64
                )

                theta,steps = optimise_Lasso_I(
                    grad_Lasso,
                    theta0,
                    0.01,
                    10000
                )

                y_pred = X_test_scaled@theta+y_mean
                fold_MSE.append(MSE(y_test,y_pred).item())

            fold_MSE_tensor = torch.tensor(fold_MSE,dtype=torch.float64)

            mean_MSE = torch.mean(fold_MSE_tensor)
            SE_MSE = torch.std(fold_MSE_tensor,correction=1)/torch.sqrt(torch.tensor(float(k)))

            lambda_I_MSE.append(mean_MSE.item())
            lambda_I_SE.append(SE_MSE.item())

        Lasso_MSE_all.append(lambda_I_MSE)
        Lasso_SE_all.append(lambda_I_SE)

    return torch.tensor(Lasso_MSE_all,dtype=torch.float64),torch.tensor(Lasso_SE_all,dtype=torch.float64)


lambda_Lasso_I = torch.logspace(-4,0,10,dtype=torch.float64)

Lasso_results = {}
Lasso_best_lambda = {}
Lasso_best_SE = {}

for k in k_values:

    Lasso_MSE,Lasso_SE = Lasso_Cross_Validation_I(
        x,y,k,lambda_Lasso_I
    )

    best_MSE,best_lambda_index = torch.min(Lasso_MSE,dim=1)
    best_lambdas = lambda_Lasso_I[best_lambda_index]

    degree_indices = torch.arange(max_degree_Lasso)
    best_SE = Lasso_SE[degree_indices,best_lambda_index]

    Lasso_results[k] = best_MSE
    Lasso_best_lambda[k] = best_lambdas
    Lasso_best_SE[k] = best_SE

    plt.plot( degrees_Lasso, best_MSE, label=f"{k}-fold Lasso" )

for k in k_values:

    best_degree_index = torch.argmin(Lasso_results[k])

    print( "Lasso",k,"fold:", "Best degree =",best_degree_index.item()+1, "Best lambda =",Lasso_best_lambda[k][best_degree_index].item(), "Best MSE =",Lasso_results[k][best_degree_index].item(), "SE =",Lasso_best_SE[k][best_degree_index].item() )
"""
OLS 5 fold: Best degree = 11 Best MSE = 0.011538521621679163 SE = 0.00141824012545958
OLS 10 fold: Best degree = 11 Best MSE = 0.011114904678573439 SE = 0.0016079566339707821
Ridge 5 fold: Best degree = 11 Best lambda = 1e-08 Best MSE = 0.011538543811253944 SE = 0.0014182134164434034
Ridge 10 fold: Best degree = 11 Best lambda = 8.68511373751352e-06 Best MSE = 0.011101914593011639 SE = 0.0016187493352700498
Lasso k = 5 degree = 1
Lasso k = 5 degree = 2
Lasso k = 5 degree = 3
Lasso k = 5 degree = 4
Lasso k = 5 degree = 5
Lasso k = 5 degree = 6
Lasso k = 5 degree = 7
Lasso k = 5 degree = 8
Lasso k = 5 degree = 9
Lasso k = 5 degree = 10
Lasso k = 5 degree = 11
Lasso k = 5 degree = 12
Lasso k = 10 degree = 1
Lasso k = 10 degree = 2
Lasso k = 10 degree = 3
Lasso k = 10 degree = 4
Lasso k = 10 degree = 5
Lasso k = 10 degree = 6
Lasso k = 10 degree = 7
Lasso k = 10 degree = 8
Lasso k = 10 degree = 9
Lasso k = 10 degree = 10
Lasso k = 10 degree = 11
Lasso k = 10 degree = 12
Lasso 5 fold: Best degree = 8 Best lambda = 0.002154434690031882 Best MSE = 0.01971781563734627 SE = 0.0034910740405452935
Lasso 10 fold: Best degree = 8 Best lambda = 0.0001 Best MSE = 0.019634108273462618 SE = 0.0029507967712314384
"""