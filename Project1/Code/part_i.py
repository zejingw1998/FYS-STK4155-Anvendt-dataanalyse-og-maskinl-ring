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


"""
steps = 558
steps = 517
steps = 511
steps = 10000
steps = 523
steps = 543
steps = 524
steps = 10000
steps = 558
steps = 569
Lasso k = 10 degree = 3
steps = 581
steps = 541
steps = 536
steps = 548
steps = 545
steps = 567
steps = 549
steps = 546
steps = 583
steps = 596
steps = 581
steps = 541
steps = 536
steps = 548
steps = 545
steps = 567
steps = 549
steps = 546
steps = 583
steps = 596
steps = 581
steps = 541
steps = 536
steps = 548
steps = 545
steps = 567
steps = 549
steps = 546
steps = 583
steps = 596
steps = 581
steps = 541
steps = 536
steps = 548
steps = 545
steps = 567
steps = 549
steps = 546
steps = 583
steps = 596
steps = 581
steps = 541
steps = 536
steps = 548
steps = 545
steps = 567
steps = 548
steps = 546
steps = 583
steps = 596
steps = 581
steps = 541
steps = 535
steps = 548
steps = 545
steps = 567
steps = 548
steps = 545
steps = 583
steps = 595
steps = 580
steps = 541
steps = 10000
steps = 547
steps = 544
steps = 566
steps = 547
steps = 10000
steps = 582
steps = 595
steps = 578
steps = 10000
steps = 10000
steps = 545
steps = 543
steps = 564
steps = 545
steps = 10000
steps = 580
steps = 592
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 558
steps = 538
steps = 10000
steps = 574
steps = 586
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
Lasso k = 10 degree = 4
steps = 7128
steps = 6118
steps = 6775
steps = 6809
steps = 7147
steps = 6809
steps = 6694
steps = 6670
steps = 6655
steps = 6449
steps = 7128
steps = 6118
steps = 6775
steps = 6809
steps = 7147
steps = 6809
steps = 6694
steps = 6670
steps = 6654
steps = 6449
steps = 7127
steps = 6117
steps = 6774
steps = 6808
steps = 7146
steps = 6808
steps = 6693
steps = 6669
steps = 6654
steps = 6448
steps = 7125
steps = 6116
steps = 6772
steps = 6806
steps = 7144
steps = 6807
steps = 6692
steps = 6667
steps = 6652
steps = 6447
steps = 7120
steps = 6111
steps = 6767
steps = 6801
steps = 7139
steps = 6802
steps = 6687
steps = 6664
steps = 6648
steps = 6442
steps = 7106
steps = 6099
steps = 6753
steps = 6787
steps = 7125
steps = 6788
steps = 6674
steps = 10000
steps = 6635
steps = 6430
steps = 7066
steps = 6064
steps = 6713
steps = 6749
steps = 7084
steps = 6751
steps = 6637
steps = 10000
steps = 6600
steps = 6397
steps = 6957
steps = 5969
steps = 6602
steps = 6643
steps = 6972
steps = 6648
steps = 6534
steps = 10000
steps = 6503
steps = 6303
steps = 6658
steps = 10000
steps = 6296
steps = 6350
steps = 10000
steps = 6366
steps = 10000
steps = 10000
steps = 6236
steps = 6046
steps = 10000
steps = 10000
steps = 10000
steps = 5562
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
Lasso k = 10 degree = 5
steps = 7123
steps = 6115
steps = 6788
steps = 6820
steps = 7184
steps = 6805
steps = 6705
steps = 6671
steps = 6654
steps = 6457
steps = 7123
steps = 6115
steps = 6787
steps = 6819
steps = 7184
steps = 6804
steps = 6704
steps = 6671
steps = 6654
steps = 6456
steps = 7122
steps = 6114
steps = 6786
steps = 6819
steps = 7183
steps = 6804
steps = 6704
steps = 6670
steps = 6653
steps = 6456
steps = 7121
steps = 6113
steps = 6783
steps = 6817
steps = 7181
steps = 6802
steps = 6702
steps = 6669
steps = 6652
steps = 6454
steps = 7116
steps = 6109
steps = 6776
steps = 6811
steps = 7174
steps = 6797
steps = 6697
steps = 6664
steps = 6647
steps = 6449
steps = 7102
steps = 6098
steps = 10000
steps = 6796
steps = 7157
steps = 6784
steps = 6682
steps = 6650
steps = 6635
steps = 6436
steps = 7065
steps = 6068
steps = 10000
steps = 6755
steps = 7109
steps = 6748
steps = 6642
steps = 10000
steps = 6600
steps = 6400
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 6374
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
Lasso k = 10 degree = 6
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
Lasso k = 10 degree = 7
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
Lasso k = 10 degree = 8
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
Lasso k = 10 degree = 9
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
Lasso k = 10 degree = 10
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
Lasso k = 10 degree = 11
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
Lasso k = 10 degree = 12
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
steps = 10000
"""