import torch 
import matplotlib.pyplot as plt 
from Rungefunction import Runge_function, MSE, R2
from sklearn.model_selection import KFold



print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


torch.manual_seed(2026)


n = 100

sigma = 0.1

x = torch.linspace(-1,1,n,dtype=torch.float64)
y = Runge_function(x) + sigma*torch.randn(n,dtype=torch.float64)


max_degree_1d = 30

def OLS_Cross_validation (x,y,k):


    Cros_Validation_MSE = []




    for degree in range(1,max_degree_1d+1):


        fold_MSE = []



        kfold = KFold(n_splits=k,shuffle=True, random_state= 2026)


        for train_index, test_index in kfold.split(x):



            x_train = x[train_index]
            x_test = x[test_index]

            y_train = y[train_index]

            y_test = y[test_index]


            X_train = torch.vander(x_train, degree +1, increasing= True)
            X_test = torch.vander(x_test,degree+1,increasing= True)

            thetha_1C_Cross_validation = torch.linalg.pinv(X_train)@y_train

            y_pred = X_test @ thetha_1C_Cross_validation


            fold_MSE.append(MSE(y_test,y_pred).item())
        Cros_Validation_MSE.append(sum(fold_MSE)/len(fold_MSE))    

    return Cros_Validation_MSE



k_value_1 = [5,10]
degrees = range(1,max_degree_1d+1)
OLS_results = {}

for k in k_value_1:
    MSE_FOLD = OLS_Cross_validation(x,y,k)
    OLS_results[k] = MSE_FOLD
    plt.plot(degrees,MSE_FOLD,label=f"{k}-fold OLS")

plt.xlabel("Polynomial degree")
plt.ylabel("MSE")
plt.yscale("log")
plt.legend()
plt.show()


def ridge_svd(X,y,lmbda):
    y = y.ravel()
    U,s,Vh = torch.linalg.svd(X,full_matrices=False)
    theta = Vh.T @ ((s/(s**2+lmbda))*(U.T@y))
    return theta

lambdas = torch.logspace(-8,4,50,dtype=torch.float64)

def Ridge_Cross_Validation(x,y,k,lambdas):
    Cross_Validation_2_MSE = []

    for degree in range(1,max_degree_1d+1):
        lambda_MSE = []

        for lmbda in lambdas:
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

                theta = ridge_svd(X_train_scaled,y_train_centered,lmbda)
                y_pred = X_test_scaled@theta+y_mean

                fold_MSE.append(MSE(y_test,y_pred).item())

            lambda_MSE.append(sum(fold_MSE)/len(fold_MSE))

        Cross_Validation_2_MSE.append(lambda_MSE)

    return torch.tensor(Cross_Validation_2_MSE,dtype=torch.float64)


k_value_2 = [5,10]
Ridge_results = {}
Ridge_best_lambda = {}

for k in k_value_2:
    Ridge_MSE = Ridge_Cross_Validation(x,y,k,lambdas)
    best_MSE,best_lambda_index = torch.min(Ridge_MSE,dim=1)
    best_lambdas = lambdas[best_lambda_index]

    Ridge_results[k] = best_MSE
    Ridge_best_lambda[k] = best_lambdas

    plt.plot(degrees,best_MSE,label=f"{k}-fold Ridge")

    for degree in range(1,max_degree_1d+1):
        print("k =",k,"Degree =",degree,"Best lambda =",best_lambdas[degree-1].item(),"Best MSE =",best_MSE[degree-1].item())

plt.xlabel("Polynomial degree")
plt.ylabel("Best CV MSE")
plt.yscale("log")
plt.legend()
plt.show()

#Compare 


# Compare OLS and Ridge

plt.plot(degrees,OLS_results[5],label="OLS 5-fold")
plt.plot(degrees,Ridge_results[5],label="Ridge 5-fold")
plt.xlabel("Polynomial degree")
plt.ylabel("CV MSE")
plt.yscale("log")
plt.legend()
plt.show()

plt.plot(degrees,OLS_results[10],label="OLS 10-fold")
plt.plot(degrees,Ridge_results[10],label="Ridge 10-fold")
plt.xlabel("Polynomial degree")
plt.ylabel("CV MSE")
plt.yscale("log")
plt.legend()
plt.show()



selected_degrees = [5,10,15,20,25,30]

Ridge_MSE = Ridge_Cross_Validation(x,y,5,lambdas)

for degree in selected_degrees:
    plt.plot(lambdas,Ridge_MSE[degree-1],label=f"Degree {degree}")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Lambda")
plt.ylabel("CV MSE")
plt.legend()
plt.show()


for k in [5,10]:
    best_index = torch.tensor(OLS_results[k]).argmin()
    print("OLS",k,"fold: Best degree =",best_index.item()+1,"Best MSE =",OLS_results[k][best_index])

for k in [5,10]:
    best_degree_index = torch.argmin(Ridge_results[k])
    print("Ridge",k,"fold: Best degree =",best_degree_index.item()+1,"Best lambda =",Ridge_best_lambda[k][best_degree_index].item(),"Best MSE =",Ridge_results[k][best_degree_index].item())



#Conclusion

#Both 5-fold and 10-fold cross-validation select polynomial degree 11 as the best model complexity.

#Ridge gives almost the same minimum MSE as OLS, because the optimal lambda is very small.

#The 5-fold and 10-fold results are very similar, showing that the model selection is stable.