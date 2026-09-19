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

    y_train_PRED_1C = X_train @ thetha_1C
    y_test_PRED_1C = X_test @thetha_1C

    MSE_1C_train = MSE(y_train,y_train_PRED_1C)
    MSE_1C_test =  MSE(y_test, y_test_PRED_1C)
    train_MSE.append(MSE_1C_train.item())
    test_MSE.append(MSE_1C_test.item())



    """print("Train MSE", MSE_1C_train)"""
    """print("Test MSE", MSE_1C_test)"""

degrees = range(1, degree_test + 1)

plt.plot(degrees, train_MSE, label="Train MSE")
plt.plot(degrees, test_MSE, label="Test MSE")
plt.xlabel("Polynomial degree")
plt.ylabel("MSE")
plt.legend()
plt.show()

maxDegree = 30
n_bootstraps_1C = 100

Error_1C = torch.zeros(maxDegree, dtype=torch.float64)
Bias_1C = torch.zeros(maxDegree, dtype=torch.float64)
Variance_1C = torch.zeros(maxDegree, dtype=torch.float64)

# Define the polynomial with degree 1-30


for degree in range(1,maxDegree+1):

    X_test_boot_1C = torch.vander(x_test, degree + 1, increasing=True)

    predictions = torch.zeros((len(x_test), n_bootstraps_1C), dtype=torch.float64)

    for i in range(n_bootstraps_1C):

        # Bootstrap sampling
        bootstrap_1C = torch.randint(0, len(x_train), (len(x_train),))

        x_boot_1C = x_train[bootstrap_1C]
        y_boot_1C = y_train[bootstrap_1C]

        # Design matrix
        X_bootstrap_1C = torch.vander(x_boot_1C, degree + 1, increasing=True)

        # OLS
        theta_boot_1C = torch.linalg.pinv(X_bootstrap_1C) @ y_boot_1C

        # Prediction
        y_boot_1C_pred = X_test_boot_1C @ theta_boot_1C

        # Store prediction
        predictions[:, i] = y_boot_1C_pred

    # Mean prediction from the 100 bootstrap models
    mean_prediction = torch.mean(predictions, dim=1)
    # Error, Bias^2 and Variance
    Error_1C[degree - 1] = torch.mean((y_test[:, None] - predictions)**2)
    Bias_1C[degree - 1] = torch.mean((y_test - mean_prediction)**2)
    Variance_1C[degree - 1] = torch.mean(torch.var(predictions, dim=1, correction=0))
degrees = range(1, maxDegree + 1)

plt.plot(degrees, Error_1C, label="Test error")
plt.plot(degrees, Bias_1C, label="Bias squared")
plt.plot(degrees, Variance_1C, label="Variance")
plt.xlabel("Polynomial degree")
plt.ylabel("MSE")
plt.yscale("log")
plt.legend()
plt.show()


print("Maximum difference:", torch.max(torch.abs(Error_1C - Bias_1C - Variance_1C)))




#Test for different n values

n_diff = [50,100,500,1000,10000]



def bias_variance_n(n, maxDegree=30, n_bootstraps=100):
    torch.manual_seed(2026)

    x = torch.linspace(-1, 1, n, dtype=torch.float64)
    y_true = Runge_function(x)
    noise = 0.1 * torch.randn(n, dtype=torch.float64)
    y = y_true + noise

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2026)

    Error = torch.zeros(maxDegree, dtype=torch.float64)
    Bias = torch.zeros(maxDegree, dtype=torch.float64)
    Variance = torch.zeros(maxDegree, dtype=torch.float64)

    for degree in range(1, maxDegree + 1):

        X_test = torch.vander(x_test, degree + 1, increasing=True)
        predictions = torch.zeros((len(x_test), n_bootstraps), dtype=torch.float64)

        for i in range(n_bootstraps):

            bootstrap = torch.randint(0, len(x_train), (len(x_train),))

            x_boot = x_train[bootstrap]
            y_boot = y_train[bootstrap]

            X_boot = torch.vander(x_boot, degree + 1, increasing=True)

            theta = torch.linalg.pinv(X_boot) @ y_boot

            predictions[:, i] = X_test @ theta

        mean_prediction = torch.mean(predictions, dim=1)

        Error[degree - 1] = torch.mean((y_test[:, None] - predictions)**2)
        Bias[degree - 1] = torch.mean((y_test - mean_prediction)**2)
        Variance[degree - 1] = torch.mean(torch.var(predictions, dim=1, correction=0))

    return Error, Bias, Variance


for n in n_diff:

    Error, Bias, Variance = bias_variance_n(n)

    degrees = range(1, 31)

    plt.plot(degrees, Error, label="Test error")
    plt.plot(degrees, Bias, label="Bias squared")
    plt.plot(degrees, Variance, label="Variance")

    plt.xlabel("Polynomial degree")
    plt.ylabel("MSE")
    plt.title(f"Bias-Variance, n = {n}")
    plt.yscale("log")
    plt.legend()
    plt.show()



# Conclusion

# When we increase the polynomial degree, the test MSE starts to increase rapidly when the degree is greater than about 24.

# However, the training MSE remains very small and changes only slightly as the polynomial degree increases.

# This indicates overfitting: the model fits the training data very well, but performs poorly on unseen test data.


#For low polynomial degrees, the variance is small because the model is simple.

#As the polynomial degree increases, the variance increases because the model becomes more sensitive to changes in the training data.

#For very high polynomial degrees, the variance and test error increase dramatically, showing strong overfitting and numerical instability.

#Therefore, increasing model complexity can reduce underfitting at first, but too much complexity leads to high variance and poor generalization.