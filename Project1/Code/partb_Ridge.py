import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split


#Part b


#Define Runge function
def Runge_function(x):
    return 1 / (1 + 25*x**2)

#MSE
def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


# R2
def R2(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

#Generate data and parameteres

n=100
sigma = 0.1
rng = np.random.default_rng(2026)
x = np.linspace(-1,1,n)
y_true = Runge_function(x)
noise = sigma * rng.standard_normal(n)
y = y_true + noise
degree = 15

#Split 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2026)
X_train = np.vander(x_train.ravel(),degree + 1,increasing=True)
X_test = np.vander(x_test.ravel(),degree + 1,increasing=True)

# Scaling
mean_X = np.mean(X_train[:, 1:], axis=0)
std_X = np.std(X_train[:, 1:], axis=0)

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[:, 1:] = (X_train[:, 1:] - mean_X) / std_X
X_test_scaled[:, 1:] = (X_test[:, 1:] - mean_X) / std_X


# Remove intercept column
Xtr_scaled = X_train_scaled[:, 1:]
Xtest_scaled = X_test_scaled[:, 1:]


# Center y
y_mean = np.mean(y_train)
y_train_centered = y_train - y_mean


# Ridge using SVD
def ridge_svd(X, y, lmbda):
    y = y.ravel()
    U, s, Vt = np.linalg.svd(X,full_matrices=False)
    theta = Vt.T @ ((s / (s**2 + lmbda))* (U.T @ y))
    return theta 




lambdas = np.logspace(-8, 4, 100)
train_mse_ridge = []
test_mse_ridge = []


for lmbda in lambdas:
    theta = ridge_svd(Xtr_scaled,y_train_centered,lmbda)
    y_train_pred = (Xtr_scaled @ theta + y_mean)
    y_test_pred = (Xtest_scaled @ theta + y_mean)
    train_mse_ridge.append(MSE(y_train, y_train_pred))
    test_mse_ridge.append(MSE(y_test, y_test_pred))


plt.plot(lambdas,train_mse_ridge,label="Training MSE")
plt.plot(lambdas,test_mse_ridge,label="Test MSE")
plt.xscale("log")
plt.xlabel("Lambda")
plt.ylabel("MSE")
plt.legend()
plt.show()

#use different value of lambdas:

selected_lambdas = [1e-8, 1e-3, 0.1, 1, 100]
for lmbda in selected_lambdas:
    theta = ridge_svd(Xtr_scaled,y_train_centered,lmbda)
    plt.plot(range(1, len(theta)+1),np.abs(theta),marker="o",label=f"lambda={lmbda}")

plt.yscale("log")
plt.xlabel("Coefficient index")
plt.ylabel("|theta|")
plt.legend()
plt.show()



#Use different degree of polynomial


degrees = [1,5,20,25,30,50]

for degree in degrees:
    #Design matrix.
    X_train = np.vander(x_train.ravel(),degree+1,increasing =True)
    X_test = np.vander(x_test.ravel(),degree+1,increasing =True)

    #Scaling
    mean_X = np.mean(X_train[:, 1:], axis=0)
    std_X = np.std(X_train[:, 1:], axis=0)

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[:, 1:] = (X_train[:, 1:] - mean_X) / std_X
    X_test_scaled[:, 1:] = (X_test[:, 1:] - mean_X) / std_X

    Xtr_scaled = X_train_scaled[:, 1:]
    Xtest_scaled = X_test_scaled[:, 1:]

    # Center y
    y_mean = np.mean(y_train)
    y_train_centered = y_train - y_mean


    # MSE 
    train_mse_ridge = []
    test_mse_ridge = []
    
    for lmbda in lambdas:
        theta = ridge_svd(Xtr_scaled,y_train_centered,lmbda)
        y_train_pred = Xtr_scaled @ theta + y_mean
        y_test_pred = Xtest_scaled @ theta + y_mean
        train_mse_ridge.append(MSE(y_train, y_train_pred))
        test_mse_ridge.append( MSE(y_test, y_test_pred))

    plt.plot(lambdas,test_mse_ridge,label="Degree = " + str(degree))
    best_index = np.argmin(test_mse_ridge)
    best_lambda = lambdas[best_index]
    best_mse = test_mse_ridge[best_index]

    print("Degree =", degree,"Best lambda =", best_lambda,"Best test MSE =", best_mse)
plt.xscale("log")
plt.xlabel("Lambda")
plt.ylabel("Test MSE")
plt.legend()
plt.show()        

best_index = np.argmin(test_mse_ridge)
best_lambda = lambdas[best_index]
best_mse = test_mse_ridge[best_index]

print("Degree =", degree,"Best lambda =", best_lambda,"Best test MSE =", best_mse)

#Conclusion 
#Degree = 1 Best lambda = 1e-08 Best test MSE = 0.09848589596923046
#Degree = 5 Best lambda = 0.7564633275546291 Best test MSE = 0.015723478996849814
#Degree = 20 Best lambda = 0.03511191734215135 Best test MSE = 0.00983042833716741
#Degree = 25 Best lambda = 0.10722672220103231 Best test MSE = 0.010128556935654396
#Degree = 30 Best lambda = 0.18738174228603868 Best test MSE = 0.010681221089528578
#Degree = 50 Best lambda = 0.7564633275546291 Best test MSE = 0.01642546255016495



train_R2_ridge = []
test_R2_ridge = []


for lmbda in lambdas:
    theta = ridge_svd(Xtr_scaled,y_train_centered,lmbda)
    y_train_pred = Xtr_scaled @ theta + y_mean
    y_test_pred = Xtest_scaled @ theta + y_mean
    train_R2_ridge.append(R2(y_train, y_train_pred))
    test_R2_ridge.append(R2(y_test, y_test_pred))

plt.plot(lambdas,train_R2_ridge,label="Training R2")
plt.plot(lambdas,test_R2_ridge,label="Test R2")
plt.xscale("log")
plt.xlabel("Lambda")
plt.ylabel("R2")
plt.legend()
plt.show()

#The conclusion is very creepy, as R2 = -100 this means this is completetly uncorrect. 


#Compare with OLS and Ridge .

degree = 15 

X_train = np.vander(x_train.ravel(),degree+1,increasing =True)
X_test = np.vander(x_test.ravel(),degree+1,increasing =True)


mean_X = np.mean(X_train[:, 1:], axis=0)
std_X = np.std(X_train[:, 1:], axis=0)

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[:, 1:] = (X_train[:, 1:] - mean_X) / std_X
X_test_scaled[:, 1:] = (X_test[:, 1:] - mean_X) / std_X

Xtr_scaled = X_train_scaled[:, 1:]
Xtest_scaled = X_test_scaled[:, 1:]

y_mean = np.mean(y_train)
y_train_centered = y_train - y_mean

#OLS

theta_ols = np.linalg.pinv(Xtr_scaled) @ y_train_centered
y_test_pred_ols = Xtest_scaled @ theta_ols + y_mean
ols_mse = MSE(y_test, y_test_pred_ols)
ols_R2 = R2(y_test, y_test_pred_ols)

test_mse_ridge = []
test_R2_ridge = []

for lmbda in lambdas:
    theta = ridge_svd(Xtr_scaled,y_train_centered,lmbda)
    y_test_pred = Xtest_scaled @ theta + y_mean
    test_mse_ridge.append(MSE(y_test, y_test_pred))
    test_R2_ridge.append(R2(y_test, y_test_pred))
best_index = np.argmin(test_mse_ridge)

print("OLS Test MSE =", ols_mse)
print("OLS Test R2 =", ols_R2)

print("Best Ridge lambda =", lambdas[best_index])
print("Best Ridge Test MSE =", test_mse_ridge[best_index])
print("Best Ridge Test R2 =", test_R2_ridge[best_index])


# R2 for different polynomial degrees

degrees = [1, 5, 20, 25, 30, 50]

for degree in degrees:

    # Design matrix
    X_train = np.vander( x_train.ravel(), degree + 1, increasing=True )

    X_test = np.vander( x_test.ravel(), degree + 1, increasing=True )

    # Scaling
    mean_X = np.mean(X_train[:, 1:], axis=0)
    std_X = np.std(X_train[:, 1:], axis=0)

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[:, 1:] = ( X_train[:, 1:] - mean_X ) / std_X

    X_test_scaled[:, 1:] = ( X_test[:, 1:] - mean_X ) / std_X

    # Remove intercept
    Xtr_scaled = X_train_scaled[:, 1:]
    Xtest_scaled = X_test_scaled[:, 1:]

    # Center y
    y_mean = np.mean(y_train)
    y_train_centered = y_train - y_mean

    # Empty lists for this degree
    train_R2_ridge = []
    test_R2_ridge = []

    # Try different lambda values
    for lmbda in lambdas:

        theta = ridge_svd(
            Xtr_scaled,
            y_train_centered,
            lmbda
        )

        y_train_pred = Xtr_scaled @ theta + y_mean
        y_test_pred = Xtest_scaled @ theta + y_mean

        train_R2_ridge.append(
            R2(y_train, y_train_pred)
        )

        test_R2_ridge.append(
            R2(y_test, y_test_pred)
        )

    # Plot one figure for this degree
    plt.figure()

    plt.plot(
        lambdas,
        train_R2_ridge,
        label="Training R2"
    )

    plt.plot(
        lambdas,
        test_R2_ridge,
        label="Test R2"
    )

    plt.xscale("log")
    plt.xlabel("Lambda")
    plt.ylabel("R2")
    plt.title("Polynomial degree = " + str(degree))
    plt.legend()
    plt.show()