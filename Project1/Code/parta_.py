import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split



#Data
rng = np.random.default_rng(2026)


#Generate x values

n = 100
x = np.linspace(-1,1,n) 

#Define the Runge fuction 
def Runge_function(x):
    return 1 / (1+25*x**2) 


#True function values
y_true = Runge_function(x)


#Add noise
sigma = 0.1
noise = sigma* rng.standard_normal(n)

y = y_true +noise 


# Plot
plt.scatter(x, y, s=10, label="Noisy data")
plt.plot(x, y_true, label="Runge function")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

#Split the data

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state= 2026)

#Degree of polynomial
degree = 15 


#Design matrix
X_train = np.vander(x_train.ravel(), degree +1, increasing=True) #Train
X_test = np.vander(x_test.ravel(), degree +1, increasing=True) #Test


#Find the OLS parameters

theta = np.linalg.pinv(X_train) @ y_train

#Predictions

y_train_pred =X_train @ theta
y_test_pred = X_test @ theta 


#Compute the MSE 

def MSE(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)


mse_train= MSE(y_train,y_train_pred)
mse_test=MSE(y_test,y_test_pred)

print("Traing MSE",mse_train)
print("Tset MSE",mse_test)

#The traing MSE is 0.008422895082031664
#The test MSE is 0.06116904521970146


#Compute the R^2

def R2(y_true,y_pred):
    return 1-np.sum((y_true-y_pred)**2)/ np.sum((y_true-np.mean(y_true))**2)


R2_train = R2(y_train,y_train_pred)
R2_test = R2(y_test,y_test_pred)

print("Train R2",R2_train)
print("Test R2",R2_test)


#The train R2 is 0.9237913217985219
#The test R2 is 0.3335873280577112


degrees = range(1,16)

train_mse = []
test_mse = []
train_R2 = []
test_R2 =[]
theta_values = []

for degree in degrees:
    #Design matrix
    X_train = np.vander(x_train.ravel(), degree +1, increasing=True)
    X_test = np.vander(x_test.ravel(), degree +1, increasing=True)

    #OLS
    theta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
    theta_values.append(theta)
    #Predictions
    y_train_pred = X_train @ theta
    y_test_pred = X_test @ theta

    #MSE

    train_mse.append(MSE ( y_train,y_train_pred))
    test_mse.append(MSE(y_test,y_test_pred))

    #R2

    train_R2.append(R2(y_train,y_train_pred))
    test_R2.append (R2(y_test,y_test_pred))
plt.plot(degrees, train_mse, marker="o", label="Training MSE")
plt.plot(degrees, test_mse, marker="o", label="Test MSE")

plt.xlabel("Polynomial degree")
plt.ylabel("MSE")
plt.legend()
plt.show()

plt.plot(degrees, train_R2, marker="o", label="Training R2")
plt.plot(degrees, test_R2, marker="o", label="Test R2")

plt.xlabel("Polynomial degree")
plt.ylabel("R2")
plt.legend()
plt.show()
#Plot theta values

for j in range(6):

    theta_j = []

    degree_j = []

    for degree in degrees:

        if degree >= j:

            theta_j.append(theta_values[degree-1][j])

            degree_j.append(degree)

    plt.plot(degree_j, theta_j, marker="o", label="theta_" + str(j))

plt.xlabel("Polynomial degree")

plt.ylabel("Theta")

plt.legend()

plt.show()

#For the MSE
#when the degree of polynomial increases then the MSE decreases.
#But increases again for high degrees.


#For R2
#As the polynomial degree increases. R2 will also increases.
#however for very high degrees the R2 decreases.

#Try diiferent n and sigama and see the behavior 

#Try different n by and use fixed sigma

n_values = [25, 50, 100, 200, 500, 1000]

degree = 15

sigma = 0.1

train_mse_n = []
test_mse_n = []

train_R2_n = []
test_R2_n = []

for n in n_values:

    x = np.linspace(-1, 1, n)
    y_true = Runge_function(x)
    noise = sigma * rng.standard_normal(n)
    y = y_true + noise

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2026)
    X_train = np.vander(x_train.ravel(),degree + 1,increasing=True)
    X_test = np.vander(x_test.ravel(),degree + 1,increasing=True)
    theta = np.linalg.pinv(X_train) @ y_train

    #Predictions
    y_train_pred = X_train @ theta
    y_test_pred = X_test @ theta

    #MSE

    mse_train = MSE(y_train, y_train_pred)
    mse_test = MSE(y_test, y_test_pred)
    train_mse_n.append(mse_train)
    test_mse_n.append(mse_test)

    #R2
    R2_train = R2(y_train, y_train_pred)
    R2_test = R2(y_test, y_test_pred)
    train_R2_n.append(R2_train)
    test_R2_n.append(R2_test)

    print("n =", n,
          "Train MSE =", mse_train,
          "Test MSE =", mse_test,
          "Train R2 =", R2_train,
          "Test R2 =", R2_test)
    
plt.plot(n_values, train_mse_n, marker="o", label="Training MSE")
plt.plot(n_values, test_mse_n, marker="o", label="Test MSE")
plt.xlabel("Number of data points")
plt.ylabel("MSE")
plt.legend()
plt.show()
plt.plot(n_values, train_R2_n, marker="o", label="Training R2")
plt.plot(n_values, test_R2_n, marker="o", label="Test R2")
plt.xlabel("Number of data points")
plt.ylabel("R2")
plt.legend() 
plt.show()

#For different number of data points
#When the number of data points increases, the training and test results become more stable.
#With a small number of data points, there is a larger difference between training and test R2.
#When the number of data points is large, the training and test R2 become close.


#Try different sigma and use fixed n

sigma_values = [0, 0.05, 0.1, 0.2, 0.5]
n = 100
degree = 10
train_mse_sigma = []
test_mse_sigma = []
train_R2_sigma = []
test_R2_sigma = []

for sigma in sigma_values:

    # Generate x
    x = np.linspace(-1, 1, n)
    # Runge function
    y_true = Runge_function(x)

    # Add noise
    noise = sigma * rng.standard_normal(n)
    y = y_true + noise

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=2026)

    # Design matrices
    X_train = np.vander(x_train.ravel(),degree + 1,increasing=True)
    X_test = np.vander(x_test.ravel(),degree + 1,increasing=True)
    # OLS
    theta = np.linalg.lstsq(X_train,y_train,rcond=None)[0]
    # Predictions
    y_train_pred = X_train @ theta
    y_test_pred = X_test @ theta
    # MSE
    mse_train = MSE(y_train, y_train_pred)
    mse_test = MSE(y_test, y_test_pred)
    train_mse_sigma.append(mse_train)
    test_mse_sigma.append(mse_test)
    # R2
    R2_train = R2(y_train, y_train_pred)
    R2_test = R2(y_test, y_test_pred)
    train_R2_sigma.append(R2_train)
    test_R2_sigma.append(R2_test)
    print("Sigma =", sigma)
    print("Train MSE =", mse_train)
    print("Test MSE =", mse_test)
    print("Train R2 =", R2_train)
    print("Test R2 =", R2_test)
    print()

plt.plot(sigma_values, train_mse_sigma, marker="o", label="Training MSE")
plt.plot(sigma_values, test_mse_sigma, marker="o", label="Test MSE")
plt.xlabel("Sigma")
plt.ylabel("MSE")
plt.legend()
plt.show()
plt.plot(sigma_values, train_R2_sigma, marker="o", label="Training R2")
plt.plot(sigma_values, test_R2_sigma, marker="o", label="Test R2")
plt.xlabel("Sigma")
plt.ylabel("R2")
plt.legend()
plt.show()

#For different sigma values
#When sigma is small, both the training and test MSE are small.
#As sigma increases, both training and test MSE increase.
#At the same time, training and test R2 decrease.
#This means that more noise makes the model less accurate.




#Different n and sigma
#And increase the degree of polynomial

n_values = [25,50,100,200,300,500,1000]

sigma_values = [0,0.05,0.1,0.2,0.5]

degree_values = [15,25,35,50]

for degree in degree_values:
    #MSE plot
    for sigma in sigma_values:
        test_mse_values = []
        for n in n_values:
            #Generate x
            x = np.linspace(-1,1,n)
            #Runge function
            y_true = Runge_function(x)
            #Add noise
            noise = sigma * rng.standard_normal(n)
            y = y_true + noise
            #Train and test split
            x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2026)
            #Design matrix
            X_train = np.vander(x_train.ravel(),degree + 1,increasing=True)
            X_test = np.vander(x_test.ravel(),degree + 1,increasing=True)
            #OLS
            theta = np.linalg.pinv(X_train) @ y_train
            #Prediction
            y_train_pred = X_train @ theta
            y_test_pred = X_test @ theta
            #MSE
            mse_test = MSE(y_test,y_test_pred)
            test_mse_values.append(mse_test)
        plt.plot(n_values,test_mse_values,marker="o",label="Sigma = " + str(sigma))
    plt.xlabel("Number of data points")
    plt.ylabel("Test MSE")
    plt.title("Polynomial degree = " + str(degree))
    plt.legend()
    plt.show()
#R2 plots
for degree in degree_values:
    for sigma in sigma_values:
        test_R2_values = []
        for n in n_values:
            #Generate x
            x = np.linspace(-1,1,n)
            #Runge function
            y_true = Runge_function(x)
            #Add noise
            noise = sigma * rng.standard_normal(n)
            y = y_true + noise
            #Train and test split
            x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2026)
            #Design matrix
            X_train = np.vander(x_train.ravel(),degree + 1,increasing=True)
            X_test = np.vander(x_test.ravel(),degree + 1,increasing=True)

            #OLS
            theta = np.linalg.pinv(X_train) @ y_train

            #Prediction
            y_train_pred = X_train @ theta
            y_test_pred = X_test @ theta
            #R2
            R2_test = R2(y_test,y_test_pred)
            test_R2_values.append(R2_test)
        plt.plot(n_values,test_R2_values,marker="o",label="Sigma = " + str(sigma))
    plt.xlabel("Number of data points")
    plt.ylabel("Test R2")
    plt.title("Polynomial degree = " + str(degree))
    plt.legend()
    plt.show()

#For different polynomial degrees
#With high polynomial degrees and a small number of data points, the test MSE can become very large.
#The test R2 can also become very negative.
#This shows that high-degree polynomial models are unstable when there are not enough data points.
#Increasing the number of data points makes the model more stable.



#Scaling and centering
n = 100
sigma = 0.1
degree = 15
x = np.linspace(-1,1,n)

y_true = Runge_function(x)
noise = sigma * rng.standard_normal(n)
y = y_true + noise
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2026)
X_train = np.vander(x_train.ravel(),degree + 1,increasing=True)
X_test = np.vander(x_test.ravel(),degree + 1,increasing=True)
mean_X = np.mean(X_train[:,1:], axis=0)
std_X = np.std(X_train[:,1:], axis=0)


X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[:,1:] = (X_train[:,1:] - mean_X) / std_X
X_test_scaled[:,1:] = (X_test[:,1:] - mean_X) / std_X


theta_scaled = np.linalg.pinv(X_train_scaled) @ y_train

#Prediction
y_train_pred_scaled = X_train_scaled @ theta_scaled
y_test_pred_scaled = X_test_scaled @ theta_scaled

#MSE

mse_train_scaled = MSE(y_train,y_train_pred_scaled)

mse_test_scaled = MSE(y_test,y_test_pred_scaled)

#R2
R2_train_scaled = R2(y_train,y_train_pred_scaled)
R2_test_scaled = R2(y_test,y_test_pred_scaled)
#OLS without scaling

theta_unscaled = np.linalg.pinv(X_train) @ y_train

y_train_pred_unscaled = X_train @ theta_unscaled
y_test_pred_unscaled = X_test @ theta_unscaled

mse_train_unscaled = MSE(y_train,y_train_pred_unscaled)
mse_test_unscaled = MSE(y_test,y_test_pred_unscaled)

R2_train_unscaled = R2(y_train,y_train_pred_unscaled)
R2_test_unscaled = R2(y_test,y_test_pred_unscaled)

print("Unscaled Train MSE =", mse_train_unscaled)
print("Unscaled Test MSE =", mse_test_unscaled)
print("Unscaled Train R2 =", R2_train_unscaled)
print("Unscaled Test R2 =", R2_test_unscaled)

print()

print("Scaled Train MSE =", mse_train_scaled)
print("Scaled Test MSE =", mse_test_scaled)
print("Scaled Train R2 =", R2_train_scaled)
print("Scaled Test R2 =", R2_test_scaled)



#The scaled and unscaled OLS results are almost identical.
#Scaling does not change the prediction much, but it can improve numerical stability.
#The mean and standard deviation are computed only from the training data to avoid data leakage.