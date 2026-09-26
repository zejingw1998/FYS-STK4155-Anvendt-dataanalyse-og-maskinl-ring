import torch
import matplotlib.pyplot as plt
from Rungefunction import Runge_function, MSE, R2
import part_f as PF
import part_e as PE

print("G")
def Lasso_Gradient(X,y,theta,lmbda):
    n = len(y)
    gradient = (2/n)*X.T@(X@theta-y) + (lmbda/n)*torch.sign(theta)

    return gradient


# Define the Lasso gradient function

lmbda_Lasso = 0.01
grad_Lasso = lambda theta: Lasso_Gradient(PE.X_train,PE.y_train,theta,lmbda_Lasso)
theta0 = torch.zeros(PE.X_train.shape[1],dtype=torch.float64)


methods = ["plain","momentum","adagrad","rmsprop","adam"]
learning_rate = 0.01
iterations = 10000
Lasso_histories = {}
Lasso_steps = {}


for method in methods:
    history = PF.optimise(grad_Lasso,theta0,method,learning_rate,iterations)
    Lasso_histories[method] = history
    Lasso_steps[method] = len(history)


for method in methods:
    theta_final = Lasso_histories[method][-1]
    print(method)
    print("iterations =",Lasso_steps[method])
    print("theta =",theta_final)


print("Lasso test results")


for method in methods:
    theta_final = Lasso_histories[method][-1]
    y_pred = PE.X_test@theta_final
    test_mse = MSE(PE.y_test,y_pred)
    test_r2 = R2(PE.y_test,y_pred)
    print(method," G MSE =",test_mse.item(),"R2 =",test_r2.item())


for method in methods:

    history = Lasso_histories[method]

    mse_values = []

    for theta in history:

        y_pred = PE.X_train@theta

        mse = MSE(PE.y_train,y_pred)

        mse_values.append(mse.item())

    plt.plot(mse_values,label=method)

plt.xlabel("Iteration")
plt.ylabel("Training MSE")
plt.yscale("log")
plt.title("G Lasso convergence")
plt.legend()
plt.show()


theta_test = torch.tensor([-1.0,0.0,1.0],dtype=torch.float64,requires_grad=True)

lasso_penalty = torch.sum(torch.abs(theta_test))

lasso_penalty.backward()

print("theta =",theta_test)
print("Autograd gradient =",theta_test.grad)
print("torch.sign(theta) =",torch.sign(theta_test.detach()))



# Lasso cost convergence

for method in methods:
    history = Lasso_histories[method]
    cost_values = []

    for theta in history:
        y_pred = PE.X_train@theta
        mse = MSE(PE.y_train,y_pred)
        l1_penalty = (lmbda_Lasso/len(PE.y_train))*torch.sum(torch.abs(theta))
        cost = mse + l1_penalty
        cost_values.append(cost.item())

    plt.plot(cost_values,label=method)

plt.xlabel("Iteration")
plt.ylabel("Lasso cost")
plt.yscale("log")
plt.title("G Lasso cost convergence")
plt.legend()
plt.show()


# Check sparsity of the Lasso coefficients

print("Lasso sparsity")

for method in methods:
    theta_final = Lasso_histories[method][-1]
    near_zero = torch.sum(torch.abs(theta_final) < 1e-3)

    print(method)
    print("theta =",theta_final)
    print("coefficients close to zero =",near_zero.item())

# Test different lambda values

lambda_values = [0.0001,0.001,0.01,0.1,1.0]

lambda_MSE = []
lambda_R2 = []
lambda_zero = []

for lmbda in lambda_values:
    grad_Lasso_lambda = lambda theta: Lasso_Gradient(PE.X_train,PE.y_train,theta,lmbda)
    history = PF.optimise(grad_Lasso_lambda,theta0,"adagrad",learning_rate,iterations)
    theta_final = history[-1]
    y_pred = PE.X_test@theta_final
    test_mse = MSE(PE.y_test,y_pred)
    test_r2 = R2(PE.y_test,y_pred)
    near_zero = torch.sum(torch.abs(theta_final)<1e-3)
    lambda_MSE.append(test_mse.item())
    lambda_R2.append(test_r2.item())
    lambda_zero.append(near_zero.item())
    print("lambda =",lmbda,"MSE =",test_mse.item(),"R2 =",test_r2.item(),"close to zero =",near_zero.item())
    print("theta =",theta_final)

plt.plot(lambda_values,lambda_MSE,marker="o")
plt.xscale("log")
plt.xlabel("Lambda")
plt.ylabel("Test MSE")
plt.title("G Lasso Test MSE vs Lambda")
plt.show()

plt.plot(lambda_values,lambda_zero,marker="o")
plt.xscale("log")
plt.xlabel("Lambda")
plt.ylabel("Coefficients close to zero")
plt.title("G Lasso Sparsity vs Lambda")
plt.show()

# OLS Ridge Lasso comparison

theta_OLS = torch.linalg.pinv(PE.X_train)@PE.y_train
y_pred_OLS = PE.X_test@theta_OLS
OLS_MSE = MSE(PE.y_test,y_pred_OLS)
OLS_R2 = R2(PE.y_test,y_pred_OLS)

lambda_Ridge = 0.01
I = torch.eye(PE.X_train.shape[1],dtype=torch.float64)
theta_Ridge = torch.linalg.solve(PE.X_train.T@PE.X_train+lambda_Ridge*I,PE.X_train.T@PE.y_train)
y_pred_Ridge = PE.X_test@theta_Ridge
Ridge_MSE = MSE(PE.y_test,y_pred_Ridge)
Ridge_R2 = R2(PE.y_test,y_pred_Ridge)

theta_Lasso = Lasso_histories["adagrad"][-1]
y_pred_Lasso = PE.X_test@theta_Lasso
Lasso_MSE = MSE(PE.y_test,y_pred_Lasso)
Lasso_R2 = R2(PE.y_test,y_pred_Lasso)

print("OLS vs Ridge vs Lasso")
print("OLS MSE =",OLS_MSE.item(),"R2 =",OLS_R2.item())
print("Ridge MSE =",Ridge_MSE.item(),"R2 =",Ridge_R2.item())
print("Lasso MSE =",Lasso_MSE.item(),"R2 =",Lasso_R2.item())

models = ["OLS","Ridge","Lasso"]
mse_results = [OLS_MSE.item(),Ridge_MSE.item(),Lasso_MSE.item()]

plt.bar(models,mse_results)
plt.ylabel("Test MSE")
plt.title("G OLS vs Ridge vs Lasso")
plt.show()

"""
plain
iterations = 10000
theta = tensor([ 0.5520, -0.0171, -1.2026,  0.0569,  0.6485, -0.0460],
       dtype=torch.float64)
momentum
iterations = 10000
theta = tensor([ 0.6746, -0.0471, -2.4086, -0.0733,  2.0771,  0.2986],
       dtype=torch.float64)
adagrad
iterations = 10000
theta = tensor([ 5.8114e-01, -4.9176e-02, -1.5484e+00,  9.9363e-02,  1.0813e+00,
         2.5157e-05], dtype=torch.float64)
rmsprop
iterations = 10000
theta = tensor([ 6.7265e-01, -1.4320e-04, -2.4515e+00, -2.7723e-01,  2.1262e+00,
         5.0169e-01], dtype=torch.float64)
adam
iterations = 10000
theta = tensor([ 0.6776, -0.0051, -2.4465, -0.2822,  2.1312,  0.4967],
       dtype=torch.float64)
Lasso test results
plain  G MSE = 0.02782270533288876 R2 = 0.6812996005352367
momentum  G MSE = 0.03590670185432995 R2 = 0.5886999453317014
adagrad  G MSE = 0.024252485438048352 R2 = 0.7221953543107574
rmsprop  G MSE = 0.04068052241950211 R2 = 0.5340173218092856
adam  G MSE = 0.039649456906781594 R2 = 0.5458278552152493
theta = tensor([-1.,  0.,  1.], dtype=torch.float64, requires_grad=True)
Autograd gradient = tensor([-1.,  0.,  1.], dtype=torch.float64)
torch.sign(theta) = tensor([-1.,  0.,  1.], dtype=torch.float64)
Lasso sparsity
plain
theta = tensor([ 0.5520, -0.0171, -1.2026,  0.0569,  0.6485, -0.0460],
       dtype=torch.float64)
coefficients close to zero = 0
momentum
theta = tensor([ 0.6746, -0.0471, -2.4086, -0.0733,  2.0771,  0.2986],
       dtype=torch.float64)
coefficients close to zero = 0
adagrad
theta = tensor([ 5.8114e-01, -4.9176e-02, -1.5484e+00,  9.9363e-02,  1.0813e+00,
         2.5157e-05], dtype=torch.float64)
coefficients close to zero = 1
rmsprop
theta = tensor([ 6.7265e-01, -1.4320e-04, -2.4515e+00, -2.7723e-01,  2.1262e+00,
         5.0169e-01], dtype=torch.float64)
coefficients close to zero = 1
adam
theta = tensor([ 0.6776, -0.0051, -2.4465, -0.2822,  2.1312,  0.4967],
       dtype=torch.float64)
coefficients close to zero = 0
lambda = 0.0001 MSE = 0.02398545103004502 R2 = 0.7252541500491034 close to zero = 0
theta = tensor([ 0.5819, -0.0713, -1.5503,  0.1971,  1.0789, -0.0855],
       dtype=torch.float64)
lambda = 0.001 MSE = 0.024025368348281027 R2 = 0.724796909636456 close to zero = 0
theta = tensor([ 0.5818, -0.0684, -1.5501,  0.1831,  1.0793, -0.0729],
       dtype=torch.float64)
lambda = 0.01 MSE = 0.024252485438048352 R2 = 0.7221953543107574 close to zero = 1
theta = tensor([ 5.8114e-01, -4.9176e-02, -1.5484e+00,  9.9363e-02,  1.0813e+00,
         2.5157e-05], dtype=torch.float64)
lambda = 0.1 MSE = 0.024316558291730444 R2 = 0.7214614197842975 close to zero = 0
theta = tensor([ 0.5750, -0.0113, -1.4884,  0.0160,  1.0065,  0.0251],
       dtype=torch.float64)
lambda = 1.0 MSE = 0.04245569378112867 R2 = 0.5136832883175808 close to zero = 4
theta = tensor([ 4.6862e-01, -1.9294e-04, -5.7749e-01, -6.6028e-05, -1.5249e-05,
        -8.3482e-05], dtype=torch.float64)
OLS vs Ridge vs Lasso
OLS MSE = 0.042982155188718334 R2 = 0.507652837328213
Ridge MSE = 0.03661581399743928 R2 = 0.5805772872159534
Lasso MSE = 0.024252485438048352 R2 = 0.7221953543107574

"""