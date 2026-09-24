import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Rungefunction import Runge_function, MSE, R2

# Part f

torch.manual_seed(2026)

n = 100
sigma = 0.1

x = torch.linspace(-1,1,n,dtype=torch.float64)
y = Runge_function(x) + sigma*torch.randn(n,dtype=torch.float64)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=2026)

degree = 5

X_train = torch.vander(x_train,degree+1,increasing=True)
X_test = torch.vander(x_test,degree+1,increasing=True)


def OLS_Gradient(X,y,theta):
    n = len(y)
    return (2/n)*X.T@(X@theta-y)


def Ridge_Gradient(X,y,theta,lmbda):
    n = len(y)
    return (2/n)*X.T@(X@theta-y)+(2*lmbda/n)*theta


def optimiser_step(method,theta,g,state,t,learning_rate,beta=0.9,rho=0.99,beta1=0.9,beta2=0.999,epsilon=1e-8):

    if method == "plain":
        return theta-learning_rate*g,state

    if method == "momentum":
        if "v" not in state:
            state["v"] = torch.zeros_like(theta)

        state["v"] = beta*state["v"]+learning_rate*g

        return theta-state["v"],state

    if method == "adagrad":
        if "r" not in state:
            state["r"] = torch.zeros_like(theta)

        state["r"] = state["r"]+g**2

        return theta-learning_rate*g/(torch.sqrt(state["r"])+epsilon),state

    if method == "rmsprop":
        if "r" not in state:
            state["r"] = torch.zeros_like(theta)

        state["r"] = rho*state["r"]+(1-rho)*g**2

        return theta-learning_rate*g/(torch.sqrt(state["r"])+epsilon),state

    if method == "adam":
        if "m" not in state:
            state["m"] = torch.zeros_like(theta)

        if "r" not in state:
            state["r"] = torch.zeros_like(theta)

        state["m"] = beta1*state["m"]+(1-beta1)*g
        state["r"] = beta2*state["r"]+(1-beta2)*g**2

        m_hat = state["m"]/(1-beta1**t)
        r_hat = state["r"]/(1-beta2**t)

        return theta-learning_rate*m_hat/(torch.sqrt(r_hat)+epsilon),state

    raise ValueError("Unknown method")


def optimise(gradient_function,theta0,method,learning_rate,iterations=10000,tol=1e-8,**kw):

    theta = theta0.clone().detach()
    state = {}
    history = [theta.clone()]

    for t in range(1,iterations+1):

        gradient = gradient_function(theta)

        theta,state = optimiser_step(method,theta,gradient,state,t,learning_rate,**kw)

        history.append(theta.clone())

        if torch.norm(gradient) < tol:
            break

    return torch.stack(history),t


# Closed form

theta_OLS_closed = torch.linalg.pinv(X_train)@y_train

lmbda = 0.01

I = torch.eye(X_train.shape[1],dtype=torch.float64)

theta_Ridge_closed = torch.linalg.solve(X_train.T@X_train+lmbda*I,X_train.T@y_train)


# Gradient functions

grad_OLS = lambda theta: OLS_Gradient(X_train,y_train,theta)
grad_Ridge = lambda theta: Ridge_Gradient(X_train,y_train,theta,lmbda)

theta0 = torch.zeros(X_train.shape[1],dtype=torch.float64)

methods = ["plain","momentum","adagrad","rmsprop","adam"]

learning_rate = 0.01
iterations = 10000


# OLS

OLS_histories = {}
OLS_steps = {}

for method in methods:

    history,steps = optimise(grad_OLS,theta0,method,learning_rate,iterations=iterations)

    OLS_histories[method] = history
    OLS_steps[method] = steps


# Ridge

Ridge_histories = {}
Ridge_steps = {}

for method in methods:

    history,steps = optimise(grad_Ridge,theta0,method,learning_rate,iterations=iterations)

    Ridge_histories[method] = history
    Ridge_steps[method] = steps


# Compare with closed form

print("OLS")

for method in methods:

    theta_final = OLS_histories[method][-1]
    difference = torch.norm(theta_final-theta_OLS_closed)

    print(method,"iterations =",OLS_steps[method],"difference =",difference.item())


print("Ridge")

for method in methods:

    theta_final = Ridge_histories[method][-1]
    difference = torch.norm(theta_final-theta_Ridge_closed)

    print(method,"iterations =",Ridge_steps[method],"difference =",difference.item())


def iterations_to_target(history,theta_closed,tol=1e-4):

    distance = torch.norm(history-theta_closed,dim=1)
    indices = torch.where(distance < tol)[0]

    if len(indices) == 0:
        return None

    return indices[0].item()


print("OLS iterations to target")

for method in methods:

    target_iteration = iterations_to_target(OLS_histories[method],theta_OLS_closed)

    print(method,"=",target_iteration)


print("Ridge iterations to target")

for method in methods:

    target_iteration = iterations_to_target(Ridge_histories[method],theta_Ridge_closed)

    print(method,"=",target_iteration)


# OLS convergence

for method in methods:

    distance = torch.norm(OLS_histories[method]-theta_OLS_closed,dim=1)

    plt.plot(distance,label=method)

plt.xlabel("Iteration")
plt.ylabel("Distance to closed-form OLS")
plt.yscale("log")
plt.legend()
plt.show()


# Ridge convergence

for method in methods:

    distance = torch.norm(Ridge_histories[method]-theta_Ridge_closed,dim=1)

    plt.plot(distance,label=method)

plt.xlabel("Iteration")
plt.ylabel("Distance to closed-form Ridge")
plt.yscale("log")
plt.legend()
plt.show()


def MSE_history(history,X,y):

    mse_values = []

    for theta in history:

        y_pred = X@theta
        mse_values.append(MSE(y,y_pred).item())

    return mse_values


# OLS MSE

for method in methods:

    mse_values = MSE_history(OLS_histories[method],X_train,y_train)

    plt.plot(mse_values,label=method)

plt.xlabel("Iteration")
plt.ylabel("Training MSE")
plt.yscale("log")
plt.legend()
plt.show()


# Ridge MSE

for method in methods:

    mse_values = MSE_history(Ridge_histories[method],X_train,y_train)

    plt.plot(mse_values,label=method)

plt.xlabel("Iteration")
plt.ylabel("Training MSE")
plt.yscale("log")
plt.legend()
plt.show()


# Test results

print("OLS test results")

for method in methods:

    theta_final = OLS_histories[method][-1]
    y_pred = X_test@theta_final

    test_mse = MSE(y_test,y_pred)
    test_r2 = R2(y_test,y_pred)

    print(method,"MSE =",test_mse.item(),"R2 =",test_r2.item())


print("Ridge test results")

for method in methods:

    theta_final = Ridge_histories[method][-1]
    y_pred = X_test@theta_final

    test_mse = MSE(y_test,y_pred)
    test_r2 = R2(y_test,y_pred)

    print(method,"MSE =",test_mse.item(),"R2 =",test_r2.item())


# Learning rate comparison

learning_rates = [0.001,0.01,0.1]

for method in methods:

    for learning_rate_test in learning_rates:

        history,steps = optimise(grad_OLS,theta0,method,learning_rate_test,iterations=iterations)

        distance = torch.norm(history-theta_OLS_closed,dim=1)

        plt.plot(distance,label=f"{learning_rate_test}")

    plt.xlabel("Iteration")
    plt.ylabel("Distance to closed-form OLS")
    plt.yscale("log")
    plt.title(method)
    plt.legend()
    plt.show()


for method in methods:

    for learning_rate_test in learning_rates:

        history,steps = optimise(grad_Ridge,theta0,method,learning_rate_test,iterations=iterations)

        distance = torch.norm(history-theta_Ridge_closed,dim=1)

        plt.plot(distance,label=f"{learning_rate_test}")

    plt.xlabel("Iteration")
    plt.ylabel("Distance to closed-form Ridge")
    plt.yscale("log")
    plt.title(method)
    plt.legend()
    plt.show()