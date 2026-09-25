import torch
import matplotlib.pyplot as plt
import part_e as PE  



method = ["plain","momentum","adagrad","rmsprop","adam"]


learning_rate_F = 0.01

iterations = 10000

theta_Task_1F = torch.zeros(PE.X_train.shape[1],dtype=torch.float64)


grad_OLS_F = lambda theta: PE.OLS_Gradient( PE.X_train, PE.y_train, theta )



def optimise(grad_func,theta_Task_1F,method,learning_rate,iterations):

    theta = theta_Task_1F.clone()

    history = []


    # Momentum
    beta = 0.9
    velocity = torch.zeros_like(theta)


    # AdaGrad
    G = torch.zeros_like(theta)


    # RMSprop
    rho = 0.9
    S = torch.zeros_like(theta)


    # Adam
    beta1 = 0.9
    beta2 = 0.999
    m = torch.zeros_like(theta)
    v = torch.zeros_like(theta)


    epsilon = 1e-8


    for i in range(iterations):

        gradient = grad_func(theta)


        if method == "plain":

            theta = theta-learning_rate*gradient


        elif method == "momentum":

            velocity = beta*velocity+gradient

            theta = theta-learning_rate*velocity


        elif method == "adagrad":

            G = G+gradient**2

            theta = theta-learning_rate*gradient/(torch.sqrt(G)+epsilon)


        elif method == "rmsprop":

            S = rho*S+(1-rho)*gradient**2

            theta = theta-learning_rate*gradient/(torch.sqrt(S)+epsilon)


        elif method == "adam":

            m = beta1*m+(1-beta1)*gradient

            v = beta2*v+(1-beta2)*gradient**2


            m_hat = m/(1-beta1**(i+1))

            v_hat = v/(1-beta2**(i+1))


            theta = theta-learning_rate*m_hat/(torch.sqrt(v_hat)+epsilon)


        history.append(theta.clone())


    return history

OLS_HISTORY_F = {}

for methods in method:

    history = optimise(grad_OLS_F,theta_Task_1F,methods,learning_rate_F,iterations)

    OLS_HISTORY_F[methods] = history

    MSE_VALUES_F = []

    for theta in history:

        y_pred = PE.X_train@theta

        MSE_E = PE.MSE(PE.y_train,y_pred)

        MSE_VALUES_F.append(MSE_E.item())

    plt.plot(MSE_VALUES_F,label=methods)

plt.xlabel("Iteration")
plt.ylabel("Training MSE")
plt.yscale("log")
plt.title("OLS optimizer comparison")
plt.legend()
plt.show()



theta_closed_F = torch.linalg.pinv(PE.X_train)@PE.y_train

print("OLS optimizer results")

for methods in method:
    theta_final = OLS_HISTORY_F[methods][-1]
    difference = torch.norm(theta_final-theta_closed_F)
    y_pred_test = PE.X_test@theta_final
    test_MSE = PE.MSE(PE.y_test,y_pred_test)
    test_R2 = PE.R2(PE.y_test,y_pred_test)
    print(methods,"Difference =",difference.item(),"Test MSE =",test_MSE.item(),"Test R2 =",test_R2.item())



"""
OLS optimizer results
plain Difference = 2.165186311179708 Test MSE = 0.027674597212556145 Test R2 = 0.6829961327936647
momentum Difference = 0.44953710463287744 Test MSE = 0.037147630018420356 Test R2 = 0.5744855008026429
adagrad Difference = 1.7373779592749174 Test MSE = 0.023981074702621688 Test R2 = 0.7253042795128388
rmsprop Difference = 0.012247428071207267 Test MSE = 0.04398676921769729 Test R2 = 0.4961452974066717
adam Difference = 1.9242852429771378e-10 Test MSE = 0.0429821551751931 Test R2 = 0.5076528374831404
"""