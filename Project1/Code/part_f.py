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

tolerance_F = 0.1

print("Iterations needed to reach tolerance =",tolerance_F)

for methods in method:
    history = OLS_HISTORY_F[methods]
    reached = False
    for i in range(len(history)):
        difference = torch.norm(history[i]-theta_closed_F)
        if difference < tolerance_F:
            print(methods,"iterations =",i+1)
            reached = True
            break
    if reached == False:
        print(methods,"did not reach tolerance")


learning_rates_F = [0.001,0.01,0.05]

print("Learning rate comparison")

for learning_rate_test in learning_rates_F:
    print("Learning rate =",learning_rate_test)
    for methods in method:
        history = optimise(grad_OLS_F,theta_Task_1F,methods,learning_rate_test,iterations)
        theta_final = history[-1]
        difference = torch.norm(theta_final-theta_closed_F)
        print(methods,"Difference =",difference.item())

for learning_rate_test in learning_rates_F:
    history = optimise(grad_OLS_F,theta_Task_1F,"adam",learning_rate_test,iterations)
    MSE_VALUES_LR = []
    for theta in history:
        y_pred = PE.X_train@theta
        mse = PE.MSE(PE.y_train,y_pred)
        MSE_VALUES_LR.append(mse.item())
    plt.plot(MSE_VALUES_LR,label="learning rate = "+str(learning_rate_test))

plt.xlabel("Iteration")
plt.ylabel("Training MSE")
plt.yscale("log")
plt.title("OLS Adam learning rate comparison")
plt.legend()
plt.show()




#Ridge

degree_Ridge_F = PE.degree_first_test

X_train_Ridge_F = torch.vander(PE.x_train,degree_Ridge_F+1,increasing=True)
X_test_Ridge_F = torch.vander(PE.x_test,degree_Ridge_F+1,increasing=True)

mean_X_F = torch.mean(X_train_Ridge_F[:,1:],dim=0)
std_X_F = torch.std(X_train_Ridge_F[:,1:],dim=0,correction=0)

Xtr_scaled_F = (X_train_Ridge_F[:,1:]-mean_X_F)/std_X_F
Xtest_scaled_F = (X_test_Ridge_F[:,1:]-mean_X_F)/std_X_F

y_mean_F = torch.mean(PE.y_train)
y_train_centered_F = PE.y_train-y_mean_F

lambda_Ridge_F = 1e-8

theta_Ridge_0_F = torch.zeros(Xtr_scaled_F.shape[1],dtype=torch.float64)

grad_Ridge_F = lambda theta: PE.Ridge_AU_Gradient(Xtr_scaled_F,y_train_centered_F,theta,lambda_Ridge_F)

theta_Ridge_closed_F = PE.ridge_svd_1E(Xtr_scaled_F,y_train_centered_F,lambda_Ridge_F)


RIDGE_HISTORY_F = {}

for methods in method:
    history = optimise(grad_Ridge_F,theta_Ridge_0_F,methods,learning_rate_F,iterations)
    RIDGE_HISTORY_F[methods] = history
    MSE_VALUES_F_RIDGE = []

    for theta in history:
        y_pred = Xtr_scaled_F@theta+y_mean_F
        MSE_E_RIDGE = PE.MSE(PE.y_train,y_pred)
        MSE_VALUES_F_RIDGE.append(MSE_E_RIDGE.item())

    plt.plot(MSE_VALUES_F_RIDGE,label=methods)

plt.xlabel("Iteration")
plt.ylabel("Training MSE")
plt.yscale("log")
plt.title("Ridge optimizer comparison")
plt.legend()
plt.show()
print("Ridge optimizer results")

for methods in method:
    theta_final = RIDGE_HISTORY_F[methods][-1]
    difference = torch.norm(theta_final-theta_Ridge_closed_F)
    y_pred_test = Xtest_scaled_F@theta_final+y_mean_F
    test_MSE = PE.MSE(PE.y_test,y_pred_test)
    test_R2 = PE.R2(PE.y_test,y_pred_test)
    print(methods,"Difference =",difference.item(),"Test MSE =",test_MSE.item(),"Test R2 =",test_R2.item())



#Ridge tolerance test

tolerance_Ridge_F = 0.1

print("Ridge iterations needed to reach tolerance =",tolerance_Ridge_F)

for methods in method:
    history = RIDGE_HISTORY_F[methods]
    reached = False

    for i in range(len(history)):
        difference = torch.norm(history[i]-theta_Ridge_closed_F)

        if difference < tolerance_Ridge_F:
            print(methods,"iterations =",i+1)
            reached = True
            break

    if reached == False:
        print(methods,"did not reach tolerance")


#Ridge learning rate comparison

learning_rates_F_RIDGE = [0.001,0.01,0.05]

print("Ridge learning rate comparison")

for learning_rate_test_Ridge in learning_rates_F_RIDGE:
    print("Learning rate =",learning_rate_test_Ridge)

    for methods in method:
        history = optimise(grad_Ridge_F,theta_Ridge_0_F,methods,learning_rate_test_Ridge,iterations)
        theta_final = history[-1]
        difference = torch.norm(theta_final-theta_Ridge_closed_F)
        print(methods,"Difference =",difference.item())



for learning_rate_test_Ridge in learning_rates_F_RIDGE:
    history = optimise(grad_Ridge_F,theta_Ridge_0_F,"adam",learning_rate_test_Ridge,iterations)
    MSE_VALUES_LR_RIDGE = []
    for theta in history:
        y_pred = Xtr_scaled_F@theta+y_mean_F
        mse = PE.MSE(PE.y_train,y_pred)
        MSE_VALUES_LR_RIDGE.append(mse.item())
    plt.plot(MSE_VALUES_LR_RIDGE,label="learning rate = "+str(learning_rate_test_Ridge))

plt.xlabel("Iteration")
plt.ylabel("Training MSE")
plt.yscale("log")
plt.title("Ridge Adam learning rate comparison")
plt.legend()
plt.show()
#Structure

"""
optimise()

OLS
→ optimizer comparison
→ final Difference / Test MSE / Test R2
→ tolerance test
→ learning-rate comparison
→ Adam learning-rate figure

Ridge
→ scaling / centering
→ optimizer comparison
→ final Difference / Test MSE / Test R2
→ tolerance test
→ learning-rate comparison
"""

"""
OLS optimizer results
plain Difference = 2.165186311179708 Test MSE = 0.027674597212556145 Test R2 = 0.6829961327936647
momentum Difference = 0.44953710463287744 Test MSE = 0.037147630018420356 Test R2 = 0.5744855008026429
adagrad Difference = 1.7373779592749174 Test MSE = 0.023981074702621688 Test R2 = 0.7253042795128388
rmsprop Difference = 0.012247428071207267 Test MSE = 0.04398676921769729 Test R2 = 0.4961452974066717
adam Difference = 1.9242852429771378e-10 Test MSE = 0.0429821551751931 Test R2 = 0.5076528374831404
Iterations needed to reach tolerance = 0.1
plain did not reach tolerance
momentum did not reach tolerance
adagrad did not reach tolerance
rmsprop iterations = 1364
adam iterations = 862
Learning rate comparison
Learning rate = 0.001
plain Difference = 3.239732341004587
momentum Difference = 2.1655862544441384
adagrad Difference = 3.507209308960997
rmsprop Difference = 0.0012247229018444105
adam Difference = 0.00023933626914118846
Learning rate = 0.01
plain Difference = 2.165186311179708
momentum Difference = 0.44953710463287744
adagrad Difference = 1.7373779592749174
rmsprop Difference = 0.012247428071207267
adam Difference = 1.9242852429771378e-10
Learning rate = 0.05
plain Difference = 0.7857676013508973
momentum Difference = 0.006823692128748889
adagrad Difference = 0.00019427191746718917
rmsprop Difference = 0.0612372229296131
adam Difference = 9.127492918148831e-05
Ridge optimizer results
Ridge iterations needed to reach tolerance = 0.1
plain did not reach tolerance
momentum iterations = 1220
adagrad iterations = 5998
rmsprop iterations = 471
adam iterations = 291
Ridge learning rate comparison
Learning rate = 0.001
plain Difference = 0.5178981791926202
momentum Difference = 0.1240467018128546
adagrad Difference = 0.9138454954539499
rmsprop Difference = 0.001118030231981842
adam Difference = 7.809348005390294e-11
Learning rate = 0.01
plain Difference = 0.1240435529627746
momentum Difference = 1.8446138931079968e-05
adagrad Difference = 0.02126565020064803
rmsprop Difference = 0.011180336130718374
adam Difference = 0.0020882333030381624
Learning rate = 0.05
plain Difference = 0.0025570692294320536
momentum Difference = 6.0844547423858416e-15
adagrad Difference = 2.8224742913489706e-08
rmsprop Difference = 0.0559016956807135
adam Difference = 0.0004943912943296487
plain Difference = 0.1240435529627746 Test MSE = 0.038448945720533895 Test R2 = 0.559579335886929
momentum Difference = 1.8446138931079968e-05 Test MSE = 0.0429814457819548 Test R2 = 0.5076609633611138
adagrad Difference = 0.02126565020064803 Test MSE = 0.04132840942824244 Test R2 = 0.5265959784847205
rmsprop Difference = 0.011180336130718374 Test MSE = 0.04058018562964669 Test R2 = 0.5351666483979598
adam Difference = 0.0020882333030381624 Test MSE = 0.04248322412216095 Test R2 = 0.5133679368598627
"""