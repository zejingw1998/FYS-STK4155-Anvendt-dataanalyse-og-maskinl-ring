import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Rungefunction import Runge_function, MSE, R2


#Gradient descent
#Analytical gradients

torch.manual_seed(2026)

n = 100
sigma = 0.1
x = torch.linspace(-1,1,n,dtype=torch.float64)
y = Runge_function(x) + sigma*torch.randn(n,dtype=torch.float64)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=2026)

degree_first_test = 5

X_train = torch.vander(x_train,degree_first_test+1,increasing=True)
X_test = torch.vander(x_test,degree_first_test+1,increasing=True)


#OLS Gradient

def OLS_Gradient(X,y,theta):
    n = len(y)
    gradient_1E = (2/n)*X.T@(X@theta-y)
    return gradient_1E


#Ridge Gradient

def Ridge_AU_Gradient(X,y,theta,lmbda):
    n = len(y)
    gradient = (2/n)*(X.T@(X@theta-y)+lmbda*theta)
    return gradient


#Ridge closed form

def ridge_svd_1E(X,y,lmbda):
    y = y.ravel()
    U,s,Vt = torch.linalg.svd(X,full_matrices=False)
    theta = Vt.T@((s/(s**2+lmbda))*(U.T@y))
    return theta


if __name__ == "__main__":

    #OLS Gradient Descent

    theta_1E = torch.zeros(X_train.shape[1],dtype=torch.float64)
    learning_rate = 0.01
    iterations = 10000
    history = []

    for i in range(iterations):
        gradient = OLS_Gradient(X_train,y_train,theta_1E)
        theta_1E = theta_1E-learning_rate*gradient
        history.append(MSE(y_train,X_train@theta_1E).item())

    print("GD theta =",theta_1E)

    theta_closed = torch.linalg.pinv(X_train)@y_train

    print("Closed form theta =",theta_closed)
    print("Difference =",torch.norm(theta_1E-theta_closed))

    plt.plot(range(iterations),history)
    plt.xlabel("Iteration")
    plt.ylabel("Training MSE")
    plt.yscale("log")
    plt.title("OLS Gradient Descent Convergence")
    plt.show()


    #Compare GD and Closed form

    y_pred_GD = X_test@theta_1E
    y_pred_closed = X_test@theta_closed

    sorted_index = torch.argsort(x_test)

    plt.scatter(x_test,y_test,label="Test data")
    plt.plot(x_test[sorted_index],y_pred_GD[sorted_index],label="Gradient Descent")
    plt.plot(x_test[sorted_index],y_pred_closed[sorted_index],label="Closed form")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    #Hessian

    H = (2/len(y_train))*X_train.T@X_train
    eigenvalues = torch.linalg.eigvalsh(H)
    lambda_max = torch.max(eigenvalues)

    print("Largest eigenvalue =",lambda_max.item())
    print("Approx maximum learning rate =",2/lambda_max.item())


    #Autograd

    theta_1E_Autograd_differation = torch.zeros(X_train.shape[1],dtype=torch.float64,requires_grad=True)

    y_pred_auto = X_train@theta_1E_Autograd_differation

    cost_auto = torch.mean((y_train-y_pred_auto)**2)

    cost_auto.backward()

    print("Automatic gradient =",theta_1E_Autograd_differation.grad)


    gradient_analytical = OLS_Gradient(X_train,y_train,theta_1E_Autograd_differation.detach())

    print("Analytical gradient =",gradient_analytical)
    print("Automatic gradient =",theta_1E_Autograd_differation.grad)
    print("Gradient difference =",torch.norm(gradient_analytical-theta_1E_Autograd_differation.grad))


    plt.plot(gradient_analytical.detach(),label="Analytical gradient")
    plt.plot(theta_1E_Autograd_differation.grad.detach(),label="Autograd gradient")
    plt.xlabel("Parameter index")
    plt.ylabel("Gradient")
    plt.title("OLS Analytical vs Autograd Gradient")
    plt.legend()
    plt.show()


    #Autograd Gradient Descent

    history_Auto = []

    for i in range(iterations):
        cost = torch.mean((y_train-X_train@theta_1E_Autograd_differation)**2)
        cost.backward()

        with torch.no_grad():
            theta_1E_Autograd_differation -= learning_rate*theta_1E_Autograd_differation.grad
            theta_1E_Autograd_differation.grad.zero_()
            history_Auto.append(cost.item())

    plt.plot(history,label="Analytical Gradient")
    plt.plot(history_Auto,label="Autograd")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.yscale("log")
    plt.title("OLS Analytical vs Autograd Convergence")
    plt.legend()
    plt.show()


    #Ridge

    lambda_1E_Ridge = 1e-8

    degrees_1E_AUTO_Ridge = [1,5,20,25,30,50]

    Ridge_Difference = []

    for degree in degrees_1E_AUTO_Ridge:

        X_train_Ridge = torch.vander(x_train,degree+1,increasing=True)
        X_test_Ridge = torch.vander(x_test,degree+1,increasing=True)

        mean_X = torch.mean(X_train_Ridge[:,1:],dim=0)
        std_X = torch.std(X_train_Ridge[:,1:],dim=0,correction=0)

        Xtr_scaled = (X_train_Ridge[:,1:]-mean_X)/std_X
        Xtest_scaled = (X_test_Ridge[:,1:]-mean_X)/std_X

        y_mean = torch.mean(y_train)
        y_train_centered = y_train-y_mean

        theta_Ridge_GD = torch.zeros(Xtr_scaled.shape[1],dtype=torch.float64)

        history_Ridge = []

        for i in range(iterations):
            gradient = Ridge_AU_Gradient(Xtr_scaled,y_train_centered,theta_Ridge_GD,lambda_1E_Ridge)
            theta_Ridge_GD = theta_Ridge_GD-learning_rate*gradient
            cost = MSE(y_train_centered,Xtr_scaled@theta_Ridge_GD)+(lambda_1E_Ridge/len(y_train_centered))*torch.sum(theta_Ridge_GD**2)
            history_Ridge.append(cost.item())

        theta_Ridge_closed = ridge_svd_1E(Xtr_scaled,y_train_centered,lambda_1E_Ridge)

        Difference = torch.norm(theta_Ridge_GD-theta_Ridge_closed)

        print("Degree =",degree)
        print("Ridge GD =",theta_Ridge_GD)
        print("Ridge closed form =",theta_Ridge_closed)
        print("Difference =",Difference)

        Ridge_Difference.append(Difference.item())


    plt.plot(degrees_1E_AUTO_Ridge,Ridge_Difference,marker="o")
    plt.xlabel("Polynomial degree")
    plt.ylabel("Difference")
    plt.yscale("log")
    plt.title("Ridge GD vs Closed-form")
    plt.show()


    #Ridge Hessian

    I = torch.eye(Xtr_scaled.shape[1],dtype=torch.float64)

    H_Ridge = (2/len(y_train_centered))*(Xtr_scaled.T@Xtr_scaled+lambda_1E_Ridge*I)

    eigenvalues_Ridge = torch.linalg.eigvalsh(H_Ridge)

    lambda_max_Ridge = torch.max(eigenvalues_Ridge)

    eta_max_Ridge = 2/lambda_max_Ridge

    print("Ridge largest eigenvalue =",lambda_max_Ridge.item())
    print("Ridge maximum learning rate =",eta_max_Ridge.item())


    #Autograd gradient Ridge

    theta_1E_Autograd_Ridge_differation = torch.randn(Xtr_scaled.shape[1],dtype=torch.float64,requires_grad=True)

    y_pred_auto_Ridge = Xtr_scaled@theta_1E_Autograd_Ridge_differation

    cost_auto_Ridge = torch.mean((y_train_centered-y_pred_auto_Ridge)**2)+(lambda_1E_Ridge/len(y_train_centered))*torch.sum(theta_1E_Autograd_Ridge_differation**2)

    cost_auto_Ridge.backward()

    gradient_Ridge_analytical = Ridge_AU_Gradient(Xtr_scaled,y_train_centered,theta_1E_Autograd_Ridge_differation.detach(),lambda_1E_Ridge)

    Difference_Auto_Norm = torch.norm(gradient_Ridge_analytical-theta_1E_Autograd_Ridge_differation.grad)

    print("Ridge analytical gradient =",gradient_Ridge_analytical)
    print("Ridge Autograd gradient =",theta_1E_Autograd_Ridge_differation.grad)
    print("Ridge gradient difference =",Difference_Auto_Norm)




"""


GD theta = tensor([ 0.5530, -0.0198, -1.2107,  0.0666,  0.6572, -0.0533],
       dtype=torch.float64)
Closed form theta = tensor([ 0.6814,  0.0209, -2.4873, -0.4195,  2.1854,  0.6311],
       dtype=torch.float64)
Difference = tensor(2.1652, dtype=torch.float64)
Largest eigenvalue = 2.303949599750079
Approx maximum learning rate = 0.8680745447803849
Automatic gradient = tensor([-5.9569e-01,  2.1579e-03, -6.8353e-02, -3.5757e-04, -2.8498e-02,
        -2.2466e-04], dtype=torch.float64)
Analytical gradient = tensor([-5.9569e-01,  2.1579e-03, -6.8353e-02, -3.5757e-04, -2.8498e-02,
        -2.2466e-04], dtype=torch.float64)
Automatic gradient = tensor([-5.9569e-01,  2.1579e-03, -6.8353e-02, -3.5757e-04, -2.8498e-02,
        -2.2466e-04], dtype=torch.float64)
Gradient difference = tensor(3.9748e-18, dtype=torch.float64)
Degree = 1
Ridge GD = tensor([0.0123], dtype=torch.float64)
Ridge closed form = tensor([0.0123], dtype=torch.float64)
Difference = tensor(4.3368e-17, dtype=torch.float64)
Degree = 5
Ridge GD = tensor([-0.0196, -0.7096, -0.0527,  0.5481,  0.1069], dtype=torch.float64)
Ridge closed form = tensor([ 0.0116, -0.7190, -0.1485,  0.5610,  0.1775], dtype=torch.float64)
Difference = tensor(0.1240, dtype=torch.float64)
Degree = 20
Ridge GD = tensor([-0.0136, -0.7821, -0.0351,  0.5416,  0.0009,  0.2904,  0.0590,  0.0446,
         0.0785, -0.0763,  0.0595, -0.1137,  0.0163, -0.1058, -0.0374, -0.0752,
        -0.0919, -0.0352, -0.1414,  0.0070], dtype=torch.float64)
Ridge closed form = tensor([  -0.4427,   -3.2267,    7.0663,   16.5359,  -47.3874,  -41.7250,
         143.9427,   68.7521, -166.5549, -102.1667,  -69.8566,   93.8129,
         247.7661,   19.6324,   63.3842,  -69.2079, -357.4217,  -24.2916,
         180.6379,   42.7940], dtype=torch.float64)
Difference = tensor(561.2506, dtype=torch.float64)
Degree = 25
Ridge GD = tensor([-0.0121, -0.7821, -0.0236,  0.5260, -0.0121,  0.2892,  0.0354,  0.0599,
         0.0600, -0.0524,  0.0562, -0.0909,  0.0341, -0.0924,  0.0043, -0.0776,
        -0.0252, -0.0578, -0.0498, -0.0387, -0.0672, -0.0232, -0.0767, -0.0126,
        -0.0785], dtype=torch.float64)
Ridge closed form = tensor([  -0.3926,   -3.2897,    5.6873,   17.0195,  -33.7052,  -38.3127,
          85.8843,   26.9447,  -71.8195,   30.7965,  -40.5258,  -32.0950,
          29.1973,  -80.0041,   68.7988,   74.2771,   33.0283,  130.3747,
         -43.8336,  -63.8694,  -79.8896, -225.2435,  -29.4891,  166.1724,
          80.0558], dtype=torch.float64)
Difference = tensor(391.6795, dtype=torch.float64)
Degree = 30
Ridge GD = tensor([-0.0135, -0.7778, -0.0187,  0.5198, -0.0126,  0.2828,  0.0313,  0.0576,
         0.0553, -0.0499,  0.0531, -0.0848,  0.0337, -0.0842,  0.0072, -0.0691,
        -0.0188, -0.0503, -0.0398, -0.0333, -0.0537, -0.0207, -0.0599, -0.0137,
        -0.0585, -0.0124, -0.0503, -0.0167, -0.0362, -0.0262],
       dtype=torch.float64)
Ridge closed form = tensor([  -0.3787,   -3.1849,    5.4234,   14.7627,  -32.8338,  -21.7744,
          93.8824,  -21.5060, -127.9973,   64.7912,   65.2077,   46.2264,
          27.6289, -140.6560,  -55.0044,  -65.1094,  -23.9083,  120.6880,
          61.5889,  144.3310,   86.7303,   -7.3984,    1.9631, -148.5376,
        -137.4758, -116.6193, -163.7905,   56.6775,  205.6988,   83.8181],
       dtype=torch.float64)
Difference = tensor(497.1366, dtype=torch.float64)
Degree = 50
Ridge GD = tensor([-0.0083, -0.7794, -0.0244,  0.5326, -0.0284,  0.2916,  0.0199,  0.0561,
         0.0556, -0.0616,  0.0660, -0.1038,  0.0565, -0.1066,  0.0357, -0.0909,
         0.0107, -0.0680, -0.0135, -0.0439, -0.0343, -0.0216, -0.0503, -0.0029,
        -0.0612,  0.0116, -0.0672,  0.0218, -0.0687,  0.0276, -0.0661,  0.0293,
        -0.0601,  0.0275, -0.0512,  0.0223, -0.0397,  0.0143, -0.0263,  0.0038,
        -0.0111, -0.0089,  0.0054, -0.0233,  0.0230, -0.0392,  0.0415, -0.0562,
         0.0607, -0.0743], dtype=torch.float64)
Ridge closed form = tensor([  -0.3680,   -3.1250,    5.2456,   14.1562,  -32.6407,  -23.5551,
         103.0663,   14.5217, -176.7288,  -59.0570,  137.5009,  153.4592,
          49.5935,  -19.0806, -118.4575, -149.7309,  -93.1017,  -76.9098,
          45.5816,   63.1102,  136.8906,  134.8817,  115.6558,  107.9229,
          16.0449,   22.7601,  -90.7227,  -65.2454, -148.6433, -118.0985,
        -137.8146, -122.5668,  -71.0289,  -84.5557,   20.5255,  -20.2616,
         102.3924,   50.9758,  147.8445,  110.7869,  143.5705,  143.2981,
          90.8804,  135.0514,    3.9327,   74.2213,  -93.5625,  -50.2589,
        -172.0689, -249.3379], dtype=torch.float64)
Difference = tensor(729.0572, dtype=torch.float64)
Ridge largest eigenvalue = 85.6780087253091
Ridge maximum learning rate = 0.02334321291723957
Ridge analytical gradient = tensor([  6.8126, -10.9187,   9.8755, -13.2958,  11.6553, -14.6925,  12.9196,
        -15.5848,  13.8717, -16.1664,  14.5961, -16.5413,  15.1440, -16.7740,
         15.5531, -16.9079,  15.8534, -16.9726,  16.0681, -16.9883,  16.2158,
        -16.9694,  16.3109, -16.9257,  16.3645, -16.8642,  16.3856, -16.7900,
         16.3810, -16.7067,  16.3563, -16.6170,  16.3158, -16.5230,  16.2631,
        -16.4262,  16.2009, -16.3277,  16.1316, -16.2287,  16.0571, -16.1297,
         15.9788, -16.0314,  15.8979, -15.9342,  15.8155, -15.8385,  15.7324,
        -15.7445], dtype=torch.float64)
Ridge Autograd gradient = tensor([  6.8126, -10.9187,   9.8755, -13.2958,  11.6553, -14.6925,  12.9196,
        -15.5848,  13.8717, -16.1664,  14.5961, -16.5413,  15.1440, -16.7740,
         15.5531, -16.9079,  15.8534, -16.9726,  16.0681, -16.9883,  16.2158,
        -16.9694,  16.3109, -16.9257,  16.3645, -16.8642,  16.3856, -16.7900,
         16.3810, -16.7067,  16.3563, -16.6170,  16.3158, -16.5230,  16.2631,
        -16.4262,  16.2009, -16.3277,  16.1316, -16.2287,  16.0571, -16.1297,
         15.9788, -16.0314,  15.8979, -15.9342,  15.8155, -15.8385,  15.7324,
        -15.7445], dtype=torch.float64)
Ridge gradient difference = tensor(3.8918e-14, dtype=torch.float64)
"""