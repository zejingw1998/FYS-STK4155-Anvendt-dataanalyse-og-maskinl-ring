import torch 





#Runge function


def Rungefunction(x):
    return 1.0/(1.0+25.0*x**2)




#Activation functions


#Sigmoid
def sigmoid(z):
    return 1.0/(1.0+torch.exp(-z))



#Derivate sigmoid
def sigmoidDerivative(z):
    s = sigmoid(z)
    return s*(1.0 -s)


#Linear 

def linear(z):
    return z

#Derivate linear

def linearDerivative(z):
    return torch.ones_like(z)


#Cost function 



#MSE


def MSE(y_true,y_pred):
    return torch.mean((y_true-y_pred)**2)


def MSEDerivative(y_true,y_pred):
    n= y_true.shape[0]
    return 2.0*(y_pred-y_true)/n








if __name__ == "__main__":

    x = torch.tensor([[-1.0],[0.0],[1.0]])

    print("x:")
    print(x)

    print("\nRunge:")
    print(Rungefunction(x))

    print("\nSigmoid:")
    print(sigmoid(x))

    print("\nSigmoid derivative:")
    print(sigmoidDerivative(x))