import torch 








#Runge function

def Runge_function(x):
    return 1/(1+25*x**2)


#Means Squared Error

def MSE(y_true,y_pred):
    return torch.mean((y_true - y_pred)**2)



#R2 score

def R2(y_true, y_pred):
    return 1 - torch.sum((y_true - y_pred)**2) / torch.sum( (y_true - torch.mean(y_true))**2 )


