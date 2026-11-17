import torch 
import torch.nn as nn 
from functions import initial_condition




#Neural Network

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        #Hidden layers
        self.network = nn.Sequential(nn.Linear(2,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,1))

    #Feed forward
    def forward(self,x,t):
        data = torch.cat([x,t],dim=1)
        return self.network(data)

#Hard-coded Initial and Boundary Conditions

def trial_solution(model,x,t):
    u = (1-t)*initial_condition(x)+x*(1-x)*t*model(x,t)
    return u



#PDE Residual

def pde_residual(model,x,t,method="hard"):

    #Enable gradients
    x = x.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)

    #Network solution
    if method=="hard":
        u = trial_solution(model,x,t)
    else:
        u = model(x,t)

    #First derivatives
    u_t = torch.autograd.grad(u,t,torch.ones_like(u),create_graph=True)[0]
    u_x = torch.autograd.grad(u,x,torch.ones_like(u),create_graph=True)[0]

    #Second derivative
    u_xx = torch.autograd.grad(u_x,x,torch.ones_like(u_x),create_graph=True)[0]

    #Heat equation residual
    residual = u_xx-u_t

    return residual





#Cost Function

def cost_function(model,x,t,x_initial,t_boundary,method="hard"):

    #PDE Loss
    residual = pde_residual(model,x,t,method)
    loss_pde = torch.mean(residual**2)

    if method=="hard":
        return loss_pde

    #Initial Condition Loss
    t_initial = torch.zeros_like(x_initial)
    u_initial = model(x_initial,t_initial)
    loss_initial = torch.mean((u_initial-initial_condition(x_initial))**2)

    #Boundary Condition Loss
    x_left = torch.zeros_like(t_boundary)
    x_right = torch.ones_like(t_boundary)

    u_left = model(x_left,t_boundary)
    u_right = model(x_right,t_boundary)

    loss_boundary = torch.mean(u_left**2+u_right**2)

    #Total Loss
    loss = loss_pde+loss_initial+loss_boundary

    return loss
