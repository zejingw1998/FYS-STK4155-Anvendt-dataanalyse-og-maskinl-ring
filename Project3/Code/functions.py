
import math
import torch


#Initial condition

def initial_condition(x):
    return torch.sin(torch.pi*x)


#Analytical solution

def analytical_solution(x,t):
    t = torch.as_tensor(t,dtype=x.dtype,device=x.device)
    return torch.exp(-torch.pi**2*t)*torch.sin(torch.pi*x)


#Explicit finite difference

def finite_difference(nx,t_end):
    dx = 1.0/nx
    steps = max(1,math.ceil(t_end/(0.4*dx**2)))
    dt = t_end/steps

    x = torch.linspace(0,1,nx+1,dtype=torch.float64)
    u = initial_condition(x)
    u[0] = 0.0
    u[-1] = 0.0

    for i in range(steps):
        u_new = u.clone()
        u_new[1:-1] = u[1:-1]+dt/dx**2*(u[2:]-2*u[1:-1]+u[:-2])
        u = u_new

    return x,u


#Error functions

def MSE(y_true,y_pred):
    return torch.mean((y_true-y_pred)**2)


def MaxError(y_true,y_pred):
    return torch.max(torch.abs(y_true-y_pred))
