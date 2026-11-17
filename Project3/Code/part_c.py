
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from functions import finite_difference,analytical_solution,MSE,MaxError
from neuralnetwork import NeuralNetwork,trial_solution,cost_function




#Random seed

torch.manual_seed(2026)
torch.set_default_dtype(torch.float64)


#Training points

t_max_C = 0.5
n_points_C = 400
n_boundary_C = 100

#Interior points

x_train_C = torch.rand(n_points_C,1)
t_train_C = t_max_C*torch.rand(n_points_C,1)

#Initial condition points

x_initial_C = torch.rand(n_boundary_C,1)

#Boundary condition points

t_boundary_C = t_max_C*torch.rand(n_boundary_C,1)


#Training Neural Networks

models_C = {}
losses_C = {}
times_C = {}

for method in ["hard","pinn"]:

    torch.manual_seed(2026)
    model = NeuralNetwork()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    history = []
    start = time.time()

    for epoch in range(1500):
        optimizer.zero_grad()

        loss = cost_function(model,x_train_C,t_train_C,x_initial_C,t_boundary_C,method)
        loss.backward()
        optimizer.step()

        history.append(loss.item())

        if (epoch+1)%500==0:
            print(method,epoch+1,loss.item())

    models_C[method] = model
    losses_C[method] = history
    times_C[method] = time.time()-start

    print(method,"Training Time:",times_C[method])


#Training Loss

plt.figure()

plt.plot(losses_C["hard"],label="Hard NN")
plt.plot(losses_C["pinn"],label="PINN")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.show()


#Compare Solutions

t_values_C = [0.1,0.5]
x_plot_C = torch.linspace(0,1,1001).reshape(-1,1)

for t in t_values_C:

    t_plot_C = torch.full_like(x_plot_C,t)
    u_exact_C = analytical_solution(x_plot_C,t)

    x10,u10 = finite_difference(10,t)
    x100,u100 = finite_difference(100,t)

    with torch.no_grad():
        u_hard_C = trial_solution(models_C["hard"],x_plot_C,t_plot_C)
        u_pinn_C = models_C["pinn"](x_plot_C,t_plot_C)

    plt.figure()

    plt.plot(x_plot_C,u_exact_C,label="Analytical",color="black",linestyle="--")
    plt.plot(x10,u10,"o",label="FD nx=10")
    plt.plot(x100,u100,label="FD nx=100")
    plt.plot(x_plot_C,u_hard_C,label="Hard NN")
    plt.plot(x_plot_C,u_pinn_C,label="PINN")

    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title(f"Solutions at t={t}")
    plt.legend()
    plt.grid(True)
    plt.show()


    #Absolute Error

    plt.figure()

    plt.plot(x10,torch.abs(analytical_solution(x10,t)-u10),label="FD nx=10")
    plt.plot(x100,torch.abs(analytical_solution(x100,t)-u100),label="FD nx=100")
    plt.plot(x_plot_C,torch.abs(u_exact_C-u_hard_C),label="Hard NN")
    plt.plot(x_plot_C,torch.abs(u_exact_C-u_pinn_C),label="PINN")

    plt.xlabel("x")
    plt.ylabel("Absolute Error")
    plt.title(f"Absolute Error at t={t}")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("t =",t)
    print("Hard NN MSE:",MSE(u_exact_C,u_hard_C).item())
    print("PINN MSE:",MSE(u_exact_C,u_pinn_C).item())
    print("Hard NN Max Error:",MaxError(u_exact_C,u_hard_C).item())
    print("PINN Max Error:",MaxError(u_exact_C,u_pinn_C).item())


#Error over the domain

x_grid_C = torch.linspace(0,1,51)
t_grid_C = torch.linspace(0,0.5,51)

X_C,T_C = torch.meshgrid(x_grid_C,t_grid_C,indexing="ij")

u_exact_C = analytical_solution(X_C,T_C)

with torch.no_grad():
    u_hard_C = trial_solution(models_C["hard"],X_C.reshape(-1,1),T_C.reshape(-1,1)).reshape(51,51)
    u_pinn_C = models_C["pinn"](X_C.reshape(-1,1),T_C.reshape(-1,1)).reshape(51,51)

u_fd_C = torch.zeros_like(u_exact_C)

for j in range(len(t_grid_C)):
    x_fd,u_fd = finite_difference(50,t_grid_C[j].item())
    u_fd_C[:,j] = u_fd

for name,u in [("FD",u_fd_C),("Hard",u_hard_C),("PINN",u_pinn_C)]:

    error = torch.abs(u_exact_C-u)

    plt.figure()
    plt.pcolormesh(X_C,T_C,error,shading="auto")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(f"{name} Absolute Error")
    plt.show()

    print(name,"Domain Max Error:",torch.max(error).item())


#Points between grid points

t = 0.1
x_fd,u_fd = finite_difference(10,t)

u_interp_C = np.interp(x_plot_C.flatten().numpy(),x_fd.numpy(),u_fd.numpy())
u_exact_C = analytical_solution(x_plot_C,t)

with torch.no_grad():
    u_hard_C = trial_solution(models_C["hard"],x_plot_C,torch.full_like(x_plot_C,t))
    u_pinn_C = models_C["pinn"](x_plot_C,torch.full_like(x_plot_C,t))

plt.figure()
plt.plot(x_plot_C,torch.abs(u_exact_C.flatten()-torch.tensor(u_interp_C)),label="FD Interpolation")
plt.plot(x_plot_C,torch.abs(u_exact_C-u_hard_C),label="Hard NN")
plt.plot(x_plot_C,torch.abs(u_exact_C-u_pinn_C),label="PINN")

plt.xlabel("x")
plt.ylabel("Absolute Error")
plt.title("Error between grid points")
plt.legend()
plt.grid(True)
plt.show()


#Computational Cost

start = time.time()
finite_difference(100,0.5)
fd_time_C = time.time()-start

print("FD Time:",fd_time_C)
print("Hard NN Training Time:",times_C["hard"])
print("PINN Training Time:",times_C["pinn"])
