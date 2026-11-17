
import time
import torch
import matplotlib.pyplot as plt
from functions import analytical_solution
from neuralnetwork import NeuralNetwork,trial_solution,cost_function




#Random seed

torch.set_default_dtype(torch.float64)
torch.manual_seed(2026)


#Training settings

quick_D = False #True for a short test
n_epochs_D = 800 if not quick_D else 5
n_boundary_D = 100
t_max_D = 0.5
results_D = []


#Train one network

#LLM-assisted: ChatGPT (OpenAI), November 2026. Training and comparison function.
def run_model(name,method="pinn",nodes=32,layers=2,activation="tanh",n_points=400,placement="random",w_initial=1.0,w_boundary=1.0,optimizer_name="adam",lr=0.001):
    torch.manual_seed(2026)

    #Training points
    if placement=="grid":
        n = int(n_points**0.5)
        x = torch.linspace(0,1,n+2)[1:-1]
        t = torch.linspace(0,t_max_D,n+1)[1:]
        X,T = torch.meshgrid(x,t,indexing="ij")
        x_train = X.reshape(-1,1)
        t_train = T.reshape(-1,1)
    else:
        x_train = torch.rand(n_points,1)
        t_train = t_max_D*torch.rand(n_points,1)

    x_initial = torch.rand(n_boundary_D,1)
    t_boundary = t_max_D*torch.rand(n_boundary_D,1)

    model = NeuralNetwork(nodes,layers,activation)

    if optimizer_name=="sgd":
        optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    elif optimizer_name=="rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(),lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    start = time.perf_counter()

    for epoch in range(n_epochs_D):
        optimizer.zero_grad()
        loss = cost_function(model,x_train,t_train,x_initial,t_boundary,method,w_initial,w_boundary)
        loss.backward()
        optimizer.step()

        if not torch.isfinite(loss):
            break

    seconds = time.perf_counter()-start

    #Validation points
    x = torch.linspace(0,1,101)
    t = torch.linspace(0,t_max_D,51)
    X,T = torch.meshgrid(x,t,indexing="ij")
    xx = X.reshape(-1,1)
    tt = T.reshape(-1,1)

    with torch.no_grad():
        if method=="hard":
            u = trial_solution(model,xx,tt)
        else:
            u = model(xx,tt)

    u_exact = analytical_solution(xx,tt)
    error = torch.sqrt(torch.sum((u-u_exact)**2)/torch.sum(u_exact**2)).item()
    parameters = sum(p.numel() for p in model.parameters())

    print(name,method,"error:",round(error,6),"time:",round(seconds,2),"parameters:",parameters)

    results_D.append([name,method,layers,nodes,activation,n_points,placement,w_initial,w_boundary,optimizer_name,lr,error,seconds,parameters])

    return error,seconds,parameters


#D1 Hidden layers and nodes

nodes_values_D = [16,32,64]
layers_values_D = [1,2,3]

plt.figure()

for method in ["hard","pinn"]:
    for layers in layers_values_D:
        errors = []

        for nodes in nodes_values_D:
            error,seconds,parameters = run_model("D1",method,nodes,layers)
            errors.append(error)

        plt.plot(nodes_values_D,errors,marker="o",label=f"{method}, {layers} layers")

plt.xlabel("Nodes per layer")
plt.ylabel("Relative L2 Error")
plt.yscale("log")
plt.title("D1 - Hidden Layers and Nodes")
plt.legend()
plt.grid(True)
plt.show()


#D2 Activation functions

activation_values_D = ["tanh","sigmoid","relu","softplus"]

plt.figure()

for method in ["hard","pinn"]:
    errors = []

    for activation in activation_values_D:
        error,seconds,parameters = run_model("D2",method,activation=activation)
        errors.append(error)

    plt.plot(activation_values_D,errors,marker="o",label=method)

plt.ylabel("Relative L2 Error")
plt.yscale("log")
plt.title("D2 - Activation Functions")
plt.legend()
plt.grid(True)
plt.show()


#D3 Number and placement of training points

points_values_D = [100,400,900]

plt.figure()

for method in ["hard","pinn"]:
    for placement in ["random","grid"]:
        errors = []

        for n_points in points_values_D:
            error,seconds,parameters = run_model("D3",method,n_points=n_points,placement=placement)
            errors.append(error)

        plt.plot(points_values_D,errors,marker="o",label=f"{method}, {placement}")

plt.xlabel("Interior points")
plt.ylabel("Relative L2 Error")
plt.yscale("log")
plt.title("D3 - Collocation Points")
plt.legend()
plt.grid(True)
plt.show()


#D4 PINN loss weights

weight_values_D = [0.1,1.0,10.0]

plt.figure()

for kind in ["initial","boundary"]:
    errors = []

    for weight in weight_values_D:
        if kind=="initial":
            error,seconds,parameters = run_model("D4","pinn",w_initial=weight)
        else:
            error,seconds,parameters = run_model("D4","pinn",w_boundary=weight)

        errors.append(error)

    plt.plot(weight_values_D,errors,marker="o",label=kind)

plt.xlabel("Weight")
plt.ylabel("Relative L2 Error")
plt.xscale("log")
plt.yscale("log")
plt.title("D4 - PINN Loss Weights")
plt.legend()
plt.grid(True)
plt.show()


#D5 Optimizer and learning rate

learning_rates_D = [0.001,0.01]

plt.figure()

for method in ["hard","pinn"]:
    for optimizer_name in ["adam","sgd","rmsprop"]:
        errors = []

        for lr in learning_rates_D:
            error,seconds,parameters = run_model("D5",method,optimizer_name=optimizer_name,lr=lr)
            errors.append(error)

        plt.plot(learning_rates_D,errors,marker="o",label=f"{method}, {optimizer_name}")

plt.xlabel("Learning rate")
plt.ylabel("Relative L2 Error")
plt.xscale("log")
plt.yscale("log")
plt.title("D5 - Optimizer and Learning Rate")
plt.legend()
plt.grid(True)
plt.show()


#Computational cost

plt.figure()

for method in ["hard","pinn"]:
    parameters = []
    times = []

    for row in results_D:
        if row[0]=="D1" and row[1]==method:
            parameters.append(row[13])
            times.append(row[12])

    plt.scatter(parameters,times,label=method)

plt.xlabel("Number of parameters")
plt.ylabel("Training Time (s)")
plt.title("Computational Cost")
plt.legend()
plt.grid(True)
plt.show()
