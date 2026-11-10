import torch
import matplotlib.pyplot as plt
import functions 
from functions import finite_difference,analytical_solution,MSE,MaxError


#we consider that x in [0,1], L = 1 

#delta x = 1/10 and 1/100

#Consider the value for nx

nx_values_B = [10,100]

#Selected time points based on the analytical solution

t_values_B = [0.1,0.5]

for t in t_values_B:
    x_exact = torch.linspace(0,1,1001,dtype=torch.float64)

    u_exact = analytical_solution(x_exact,t)
    plt.figure()
    plt.plot(x_exact,u_exact,label="Analytical Solution for Heat equation",color="black",linestyle="--")
    #Finite Difference Solution

    for nx in nx_values_B:
        x,u = finite_difference(nx,t)

        plt.plot(x,u,label=f"Finite Difference nx={nx}",marker="o" if nx==10 else None,markersize=3)

        u_analytical = analytical_solution(x,t)

        mse = MSE(u_analytical,u)
        max_error = MaxError(u_analytical,u)

        print(f"t={t}, nx={nx}, MSE={mse.item():.8e}, Max Error={max_error.item():.8e}")

    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title(f"Finite Difference vs Analytical Solution, t={t}")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    #Absolute Error
    plt.figure()

    for nx in nx_values_B:
        x,u = finite_difference(nx,t)

        u_analytical = analytical_solution(x,t)

        error = torch.abs(u_analytical-u)

        plt.plot(x,error,label=f"nx={nx}")

    plt.xlabel("x")
    plt.ylabel("Absolute Error")
    plt.title(f"Absolute Error, t={t}")
    plt.legend()
    plt.grid(True)
    plt.show()   


    
#Maximum Error vs dx

nx_values_error_B = [10,20,40,80,100]
dx_values_B = [1.0/nx for nx in nx_values_error_B]

plt.figure()

for t in t_values_B:
    errors_B = []

    for nx in nx_values_error_B:
        x,u = finite_difference(nx,t)
        u_analytical = analytical_solution(x,t)

        max_error = MaxError(u_analytical,u)
        errors_B.append(max_error.item())

    plt.loglog(dx_values_B,errors_B,marker="o",label=f"t={t}")
    
    slope = torch.log(torch.tensor(errors_B[-1]/errors_B[0]))/torch.log(torch.tensor(dx_values_B[-1]/dx_values_B[0]))
    print(f"t={t}, Convergence order={slope.item():.4f}")



plt.xlabel("dx")
plt.ylabel("Maximum Error")
plt.title("Maximum Error vs dx")
plt.legend()
plt.grid(True)
plt.show()