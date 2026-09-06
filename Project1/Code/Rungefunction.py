import numpy as np
import matplotlib.pyplot as plt 



rng = np.random.default_rng(2026)


#Generate x values

n = 100
x = np.linspace(-1,1,n)

#Define the Runge fuction 
def Runge_function(x):
    return 1 / (1+25*x)**2 


#True function values
y_true = Runge_function(x)


#Add noise
sigma = 0.1
noise = sigma* rng.standard_normal(n)

y = y_true +noise 


# Plot
plt.scatter(x, y, s=10, label="Noisy data")
plt.plot(x, y_true, label="Runge function")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
