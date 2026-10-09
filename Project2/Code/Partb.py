import torch
import  matplotlib.pyplot as plot 
from functions import sigmoid, sigmoidDerivative, linear,linearDerivative,MSE,MSEDerivative,Rungefunction
from NeuralNetwork import NeuralNetwork
from sklearn.model_selection import train_test_split


torch.manual_seed(2026)

X = 2 * torch.rand(100, 1) - 1
y = Rungefunction(X) + 0.1 * torch.randn_like(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2026)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=2026)

mu = X_train.mean()
std = X_train.std()

X_train = (X_train - mu) / std
X_val = (X_val - mu) / std
X_test = (X_test - mu) / std


model = NeuralNetwork([1, 50, 1])
history = model.train(X_train, y_train, epochs=300, learning_rate=0.01, batch_size=10)

print("Train MSE:", MSE(y_train, model.predict(X_train)).item())
print("Test MSE:", MSE(y_test, model.predict(X_test)).item())

plot.plot(history)
plot.xlabel("Epoch")
plot.ylabel("Training MSE")
plot.title("Neural Network Training Loss")
plot.grid(True)
plot.show()

#Different structure of the network


architectures = [[1, 50, 1], [1, 100, 100, 1]]

for layers in architectures:
    torch.manual_seed(2026)

    model = NeuralNetwork(layers)
    history = model.train(X_train, y_train, epochs=300, learning_rate=0.01, batch_size=10)

    train_mse = MSE(y_train, model.predict(X_train)).item()
    val_mse = MSE(y_val, model.predict(X_val)).item()

    print(f"Architecture: {layers}")
    print(f"Train MSE: {train_mse:.6f}")
    print(f"Validation MSE: {val_mse:.6f}")

    plot.plot(history, label=str(layers))

plot.xlabel("Epoch")
plot.ylabel("Training MSE")
plot.title("Different Neural Network Structures")
plot.legend()
plot.grid(True)
plot.show()


#Different learning rate

learning_Rate_1B = [0.0001,0.001,0.01,0.1]


for L1B in learning_Rate_1B:

    torch.manual_seed(2026)


    model = NeuralNetwork([1,50,1])

    history = model.train(X_train,y_train,epochs= 300,learning_rate = L1B, batch_size = 10)

    train_mse = MSE(y_train,model.predict(X_train)).item()

    val_mse= MSE(y_val,model.predict(X_val)).item()

    print("Learning rate ", L1B)

    print("Train MSE", train_mse)

    print("Val MSE",val_mse)




#Adam


#RMS





