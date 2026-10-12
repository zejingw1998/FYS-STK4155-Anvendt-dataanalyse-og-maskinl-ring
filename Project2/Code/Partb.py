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

    plot.plot(history, label=f"lr={L1B}")

plot.xlabel("Epoch")
plot.ylabel("Training MSE")
plot.title("Different Learning Rates")
plot.legend()
plot.grid(True)
plot.show()

#Adam

learning_Rate_Adam = [0.0001, 0.001, 0.01]

for lr in learning_Rate_Adam:

    torch.manual_seed(2026)

    model = NeuralNetwork([1,50,1])

    m_w = [torch.zeros_like(w) for w in model.weights]
    v_w = [torch.zeros_like(w) for w in model.weights]
    m_b = [torch.zeros_like(b) for b in model.biases]
    v_b = [torch.zeros_like(b) for b in model.biases]

    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    t = 0
    history = []

    for epoch in range(300):

        indices = torch.randperm(X_train.shape[0])

        for batch in indices.split(10):

            model.backpropagation(X_train[batch], y_train[batch])
            t += 1

            for i in range(len(model.weights)):

                gw = model.grad_weights[i]
                gb = model.grad_biases[i]

                m_w[i] = beta1 * m_w[i] + (1-beta1) * gw
                v_w[i] = beta2 * v_w[i] + (1-beta2) * gw**2

                m_b[i] = beta1 * m_b[i] + (1-beta1) * gb
                v_b[i] = beta2 * v_b[i] + (1-beta2) * gb**2

                mw_hat = m_w[i] / (1-beta1**t)
                vw_hat = v_w[i] / (1-beta2**t)

                mb_hat = m_b[i] / (1-beta1**t)
                vb_hat = v_b[i] / (1-beta2**t)

                model.weights[i] -= lr * mw_hat / (torch.sqrt(vw_hat) + eps)
                model.biases[i] -= lr * mb_hat / (torch.sqrt(vb_hat) + eps)

        history.append(MSE(y_train, model.predict(X_train)).item())

    train_mse = MSE(y_train, model.predict(X_train)).item()
    val_mse = MSE(y_val, model.predict(X_val)).item()

    print("Adam Learning Rate:", lr)
    print("Train MSE:", train_mse)
    print("Val MSE:", val_mse)

    plot.plot(history, label=f"Adam lr={lr}")

plot.xlabel("Epoch")
plot.ylabel("Training MSE")
plot.title("Adam Learning Rates")
plot.legend()
plot.grid(True)
plot.show()



#RMS





"""
Epoch 100, MSE: 0.088513
Epoch 200, MSE: 0.088330
Train MSE: 0.08829797804355621
Test MSE: 0.11863492429256439
"""