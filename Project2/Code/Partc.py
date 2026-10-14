import torch
from functions import Rungefunction,sigmoid, sigmoidDerivative, linear, linearDerivative, MSE, MSEDerivative,leakyRelu,leakyReluDerivative,relu,reluDerivative
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
from sklearn.model_selection import train_test_split




#Generate Data

torch.manual_seed(2026)

X = torch.linspace(-1, 1, 100).reshape(-1, 1)
y = Rungefunction(X) + 0.1 * torch.randn_like(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2026)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=2026)

mu = X_train.mean()
std = X_train.std()

X_train = (X_train - mu) / std
X_val = (X_val - mu) / std
X_test = (X_test - mu) / std


#Different Activation Functions

activations = ["sigmoid", "relu", "leaky_relu"]

for activation in activations:

    torch.manual_seed(2026)

    model = NeuralNetwork([1,50,1])
    model.set_activation(activation)

    history = model.train(X_train, y_train, epochs=300, learning_rate=0.01, batch_size=10)

    train_mse = MSE(y_train, model.predict(X_train)).item()
    val_mse = MSE(y_val, model.predict(X_val)).item()

    print("Activation Function:", activation)
    print("Train MSE:", train_mse)
    print("Validation MSE:", val_mse)

    plt.plot(history, label=activation)

plt.xlabel("Epoch")
plt.ylabel("Training MSE")
plt.ylim(0, 0.2)
plt.title("Different Activation Functions")
plt.legend()
plt.grid(True)
plt.show()

#Different Number of Hidden Layers

architectures = [[1,50,1], [1,50,50,1], [1,50,50,50,1]]

for layers in architectures:

    torch.manual_seed(2026)

    model = NeuralNetwork(layers)
    model.set_activation("relu")

    history = model.train(X_train, y_train, epochs=300, learning_rate=0.01, batch_size=10)

    train_mse = MSE(y_train, model.predict(X_train)).item()
    val_mse = MSE(y_val, model.predict(X_val)).item()

    print("Architecture:", layers)
    print("Train MSE:", train_mse)
    print("Validation MSE:", val_mse)

    plt.plot(history, label=str(layers))

plt.xlabel("Epoch")
plt.ylabel("Training MSE")
plt.title("Different Number of Hidden Layers")
plt.legend()
plt.grid(True)
plt.show()



#Different Number of Nodes

nodes_list = [10,25,50,100]

train_results = []
val_results = []

for nodes in nodes_list:

    torch.manual_seed(2026)

    model = NeuralNetwork([1,nodes,1])
    model.set_activation("relu")

    history = model.train(X_train, y_train, epochs=300, learning_rate=0.01, batch_size=10)

    train_mse = MSE(y_train, model.predict(X_train)).item()
    val_mse = MSE(y_val, model.predict(X_val)).item()

    train_results.append(train_mse)
    val_results.append(val_mse)

    print("Number of Nodes:", nodes)
    print("Train MSE:", train_mse)
    print("Validation MSE:", val_mse)

    plt.plot(history, label=f"{nodes} nodes")

plt.xlabel("Epoch")
plt.ylabel("Training MSE")
plt.title("Different Number of Nodes")
plt.legend()
plt.grid(True)
plt.show()

#Training and Validation MSE Comparison

plt.plot(nodes_list, train_results, marker="o", label="Training MSE")
plt.plot(nodes_list, val_results, marker="o", label="Validation MSE")
plt.xlabel("Number of Nodes")
plt.ylabel("MSE")
plt.title("Training and Validation MSE vs Number of Nodes")
plt.legend()
plt.grid(True)
plt.show()

#Overfitting Analysis



model = NeuralNetwork([1,50,50,50,1])
model.set_activation("relu")

train_history = []
val_history = []

epochs = 1000
learning_rate = 0.01
batch_size = 10

for epoch in range(epochs):

    indices = torch.randperm(X_train.shape[0])

    for batch in indices.split(batch_size):

        model.backpropagation(X_train[batch], y_train[batch])
        model.gradient(learning_rate)

    train_mse = MSE(y_train, model.predict(X_train)).item()
    val_mse = MSE(y_val, model.predict(X_val)).item()

    train_history.append(train_mse)
    val_history.append(val_mse)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Train MSE: {train_mse:.6f}, Validation MSE: {val_mse:.6f}")

best_epoch = val_history.index(min(val_history)) + 1

print("Final Train MSE:", train_history[-1])
print("Final Validation MSE:", val_history[-1])
print("Best Epoch:", best_epoch)
print("Best Validation MSE:", min(val_history))

plt.plot(train_history, label="Training MSE")
plt.plot(val_history, label="Validation MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Overfitting Analysis")
plt.legend()
plt.grid(True)
plt.show()


"""
Epoch 0, MSE: 0.103366
Epoch 100, MSE: 0.082740
Epoch 200, MSE: 0.081691
Activation Function: sigmoid
Train MSE: 0.08319370448589325
Validation MSE: 0.060923002660274506
Epoch 0, MSE: 0.187893
Epoch 100, MSE: 0.021904
Epoch 200, MSE: 0.019834
Activation Function: relu
Train MSE: 0.019200043752789497
Validation MSE: 0.018120339140295982
Epoch 0, MSE: 0.188676
Epoch 100, MSE: 0.021994
Epoch 200, MSE: 0.019886
Activation Function: leaky_relu
Train MSE: 0.019257934764027596
Validation MSE: 0.018163392320275307
Epoch 0, MSE: 0.187893
Epoch 100, MSE: 0.021904
Epoch 200, MSE: 0.019834
Architecture: [1, 50, 1]
Train MSE: 0.019200043752789497
Validation MSE: 0.018120339140295982
Epoch 0, MSE: 0.181676
Epoch 100, MSE: 0.017834
Epoch 200, MSE: 0.015757
Architecture: [1, 50, 50, 1]
Train MSE: 0.014388344250619411
Validation MSE: 0.01393283810466528
Epoch 0, MSE: 0.136168
Epoch 100, MSE: 0.018270
Epoch 200, MSE: 0.012672
Architecture: [1, 50, 50, 50, 1]
Train MSE: 0.010309194214642048
Validation MSE: 0.01137164793908596
Epoch 0, MSE: 0.229659
Epoch 100, MSE: 0.025621
Epoch 200, MSE: 0.022853
Number of Nodes: 10
Train MSE: 0.02255183644592762
Validation MSE: 0.02121472731232643
Epoch 0, MSE: 0.150888
Epoch 100, MSE: 0.023275
Epoch 200, MSE: 0.021024
Number of Nodes: 25
Train MSE: 0.01954273320734501
Validation MSE: 0.019011210650205612
Epoch 0, MSE: 0.187893
Epoch 100, MSE: 0.021904
Epoch 200, MSE: 0.019834
Number of Nodes: 50
Train MSE: 0.019200043752789497
Validation MSE: 0.018120339140295982
Epoch 0, MSE: 0.155380
Epoch 100, MSE: 0.020607
Epoch 200, MSE: 0.019201
Number of Nodes: 100
Train MSE: 0.018482986837625504
Validation MSE: 0.017110219225287437
Epoch 0, Train MSE: 0.133449, Validation MSE: 0.100265
Epoch 100, Train MSE: 0.016016, Validation MSE: 0.015824
Epoch 200, Train MSE: 0.011821, Validation MSE: 0.012207
Epoch 300, Train MSE: 0.009803, Validation MSE: 0.010951
Epoch 400, Train MSE: 0.008558, Validation MSE: 0.010053
Epoch 500, Train MSE: 0.007854, Validation MSE: 0.009383
Epoch 600, Train MSE: 0.007393, Validation MSE: 0.009495
Epoch 700, Train MSE: 0.007115, Validation MSE: 0.009505
Epoch 800, Train MSE: 0.006945, Validation MSE: 0.009359
Epoch 900, Train MSE: 0.006823, Validation MSE: 0.009250
Final Train MSE: 0.006744292564690113
Final Validation MSE: 0.009252681396901608
Best Epoch: 990
Best Validation MSE: 0.00905278418213129
"""
