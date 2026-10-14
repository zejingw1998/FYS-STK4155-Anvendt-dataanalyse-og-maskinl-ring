
import torch
import matplotlib.pyplot as plt
from functions import sigmoid, sigmoidDerivative, linear, linearDerivative, MSE, MSEDerivative, Rungefunction
from NeuralNetwork import NeuralNetwork
from sklearn.model_selection import train_test_split

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

model = NeuralNetwork([1, 50, 1])
history = model.train(X_train, y_train, epochs=300, learning_rate=0.01, batch_size=10)

print("Train MSE:", MSE(y_train, model.predict(X_train)).item())
print("Test MSE:", MSE(y_test, model.predict(X_test)).item())

plt.plot(history)
plt.xlabel("Epoch")
plt.ylabel("Training MSE")
plt.title("Neural Network Training Loss")
plt.grid(True)
plt.show()

# Different structure of the network

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

    plt.plot(history, label=str(layers))

plt.xlabel("Epoch")
plt.ylabel("Training MSE")
plt.title("Different Neural Network Structures")
plt.legend()
plt.grid(True)
plt.show()

# Different learning rate

learning_Rate_1B = [0.0001, 0.001, 0.01, 0.1]

for L1B in learning_Rate_1B:
    torch.manual_seed(2026)
    model = NeuralNetwork([1,50,1])
    history = model.train(X_train, y_train, epochs=300, learning_rate=L1B, batch_size=10)

    train_mse = MSE(y_train, model.predict(X_train)).item()
    val_mse = MSE(y_val, model.predict(X_val)).item()

    print("Learning rate ", L1B)
    print("Train MSE", train_mse)
    print("Val MSE", val_mse)

    plt.plot(history, label=f"lr={L1B}")

plt.xlabel("Epoch")
plt.ylabel("Training MSE")
plt.title("Different Learning Rates")
plt.ylim(0, 0.2)
plt.legend()
plt.grid(True)
plt.show()

# Adam

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

    plt.plot(history, label=f"Adam lr={lr}")

plt.xlabel("Epoch")
plt.ylabel("Training MSE")
plt.title("Adam Learning Rates")
plt.legend()
plt.grid(True)
plt.show()

# RMS

learning_Rate_RMS = [0.0001, 0.001, 0.01]

for lr in learning_Rate_RMS:
    torch.manual_seed(2026)
    model = NeuralNetwork([1,50,1])

    v_w = [torch.zeros_like(w) for w in model.weights]
    v_b = [torch.zeros_like(b) for b in model.biases]

    beta = 0.9
    eps = 1e-8
    history = []

    for epoch in range(300):
        indices = torch.randperm(X_train.shape[0])

        for batch in indices.split(10):
            model.backpropagation(X_train[batch], y_train[batch])

            for i in range(len(model.weights)):
                gw = model.grad_weights[i]
                gb = model.grad_biases[i]

                v_w[i] = beta * v_w[i] + (1-beta) * gw**2
                v_b[i] = beta * v_b[i] + (1-beta) * gb**2

                model.weights[i] -= lr * gw / (torch.sqrt(v_w[i]) + eps)
                model.biases[i] -= lr * gb / (torch.sqrt(v_b[i]) + eps)

        history.append(MSE(y_train, model.predict(X_train)).item())

    train_mse = MSE(y_train, model.predict(X_train)).item()
    val_mse = MSE(y_val, model.predict(X_val)).item()

    print("RMS Learning Rate:", lr)
    print("Train MSE:", train_mse)
    print("Val MSE:", val_mse)

    plt.plot(history, label=f"RMS lr={lr}")

plt.xlabel("Epoch")
plt.ylabel("Training MSE")
plt.title("RMSprop Learning Rates")
plt.legend()
plt.grid(True)
plt.show()

# Full Batch Gradient Descent

torch.manual_seed(2026)

model = NeuralNetwork([1,50,1])
history = model.train(X_train, y_train, epochs=300, learning_rate=0.01, batch_size=X_train.shape[0])

train_mse = MSE(y_train, model.predict(X_train)).item()
val_mse = MSE(y_val, model.predict(X_val)).item()

print("GD Train MSE:", train_mse)
print("GD Val MSE:", val_mse)

plt.plot(history, label="Full Batch GD")
plt.xlabel("Epoch")
plt.ylabel("Training MSE")
plt.title("Full Batch Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()

# OLS Comparison

degree = 11

X_ols_train = (X_train * std + mu).double()
X_ols_val = (X_val * std + mu).double()

X_design_train = torch.vander(X_ols_train.flatten(), N=degree+1, increasing=True)
X_design_val = torch.vander(X_ols_val.flatten(), N=degree+1, increasing=True)

beta = torch.linalg.pinv(X_design_train) @ y_train.double()

y_pred_train = X_design_train @ beta
y_pred_val = X_design_val @ beta

train_mse = MSE(y_train.double(), y_pred_train).item()
val_mse = MSE(y_val.double(), y_pred_val).item()

print("OLS Degree:", degree)
print("OLS Train MSE:", train_mse)
print("OLS Validation MSE:", val_mse)

# Backpropagation Gradient Check

torch.manual_seed(2026)

check_model = NeuralNetwork([1,50,1])
check_model.backpropagation(X_train[:8], y_train[:8])

W = [w.clone().detach().requires_grad_(True) for w in check_model.weights]
B = [b.clone().detach().requires_grad_(True) for b in check_model.biases]

a = X_train[:8]

for i in range(len(W)):
    z = a @ W[i] + B[i]
    a = linear(z) if i == len(W)-1 else sigmoid(z)

loss = MSE(y_train[:8], a)

grad_auto = torch.autograd.grad(loss, W+B)
grad_manual = check_model.grad_weights + check_model.grad_biases

max_error = max((g1-g2).abs().max().item() for g1,g2 in zip(grad_manual,grad_auto))

print("Maximum Gradient Difference:", max_error)

# Final Model Comparison

X_full = torch.cat([X_train, X_val], dim=0)
y_full = torch.cat([y_train, y_val], dim=0)

# Final Neural Network with RMSprop

torch.manual_seed(2026)

final_model = NeuralNetwork([1,50,1])

v_w = [torch.zeros_like(w) for w in final_model.weights]
v_b = [torch.zeros_like(b) for b in final_model.biases]

lr = 0.01
beta = 0.9
eps = 1e-8

for epoch in range(300):
    indices = torch.randperm(X_full.shape[0])

    for batch in indices.split(10):
        final_model.backpropagation(X_full[batch], y_full[batch])

        for i in range(len(final_model.weights)):
            gw = final_model.grad_weights[i]
            gb = final_model.grad_biases[i]

            v_w[i] = beta * v_w[i] + (1-beta) * gw**2
            v_b[i] = beta * v_b[i] + (1-beta) * gb**2

            final_model.weights[i] -= lr * gw / (torch.sqrt(v_w[i]) + eps)
            final_model.biases[i] -= lr * gb / (torch.sqrt(v_b[i]) + eps)

nn_train_mse = MSE(y_full, final_model.predict(X_full)).item()
nn_test_mse = MSE(y_test, final_model.predict(X_test)).item()

print("Final NN Train MSE:", nn_train_mse)
print("Final NN Test MSE:", nn_test_mse)

# Final OLS Comparison

degree = 11

X_ols_full = (X_full * std + mu).double()
X_ols_test = (X_test * std + mu).double()

X_design_full = torch.vander(X_ols_full.flatten(), N=degree+1, increasing=True)
X_design_test = torch.vander(X_ols_test.flatten(), N=degree+1, increasing=True)

beta_ols = torch.linalg.pinv(X_design_full) @ y_full.double()

ols_train_pred = X_design_full @ beta_ols
ols_test_pred = X_design_test @ beta_ols

ols_train_mse = MSE(y_full.double(), ols_train_pred).item()
ols_test_mse = MSE(y_test.double(), ols_test_pred).item()

print("Final OLS Train MSE:", ols_train_mse)
print("Final OLS Test MSE:", ols_test_mse)

# Final Prediction Comparison

x_plot = torch.linspace(-1, 1, 500).reshape(-1, 1)
x_plot_scaled = (x_plot - mu) / std

nn_prediction = final_model.predict(x_plot_scaled)

X_plot_ols = torch.vander(x_plot.flatten().double(), N=degree+1, increasing=True)
ols_prediction = X_plot_ols @ beta_ols

plt.figure(figsize=(9,5))
plt.plot(x_plot, Rungefunction(x_plot), label="True Runge Function")
plt.plot(x_plot, nn_prediction, label="Neural Network RMSprop")
plt.plot(x_plot, ols_prediction, label="OLS Degree 11")
plt.scatter(X_test * std + mu, y_test, label="Test Data", s=15)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Neural Network vs OLS Regression")
plt.legend()
plt.grid(True)
plt.show()


"""
Epoch 100, MSE: 0.088513
Epoch 200, MSE: 0.088330
Train MSE: 0.08829797804355621
Test MSE: 0.11863492429256439
Epoch 0, MSE: 0.097838
Epoch 100, MSE: 0.090739
Epoch 200, MSE: 0.088228
Architecture: [1, 50, 1]
Train MSE: 0.088251
Validation MSE: 0.101186
Epoch 0, MSE: 0.091494
Epoch 100, MSE: 0.090483
Epoch 200, MSE: 0.089810
Architecture: [1, 100, 100, 1]
Train MSE: 0.091099
Validation MSE: 0.092334
Epoch 0, MSE: 0.164151
Epoch 100, MSE: 0.100176
Epoch 200, MSE: 0.097176
Learning rate  0.0001
Train MSE 0.09662284702062607
Val MSE 0.1187673807144165
Epoch 0, MSE: 0.146152
Epoch 100, MSE: 0.094075
Epoch 200, MSE: 0.091715
Learning rate  0.001
Train MSE 0.09033305197954178
Val MSE 0.10786399990320206
Epoch 0, MSE: 0.097838
Epoch 100, MSE: 0.090739
Epoch 200, MSE: 0.088228
Learning rate  0.01
Train MSE 0.08825132995843887
Val MSE 0.10118554532527924
Epoch 0, MSE: 3.017045
Epoch 100, MSE: 0.116916
Epoch 200, MSE: 0.080073
Learning rate  0.1
Train MSE 0.05202668160200119
Val MSE 0.053235165774822235
Adam Learning Rate: 0.0001
Train MSE: 0.08819089829921722
Val MSE: 0.10063659399747849
Adam Learning Rate: 0.001
Train MSE: 0.08736450970172882
Val MSE: 0.09904645383358002
Adam Learning Rate: 0.01
Train MSE: 0.013812021352350712
Val MSE: 0.014152584597468376
RMS Learning Rate: 0.0001
Train MSE: 0.08818434178829193
Val MSE: 0.10065648704767227
RMS Learning Rate: 0.001
Train MSE: 0.0874515250325203
Val MSE: 0.10047679394483566
RMS Learning Rate: 0.01
Train MSE: 0.01157435029745102
Val MSE: 0.010490930639207363
Epoch 0, MSE: 0.134155
Epoch 100, MSE: 0.092364
Epoch 200, MSE: 0.090016
GD Train MSE: 0.08901924639940262
GD Val MSE: 0.10382972657680511


OLS Degree: 11
OLS Train MSE: 0.0078068542696133085
OLS Validation MSE: 0.015104051476843417


Maximum Gradient Difference: 3.725290298461914e-09
"""