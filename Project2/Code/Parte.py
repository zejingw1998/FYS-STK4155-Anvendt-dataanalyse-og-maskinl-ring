import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from NeuralNetwork import NeuralNetwork


#Softmax

def softmax(z):
    z = z - torch.max(z, dim=1, keepdim=True).values
    exp_z = torch.exp(z)
    return exp_z / torch.sum(exp_z, dim=1, keepdim=True)


#Cross Entropy

def cross_entropy(y_true, y_pred):
    return -torch.mean(torch.sum(y_true * torch.log(y_pred.clamp_min(1e-12)), dim=1))


#Classification Neural Network

class ClassificationNetwork(NeuralNetwork):
    def __init__(self, layer_sizes, reg_type=None, lam=0.0):
        super().__init__(layer_sizes)
        self.reg_type = reg_type
        self.lam = lam


 #Feed forward
    def feedforward(self, X):
        a = X
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.weights)):
            z = a @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            a = softmax(z) if i == len(self.weights)-1 else self.activation(z)
            self.activations.append(a)

        return a

    #Backpropagation
    def backpropagation(self, X, y):
        y_pred = self.feedforward(X)
        delta = (y_pred - y) / X.shape[0]
        self.grad_weights = [torch.zeros_like(w) for w in self.weights]
        self.grad_biases = [torch.zeros_like(b) for b in self.biases]

        for i in reversed(range(len(self.weights))):
            self.grad_weights[i] = self.activations[i].T @ delta
            self.grad_biases[i] = torch.sum(delta, dim=0, keepdim=True)

            if self.reg_type == "L1":
                self.grad_weights[i] += self.lam * torch.sign(self.weights[i])
                self.grad_biases[i] += self.lam * torch.sign(self.biases[i])
            elif self.reg_type == "L2":
                self.grad_weights[i] += 2 * self.lam * self.weights[i]
                self.grad_biases[i] += 2 * self.lam * self.biases[i]

            if i > 0:
                delta = (delta @ self.weights[i].T) * self.activationDerivative(self.z_values[i-1])

        return cross_entropy(y, y_pred).item()


#Generate Data

def load_data():
    path = Path("mnist.npz")
    if not path.exists():
        urlretrieve("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", path)

    with np.load(path) as data:
        X_train = data["x_train"].reshape(-1, 784).astype(np.float32) / 255.0
        y_train = data["y_train"].astype(np.int64)
        X_test = data["x_test"].reshape(-1, 784).astype(np.float32) / 255.0
        y_test = data["y_test"].astype(np.int64)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=2026, stratify=y_train)
    X_train = torch.from_numpy(X_train)
    X_val = torch.from_numpy(X_val)
    X_test = torch.from_numpy(X_test)
    y_train = torch.from_numpy(y_train)
    y_val = torch.from_numpy(y_val)
    y_test = torch.from_numpy(y_test)
    y_train_onehot = torch.eye(10)[y_train]
    return X_train, X_val, X_test, y_train, y_val, y_test, y_train_onehot


#Training

def train_classification(model, X_train, y_train_onehot, X_val, y_val, epochs=5, learning_rate=0.1, batch_size=128):
    history = []

    for epoch in range(epochs):
        indices = torch.randperm(X_train.shape[0])
        total_loss = 0.0

        for batch in indices.split(batch_size):
            loss = model.backpropagation(X_train[batch], y_train_onehot[batch])
            model.gradient(learning_rate)
            total_loss += loss * len(batch)

        history.append(total_loss / len(X_train))
        val_pred = model.predict(X_val).argmax(dim=1)
        val_accuracy = (val_pred == y_val).float().mean().item()
        print(f"Epoch {epoch+1}: Cross Entropy = {history[-1]:.4f}, Validation Accuracy = {val_accuracy:.4f}")

    return history


#Model Comparison

if __name__ == "__main__":
    torch.manual_seed(2026)
    X_train, X_val, X_test, y_train, y_val, y_test, y_train_onehot = load_data()
    print("Train:", X_train.shape, "Validation:", X_val.shape, "Test:", X_test.shape)

    experiments = [
        ("Logistic Regression", [784,10], "sigmoid", 0.1, None, 0.0),
        ("Neural Network ReLU", [784,64,10], "relu", 0.1, None, 0.0),
        ("Neural Network Leaky ReLU", [784,64,10], "leaky_relu", 0.1, None, 0.0),
        ("Neural Network 2 Layers", [784,64,64,10], "relu", 0.1, None, 0.0),
        ("Neural Network L2", [784,64,10], "relu", 0.1, "L2", 0.00001)
    ]
    trained_models = {}
    validation_results = {}

    for name, layers, activation, lr, reg_type, lam in experiments:
        print("\n", name)
        torch.manual_seed(2026)
        model = ClassificationNetwork(layers, reg_type, lam)
        model.set_activation(activation)
        history = train_classification(model, X_train, y_train_onehot, X_val, y_val, epochs=5, learning_rate=lr, batch_size=128)
        val_accuracy = (model.predict(X_val).argmax(dim=1) == y_val).float().mean().item()
        trained_models[name] = model
        validation_results[name] = val_accuracy
        print("Validation Accuracy:", val_accuracy)
        plt.plot(history, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Training Cross Entropy")
    plt.title("Classification Training Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    #Final Test Comparison
    best_nn_name = max((name for name in validation_results if name != "Logistic Regression"), key=lambda name: validation_results[name])
    print("Best Neural Network:", best_nn_name)

    for name in ["Logistic Regression", best_nn_name]:
        model = trained_models[name]
        predictions = model.predict(X_test).argmax(dim=1)
        accuracy = (predictions == y_test).float().mean().item()
        print(name, "Test Accuracy:", accuracy)

    #Confusion Matrix
    best_model = trained_models[best_nn_name]
    y_pred = best_model.predict(X_test).argmax(dim=1).numpy()
    matrix = confusion_matrix(y_test.numpy(), y_pred, labels=list(range(10)))
    ConfusionMatrixDisplay(matrix, display_labels=list(range(10))).plot(cmap="Blues")
    plt.title("MNIST Confusion Matrix")
    plt.show()


"""
Train: torch.Size([54000, 784]) Validation: torch.Size([6000, 784]) Test: torch.Size([10000, 784])

 Logistic Regression
Epoch 1: Cross Entropy = 0.6013, Validation Accuracy = 0.8988
Epoch 2: Cross Entropy = 0.3850, Validation Accuracy = 0.9095
Epoch 3: Cross Entropy = 0.3509, Validation Accuracy = 0.9113
Epoch 4: Cross Entropy = 0.3331, Validation Accuracy = 0.9163
Epoch 5: Cross Entropy = 0.3217, Validation Accuracy = 0.9178
Validation Accuracy: 0.9178333282470703

 Neural Network ReLU
Epoch 1: Cross Entropy = 0.4979, Validation Accuracy = 0.9167
Epoch 2: Cross Entropy = 0.2722, Validation Accuracy = 0.9347
Epoch 3: Cross Entropy = 0.2282, Validation Accuracy = 0.9430
Epoch 4: Cross Entropy = 0.1988, Validation Accuracy = 0.9487
Epoch 5: Cross Entropy = 0.1777, Validation Accuracy = 0.9530
Validation Accuracy: 0.953000009059906

 Neural Network Leaky ReLU
Epoch 1: Cross Entropy = 0.4974, Validation Accuracy = 0.9165
Epoch 2: Cross Entropy = 0.2729, Validation Accuracy = 0.9343
Epoch 3: Cross Entropy = 0.2289, Validation Accuracy = 0.9437
Epoch 4: Cross Entropy = 0.1995, Validation Accuracy = 0.9490
Epoch 5: Cross Entropy = 0.1783, Validation Accuracy = 0.9532
Validation Accuracy: 0.953166663646698

 Neural Network 2 Layers
Epoch 1: Cross Entropy = 0.5039, Validation Accuracy = 0.9083
Epoch 2: Cross Entropy = 0.2403, Validation Accuracy = 0.9443
Epoch 3: Cross Entropy = 0.1874, Validation Accuracy = 0.9538
Epoch 4: Cross Entropy = 0.1540, Validation Accuracy = 0.9583
Epoch 5: Cross Entropy = 0.1310, Validation Accuracy = 0.9633
Validation Accuracy: 0.9633333086967468

 Neural Network L2
Epoch 1: Cross Entropy = 0.4979, Validation Accuracy = 0.9170
Epoch 2: Cross Entropy = 0.2723, Validation Accuracy = 0.9343
Epoch 3: Cross Entropy = 0.2282, Validation Accuracy = 0.9428
Epoch 4: Cross Entropy = 0.1987, Validation Accuracy = 0.9485
Epoch 5: Cross Entropy = 0.1775, Validation Accuracy = 0.9527
Validation Accuracy: 0.9526666402816772
Best Neural Network: Neural Network 2 Layers
Logistic Regression Test Accuracy: 0.9156000018119812
Neural Network 2 Layers Test Accuracy: 0.9620000123977661
"""