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


    #Different learning rates

    Learning_Rata_E = [0.001, 0.01, 0.1, 0.5, 0.9]
    learning_accuracy = []

    for lr in Learning_Rata_E:
        torch.manual_seed(2026)

        #Use Neurarnetwork 2, since there has the best result.
        model = ClassificationNetwork([784,64,64,10])
        model.set_activation("relu")

        history = train_classification(model, X_train, y_train_onehot, X_val, y_val, epochs=5, learning_rate=lr, batch_size=128)

        val_pred = model.predict(X_val).argmax(dim=1)
        val_accuracy = (val_pred == y_val).float().mean().item()

        learning_accuracy.append(val_accuracy)

        print("Learning Rate:", lr)
        print("Validation Accuracy:", val_accuracy)

        plt.plot(history, label=f"lr={lr}")

    plt.xlabel("Epoch")
    plt.ylabel("Training Cross Entropy")
    plt.title("Different Learning Rates")
    plt.legend()
    plt.grid(True)
    plt.show()


    #Different gamma

    Gamma_E = Learning_Rata_E

    plt.plot(Gamma_E, learning_accuracy, marker="o")
    plt.xlabel("Gamma")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs Gamma")
    plt.grid(True)
    plt.show()


    #Hidden layer

    Hidden_Layer_E = [1,2,3]
    hidden_accuracy = []

    for n in Hidden_Layer_E:
        torch.manual_seed(2026)

        layers = [784] + [64]*n + [10]
        model = ClassificationNetwork(layers)
        model.set_activation("relu")

        history = train_classification(model, X_train, y_train_onehot, X_val, y_val, epochs=5, learning_rate=0.1, batch_size=128)

        val_pred = model.predict(X_val).argmax(dim=1)
        val_accuracy = (val_pred == y_val).float().mean().item()

        hidden_accuracy.append(val_accuracy)

        print("Hidden Layers:", n)
        print("Validation Accuracy:", val_accuracy)

        plt.plot(history, label=f"{n} Hidden Layers")

    plt.xlabel("Epoch")
    plt.ylabel("Training Cross Entropy")
    plt.title("Different Hidden Layers")
    plt.legend()
    plt.grid(True)
    plt.show()


    #Validation Accuracy

    plt.plot(Hidden_Layer_E, hidden_accuracy, marker="o")
    plt.xlabel("Number of Hidden Layers")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs Hidden Layers")
    plt.xticks(Hidden_Layer_E)
    plt.grid(True)
    plt.show()

#Sigmoid

    torch.manual_seed(2026)

    model = ClassificationNetwork([784,64,64,10])
    model.set_activation("sigmoid")

    history = train_classification(model, X_train, y_train_onehot, X_val, y_val, epochs=5, learning_rate=0.1, batch_size=128)

    val_pred = model.predict(X_val).argmax(dim=1)
    val_accuracy = (val_pred == y_val).float().mean().item()

    print("Sigmoid Validation Accuracy:", val_accuracy)
    print("ReLU Validation Accuracy:", validation_results["Neural Network 2 Layers"])

    plt.plot(history, label="Sigmoid")
    plt.xlabel("Epoch")
    plt.ylabel("Training Cross Entropy")
    plt.title("Sigmoid Training Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


    #Different Lambda

    Lambda_E = [0.000001, 0.00001, 0.0001, 0.001]
    lambda_results = {"L1": [], "L2": []}

    for reg_type in ["L1", "L2"]:
        for lam in Lambda_E:
            torch.manual_seed(2026)

            model = ClassificationNetwork([784,64,64,10], reg_type, lam)
            model.set_activation("relu")

            history = train_classification(model, X_train, y_train_onehot, X_val, y_val, epochs=5, learning_rate=0.5, batch_size=128)

            val_pred = model.predict(X_val).argmax(dim=1)
            val_accuracy = (val_pred == y_val).float().mean().item()

            lambda_results[reg_type].append(val_accuracy)

            print("Regularization:", reg_type)
            print("Lambda:", lam)
            print("Validation Accuracy:", val_accuracy)


    #Lambda Comparison

    for reg_type in ["L1", "L2"]:
        plt.plot(Lambda_E, lambda_results[reg_type], marker="o", label=reg_type)

    plt.xscale("log")
    plt.xlabel("Lambda")
    plt.ylabel("Validation Accuracy")
    plt.title("Different Lambda")
    plt.legend()
    plt.grid(True)
    plt.show()



    #Different Hidden Nodes

    Hidden_Nodes_E = [32,64,128]
    nodes_accuracy = []

    for nodes in Hidden_Nodes_E:
        torch.manual_seed(2026)

        model = ClassificationNetwork([784,nodes,nodes,10])
        model.set_activation("relu")

        history = train_classification(model, X_train, y_train_onehot, X_val, y_val, epochs=5, learning_rate=0.5, batch_size=128)

        val_pred = model.predict(X_val).argmax(dim=1)
        val_accuracy = (val_pred == y_val).float().mean().item()

        nodes_accuracy.append(val_accuracy)

        print("Hidden Nodes:", nodes)
        print("Validation Accuracy:", val_accuracy)

        plt.plot(history, label=f"{nodes} Nodes")

    plt.xlabel("Epoch")
    plt.ylabel("Training Cross Entropy")
    plt.title("Different Hidden Nodes")
    plt.legend()
    plt.grid(True)
    plt.show()


    #Validation Accuracy

    plt.plot(Hidden_Nodes_E, nodes_accuracy, marker="o")
    plt.xlabel("Number of Hidden Nodes")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs Hidden Nodes")
    plt.xticks(Hidden_Nodes_E)
    plt.grid(True)
    plt.show()
#Scikit vs PyTorch

    from sklearn.neural_network import MLPClassifier
    import torch.nn as nn

    #Scikit-Learn

    sklearn_model = MLPClassifier(hidden_layer_sizes=(64,64), activation="relu", solver="sgd", learning_rate_init=0.1, batch_size=128, max_iter=5, momentum=0.0, alpha=0.0, random_state=2026)

    sklearn_model.fit(X_train.numpy(), y_train.numpy())

    sklearn_pred = sklearn_model.predict(X_val.numpy())
    sklearn_accuracy = np.mean(sklearn_pred == y_val.numpy())

    print("Scikit-Learn Validation Accuracy:", sklearn_accuracy)


    #PyTorch

    torch.manual_seed(2026)

    pytorch_model = nn.Sequential(nn.Linear(784,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU(), nn.Linear(64,10))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(pytorch_model.parameters(), lr=0.1)

    pytorch_history = []

    for epoch in range(5):
        indices = torch.randperm(X_train.shape[0])

        for batch in indices.split(128):
            optimizer.zero_grad()
            output = pytorch_model(X_train[batch])
            loss = criterion(output, y_train[batch])
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            loss = criterion(pytorch_model(X_train), y_train).item()
            pytorch_history.append(loss)

        print("Epoch:", epoch+1, "Cross Entropy:", loss)

    with torch.no_grad():
        pytorch_pred = pytorch_model(X_val).argmax(dim=1)
        pytorch_accuracy = (pytorch_pred == y_val).float().mean().item()

    print("PyTorch Validation Accuracy:", pytorch_accuracy)


    #Model Comparison

    print("Own Neural Network:", validation_results["Neural Network 2 Layers"])
    print("Scikit-Learn:", sklearn_accuracy)
    print("PyTorch:", pytorch_accuracy)

    plt.plot(sklearn_model.loss_curve_, label="Scikit-Learn")
    plt.plot(pytorch_history, label="PyTorch")
    plt.xlabel("Epoch")
    plt.ylabel("Training Cross Entropy")
    plt.title("Scikit-Learn vs PyTorch")
    plt.legend()
    plt.grid(True)
    plt.show()






