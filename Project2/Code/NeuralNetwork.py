
import torch
from functions import sigmoid, sigmoidDerivative, linear, linearDerivative, MSE, MSEDerivative


class NeuralNetwork:

    def __init__(self, layer_sizes):

        # Initial weights and biases
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []



        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):

            std = (2.0 / (n_in + n_out)) ** 0.5

            w = torch.randn(n_in, n_out) * std

            b = torch.zeros(1, n_out)

            self.weights.append(w)

            self.biases.append(b)

    # Feed forward

    def feedforward(self, X):

        a = X

        self.activations = [X]

        self.z_values = []

        for i in range(len(self.weights)):

            z = a @ self.weights[i] + self.biases[i]

            self.z_values.append(z)

            if i == len(self.weights) - 1:

                a = linear(z)

            else:

                a = sigmoid(z)

            self.activations.append(a)

        return a

    
    # Backpropagation


    def backpropagation(self, X, y):

        y_pred = self.feedforward(X)

        # Initialize gradients

        self.grad_weights = [torch.zeros_like(w) for w in self.weights]

        self.grad_biases = [torch.zeros_like(b) for b in self.biases]

        # Output layer


        delta = MSEDerivative(y, y_pred) * linearDerivative(self.z_values[-1])

        # Hidden layers

        for i in reversed(range(len(self.weights))):

            self.grad_weights[i] = self.activations[i].T @ delta

            self.grad_biases[i] = torch.sum(delta, dim=0, keepdim=True)

            if i > 0:

                delta = (delta @ self.weights[i].T) * sigmoidDerivative(self.z_values[i - 1])

        return MSE(y, y_pred).item()
    
    # Gradient

    def gradient(self, learning_rate):

        for i in range(len(self.weights)):

            self.weights[i] -= learning_rate * self.grad_weights[i]

            self.biases[i] -= learning_rate * self.grad_biases[i]

    # Training

    def train(self, X, y, epochs=1000, learning_rate=0.01, batch_size=None):

        n = X.shape[0]

        if batch_size is None:

            batch_size = n

        if batch_size <= 0:

            raise ValueError("batch_size must be positive")
        
        history = []

        for epoch in range(epochs):

            # Shuffle data

            indices = torch.randperm(n)

            X_shuffled = X[indices]

            y_shuffled = y[indices]

            for start in range(0, n, batch_size):

                X_batch = X_shuffled[start:start + batch_size]

                y_batch = y_shuffled[start:start + batch_size]

                # Backpropagation

                self.backpropagation(X_batch, y_batch)

                # Update weights and biases

                self.gradient(learning_rate)

            # Calculate training loss

            y_pred = self.feedforward(X)

            loss = MSE(y, y_pred).item()

            history.append(loss)

            if epoch % 100 == 0:

                print(f"Epoch {epoch}, MSE: {loss:.6f}")

        return history
    
    # Prediction

    def predict(self, X):

        
        return self.feedforward(X)
