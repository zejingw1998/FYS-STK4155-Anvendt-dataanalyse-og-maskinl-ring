import torch
from functions import Rungefunction,sigmoid, sigmoidDerivative, linear, linearDerivative, MSE, MSEDerivative,leakyRelu,leakyReluDerivative,relu,reluDerivative
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline



#L1 and L2 Regularization

class RegularizedNeuralNetwork(NeuralNetwork):
    def __init__(self, layer_sizes, reg_type, lam):
        super().__init__(layer_sizes)
        self.reg_type = reg_type
        self.lam = lam

    def backpropagation(self, X, y):
        loss = super().backpropagation(X, y)
        penalty = 0.0

        for i, W in enumerate(self.weights):
            b = self.biases[i]

            if self.reg_type == "L1":
                self.grad_weights[i] += self.lam * torch.sign(W)
                self.grad_biases[i] += self.lam * torch.sign(b)
                penalty += self.lam * (torch.norm(W,p=1) + torch.norm(b,p=1)).item()

            elif self.reg_type == "L2":
                self.grad_weights[i] += 2 * self.lam * W
                self.grad_biases[i] += 2 * self.lam * b
                penalty += self.lam * (torch.norm(W,p=2)**2 + torch.norm(b,p=2)**2).item()

        return loss + penalty




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




#Hyperparameter Search

best_models = {}
best_results = {}

for reg_type in ["L1", "L2"]:
    best_val = float("inf")

    for layers in [[1, 20, 1], [1, 50, 1]]:
        for lr in [0.001, 0.01]:
            for lam in [0.00001, 0.0001, 0.001]:

                torch.manual_seed(2026)
                model = RegularizedNeuralNetwork(layers, reg_type, lam)
                history = model.train(X_train, y_train, epochs=300, learning_rate=lr, batch_size=10)

                val_mse = MSE(y_val, model.predict(X_val)).item()

                print(reg_type, "Layers:", layers, "LR:", lr, "Lambda:", lam, "Validation MSE:", val_mse)

                if val_mse < best_val:
                    best_val = val_mse
                    best_models[reg_type] = model
                    best_results[reg_type] = (layers, lr, lam, val_mse)


#Best Results

for reg_type in ["L1", "L2"]:
    layers, lr, lam, val_mse = best_results[reg_type]
    

    print(reg_type, "Best Layers:", layers)
    print("Learning Rate:", lr)
    print("Lambda:", lam)
    print("Validation MSE:", val_mse)
    print("----------")



#L1 and L2 Comparison

x_plot = torch.linspace(-1, 1, 200).reshape(-1, 1)
x_scaled = (x_plot - mu) / std

plt.figure(figsize=(10, 6))
plt.plot(x_plot, Rungefunction(x_plot), label="True Runge Function")

for reg_type in ["L1", "L2"]:
    y_pred = best_models[reg_type].predict(x_scaled)
    plt.plot(x_plot, y_pred.detach(), label=reg_type + " Neural Network")

plt.scatter(X, y, alpha=0.3, label="Data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()



#L1/L2 weights and bias

for reg_type in ["L1", "L2"]:
    model = best_models[reg_type]

#Weights
    weights = torch.cat([w.flatten() for w in model.weights])

#Bias
    biases = torch.cat([b.flatten() for b in model.biases])


#L1 norm 

    L1_NORM = torch.norm(weights, p=1) + torch.norm(biases, p=1)

#L2 norm

    L2_NORM = torch.norm(weights,p=2)**2+torch.norm(biases,p= 2)**2

    print("Regularization", reg_type)
    print("L1 Norm:", L1_NORM.item())
    print("L2 Norm Squared:", L2_NORM.item())
    print("--------------")
#RelU


#ReLU

for reg_type in ["L1", "L2"]:
    layers, lr, lam, val_mse = best_results[reg_type]

    torch.manual_seed(2026)
    model = RegularizedNeuralNetwork(layers, reg_type, lam)
    model.set_activation("relu")

    #Training
    history = model.train(X_train, y_train, epochs=300, learning_rate=lr, batch_size=10)

    #Validation MSE
    relu_val_mse = MSE(y_val, model.predict(X_val)).item()
    
    #Comparison
    print("Regularization:", reg_type)
    print("Sigmoid Validation MSE:", val_mse)
    print("ReLU Validation MSE:", relu_val_mse)



#Different layers

Different_layers = [[1,50,1], [1,50,50,1], [1,50,50,50,1]]
layer_results = {"L1": [], "L2": []}
for reg_type in ["L1", "L2"]:
    _, lr, lam, _ = best_results[reg_type]

    for layers in Different_layers:
        torch.manual_seed(2026)
        model = RegularizedNeuralNetwork(layers, reg_type, lam)
        model.set_activation("relu")


        #Training
        history = model.train(X_train, y_train, epochs=300, learning_rate=lr, batch_size=10)

        #MSE
        train_mse = MSE(y_train, model.predict(X_train)).item()
        val_mse = MSE(y_val, model.predict(X_val)).item()


        #Results
        print("Regularization:", reg_type)
        print("Architecture:", layers)
        print("Train MSE:", train_mse)
        print("Validation MSE:", val_mse)
        layer_results[reg_type].append(val_mse)

#Plot Different Layers

for reg_type in ["L1", "L2"]:
    plt.plot([len(layers)-2 for layers in Different_layers], layer_results[reg_type], marker="o", label=reg_type)

plt.xlabel("Number of Hidden Layers")
plt.ylabel("Validation MSE")
plt.title("L1 and L2 Regularization with Different Hidden Layers")
plt.xticks([1,2,3])
plt.legend()
plt.grid(True)
plt.show()


#Different learning rate

Different_Learning_Rate = [0.0001,0.001,0.01,0.1,0.8]
learning_rate_results = {"L1": [], "L2": []}


for reg_type in ["L1", "L2"]:
    layers, _, lam, _ = best_results[reg_type]

    for lr in Different_Learning_Rate:
        torch.manual_seed(2026)
        model = RegularizedNeuralNetwork(layers, reg_type, lam)
        model.set_activation("relu")

        #Training
        history = model.train(X_train, y_train, epochs=300, learning_rate=lr, batch_size=10)

        #MSE
        train_mse = MSE(y_train, model.predict(X_train)).item()
        val_mse = MSE(y_val, model.predict(X_val)).item()

        #Results
        print("Regularization:", reg_type)
        print("Learning Rate:", lr)
        print("Train MSE:", train_mse)
        print("Validation MSE:", val_mse)

        learning_rate_results[reg_type].append(val_mse)


#Plot Different Learning Rates

for reg_type in ["L1", "L2"]:
    plt.plot(Different_Learning_Rate, learning_rate_results[reg_type], marker="o", label=reg_type)

plt.xscale("log")
plt.xlabel("Learning Rate")
plt.ylabel("Validation MSE")
plt.title("L1 and L2 Regularization with Different Learning Rates")
plt.legend()
plt.grid(True)
plt.show()





#Heatmap

Lambda_values = [0.00001,0.0001,0.001]

heatmap_results = {"L1": torch.full((len(Different_Learning_Rate),len(Lambda_values)),float("nan")), "L2": torch.full((len(Different_Learning_Rate),len(Lambda_values)),float("nan"))}
heatmap_best_models = {}
heatmap_best_results = {}

for reg_type in ["L1", "L2"]:
    layers, _, _, _ = best_results[reg_type]
    best_val = float("inf")

    for i, lr in enumerate(Different_Learning_Rate):
        for j, lam in enumerate(Lambda_values):

            torch.manual_seed(2026)
            model = RegularizedNeuralNetwork(layers, reg_type, lam)
            model.set_activation("relu")

            #Training
            history = model.train(X_train, y_train, epochs=300, learning_rate=lr, batch_size=10)

            #Validation MSE
            val_mse = MSE(y_val, model.predict(X_val)).item()

            #Results
            print("Regularization:", reg_type)
            print("Learning Rate:", lr)
            print("Lambda:", lam)
            print("Validation MSE:", val_mse)

            if torch.isfinite(torch.tensor(val_mse)).item():
                heatmap_results[reg_type][i,j] = val_mse

                if val_mse < best_val:
                    best_val = val_mse
                    heatmap_best_models[reg_type] = model
                    heatmap_best_results[reg_type] = (layers, lr, lam, val_mse)


#Plot Heatmap

for reg_type in ["L1", "L2"]:
    plt.figure(figsize=(8,6))
    plt.imshow(heatmap_results[reg_type].numpy(), aspect="auto", origin="lower")
    plt.colorbar(label="Validation MSE")
    plt.xticks(range(len(Lambda_values)), Lambda_values)
    plt.yticks(range(len(Different_Learning_Rate)), Different_Learning_Rate)
    plt.xlabel("Lambda")
    plt.ylabel("Learning Rate")
    plt.title(reg_type + " Regularization Heatmap")
    plt.show()


#Best Hyperparameters

for reg_type in ["L1", "L2"]:
    layers, lr, lam, val_mse = heatmap_best_results[reg_type]
    print("Regularization:", reg_type)
    print("Best Architecture:", layers)
    print("Best Learning Rate:", lr)
    print("Best Lambda:", lam)
    print("Best Validation MSE:", val_mse)



#Ridge comparison

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

#Polynomial Features

degree = 11
poly = PolynomialFeatures(degree=degree, include_bias=False)

X_train_original = (X_train * std + mu).numpy()
X_val_original = (X_val * std + mu).numpy()

X_poly_train = poly.fit_transform(X_train_original)
X_poly_val = poly.transform(X_val_original)

#Ridge Hyperparameter Search

Ridge_Lambda = [0.00000001,0.000001,0.00001,0.0001,0.001,0.01]

best_ridge_val = float("inf")
best_ridge_model = None

for lam in Ridge_Lambda:
    ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=lam, solver="svd"))
    ridge_model.fit(X_poly_train, y_train.numpy().flatten())

    #Validation MSE
    ridge_prediction = torch.tensor(ridge_model.predict(X_poly_val), dtype=y_val.dtype).reshape(-1,1)
    val_mse = MSE(y_val, ridge_prediction).item()

    print("Ridge Lambda:", lam)
    print("Validation MSE:", val_mse)

    if val_mse < best_ridge_val:
        best_ridge_val = val_mse
        best_ridge_model = ridge_model
        best_ridge_lambda = lam


#Ridge and L2 Comparison

l2_val_mse = MSE(y_val, heatmap_best_models["L2"].predict(X_val)).item()

print("Best Ridge Lambda:", best_ridge_lambda)
print("Ridge Validation MSE:", best_ridge_val)
print("L2 Neural Network Validation MSE:", l2_val_mse)

#Plot Ridge and L2

x_compare = torch.linspace(-1,1,200).reshape(-1,1)
x_compare_scaled = (x_compare - mu) / std
X_poly_plot = poly.transform(x_compare.numpy())

l2_prediction = heatmap_best_models["L2"].predict(x_compare_scaled).detach()
ridge_prediction = best_ridge_model.predict(X_poly_plot)

plt.figure(figsize=(10,6))
plt.plot(x_compare, Rungefunction(x_compare), label="True Runge Function")
plt.plot(x_compare, l2_prediction, label="L2 Neural Network")
plt.plot(x_compare, ridge_prediction, label="Ridge Regression")
plt.scatter(X_train_original, y_train.numpy(), alpha=0.3, label="Training Data")
plt.xlabel("x")
plt.ylabel("y")
plt.title("L2 Neural Network vs Ridge Regression")
plt.legend()
plt.grid(True)
plt.show()



#Lasson Comparison

from sklearn.linear_model import Lasso

#Lasso Hyperparameter Search

Lasso_Lambda = [0.0001,0.0003,0.001,0.01]

best_lasso_val = float("inf")
best_lasso_model = None

for lam in Lasso_Lambda:
    lasso_model = make_pipeline(StandardScaler(), Lasso(alpha=lam, max_iter=100000, tol=1e-4))
    lasso_model.fit(X_poly_train, y_train.numpy().flatten())

    #Validation MSE
    lasso_prediction = torch.tensor(lasso_model.predict(X_poly_val), dtype=y_val.dtype).reshape(-1,1)
    val_mse = MSE(y_val, lasso_prediction).item()

    print("Lasso Lambda:", lam)
    print("Validation MSE:", val_mse)

    if val_mse < best_lasso_val:
        best_lasso_val = val_mse
        best_lasso_model = lasso_model
        best_lasso_lambda = lam


#Lasso and L1 Comparison

l1_val_mse = MSE(y_val, heatmap_best_models["L1"].predict(X_val)).item()

print("Best Lasso Lambda:", best_lasso_lambda)
print("Lasso Validation MSE:", best_lasso_val)
print("L1 Neural Network Validation MSE:", l1_val_mse)

#Plot Lasso and L1

l1_prediction = heatmap_best_models["L1"].predict(x_compare_scaled).detach()
lasso_prediction = best_lasso_model.predict(X_poly_plot)

plt.figure(figsize=(10,6))
plt.plot(x_compare, Rungefunction(x_compare), label="True Runge Function")
plt.plot(x_compare, l1_prediction, label="L1 Neural Network")
plt.plot(x_compare, lasso_prediction, label="Lasso Regression")
plt.scatter(X_train_original, y_train.numpy(), alpha=0.3, label="Training Data")
plt.xlabel("x")
plt.ylabel("y")
plt.title("L1 Neural Network vs Lasso Regression")
plt.legend()
plt.grid(True)
plt.show()



#Final Test Comparison

X_test_original = (X_test * std + mu).numpy()
X_poly_test = poly.transform(X_test_original)

l1_test_mse = MSE(y_test, heatmap_best_models["L1"].predict(X_test)).item()
l2_test_mse = MSE(y_test, heatmap_best_models["L2"].predict(X_test)).item()

ridge_test_prediction = torch.tensor(best_ridge_model.predict(X_poly_test), dtype=y_test.dtype).reshape(-1,1)
lasso_test_prediction = torch.tensor(best_lasso_model.predict(X_poly_test), dtype=y_test.dtype).reshape(-1,1)

ridge_test_mse = MSE(y_test, ridge_test_prediction).item()
lasso_test_mse = MSE(y_test, lasso_test_prediction).item()

print("Final L1 Neural Network Test MSE:", l1_test_mse)
print("Final L2 Neural Network Test MSE:", l2_test_mse)
print("Final Ridge Test MSE:", ridge_test_mse)
print("Final Lasso Test MSE:", lasso_test_mse)





"""
Epoch 0, MSE: 0.325949
Epoch 100, MSE: 0.083751
Epoch 200, MSE: 0.082864
L1 Layers: [1, 20, 1] LR: 0.001 Lambda: 1e-05 Validation MSE: 0.06320439279079437
Epoch 0, MSE: 0.325950
Epoch 100, MSE: 0.083749
Epoch 200, MSE: 0.082862
L1 Layers: [1, 20, 1] LR: 0.001 Lambda: 0.0001 Validation MSE: 0.06319951266050339
Epoch 0, MSE: 0.325960
Epoch 100, MSE: 0.083729
Epoch 200, MSE: 0.082842
L1 Layers: [1, 20, 1] LR: 0.001 Lambda: 0.001 Validation MSE: 0.06315117329359055
Epoch 0, MSE: 0.141648
Epoch 100, MSE: 0.081437
Epoch 200, MSE: 0.081317
L1 Layers: [1, 20, 1] LR: 0.01 Lambda: 1e-05 Validation MSE: 0.061323102563619614
Epoch 0, MSE: 0.141651
Epoch 100, MSE: 0.081443
Epoch 200, MSE: 0.081333
L1 Layers: [1, 20, 1] LR: 0.01 Lambda: 0.0001 Validation MSE: 0.06133570522069931
Epoch 0, MSE: 0.141688
Epoch 100, MSE: 0.081502
Epoch 200, MSE: 0.081472
L1 Layers: [1, 20, 1] LR: 0.01 Lambda: 0.001 Validation MSE: 0.06143587455153465
Epoch 0, MSE: 0.132834
Epoch 100, MSE: 0.090186
Epoch 200, MSE: 0.086730
L1 Layers: [1, 50, 1] LR: 0.001 Lambda: 1e-05 Validation MSE: 0.06658447533845901
Epoch 0, MSE: 0.132835
Epoch 100, MSE: 0.090180
Epoch 200, MSE: 0.086724
L1 Layers: [1, 50, 1] LR: 0.001 Lambda: 0.0001 Validation MSE: 0.06657636910676956
Epoch 0, MSE: 0.132844
Epoch 100, MSE: 0.090112
Epoch 200, MSE: 0.086660
L1 Layers: [1, 50, 1] LR: 0.001 Lambda: 0.001 Validation MSE: 0.06649743765592575
Epoch 0, MSE: 0.103366
Epoch 100, MSE: 0.082741
Epoch 200, MSE: 0.081692
L1 Layers: [1, 50, 1] LR: 0.01 Lambda: 1e-05 Validation MSE: 0.0609237477183342
Epoch 0, MSE: 0.103367
Epoch 100, MSE: 0.082745
Epoch 200, MSE: 0.081700
L1 Layers: [1, 50, 1] LR: 0.01 Lambda: 0.0001 Validation MSE: 0.060930024832487106
Epoch 0, MSE: 0.103381
Epoch 100, MSE: 0.082775
Epoch 200, MSE: 0.081742
L1 Layers: [1, 50, 1] LR: 0.01 Lambda: 0.001 Validation MSE: 0.06097126379609108
Epoch 0, MSE: 0.325949
Epoch 100, MSE: 0.083751
Epoch 200, MSE: 0.082864
L2 Layers: [1, 20, 1] LR: 0.001 Lambda: 1e-05 Validation MSE: 0.06320474296808243
Epoch 0, MSE: 0.325949
Epoch 100, MSE: 0.083750
Epoch 200, MSE: 0.082863
L2 Layers: [1, 20, 1] LR: 0.001 Lambda: 0.0001 Validation MSE: 0.06320320069789886
Epoch 0, MSE: 0.325946
Epoch 100, MSE: 0.083740
Epoch 200, MSE: 0.082853
L2 Layers: [1, 20, 1] LR: 0.001 Lambda: 0.001 Validation MSE: 0.06318767368793488
Epoch 0, MSE: 0.141647
Epoch 100, MSE: 0.081436
Epoch 200, MSE: 0.081315
L2 Layers: [1, 20, 1] LR: 0.01 Lambda: 1e-05 Validation MSE: 0.06132214143872261
Epoch 0, MSE: 0.141647
Epoch 100, MSE: 0.081437
Epoch 200, MSE: 0.081318
L2 Layers: [1, 20, 1] LR: 0.01 Lambda: 0.0001 Validation MSE: 0.06132596731185913
Epoch 0, MSE: 0.141642
Epoch 100, MSE: 0.081445
Epoch 200, MSE: 0.081348
L2 Layers: [1, 20, 1] LR: 0.01 Lambda: 0.001 Validation MSE: 0.06136085465550423
Epoch 0, MSE: 0.132834
Epoch 100, MSE: 0.090187
Epoch 200, MSE: 0.086731
L2 Layers: [1, 50, 1] LR: 0.001 Lambda: 1e-05 Validation MSE: 0.06658489257097244
Epoch 0, MSE: 0.132834
Epoch 100, MSE: 0.090183
Epoch 200, MSE: 0.086727
L2 Layers: [1, 50, 1] LR: 0.001 Lambda: 0.0001 Validation MSE: 0.06658072024583817
Epoch 0, MSE: 0.132833
Epoch 100, MSE: 0.090148
Epoch 200, MSE: 0.086689
L2 Layers: [1, 50, 1] LR: 0.001 Lambda: 0.001 Validation MSE: 0.06653878837823868
Epoch 0, MSE: 0.103366
Epoch 100, MSE: 0.082740
Epoch 200, MSE: 0.081691
L2 Layers: [1, 50, 1] LR: 0.01 Lambda: 1e-05 Validation MSE: 0.06092306226491928
Epoch 0, MSE: 0.103365
Epoch 100, MSE: 0.082740
Epoch 200, MSE: 0.081692
L2 Layers: [1, 50, 1] LR: 0.01 Lambda: 0.0001 Validation MSE: 0.06092355400323868
Epoch 0, MSE: 0.103360
Epoch 100, MSE: 0.082743
Epoch 200, MSE: 0.081698
L2 Layers: [1, 50, 1] LR: 0.01 Lambda: 0.001 Validation MSE: 0.06092783808708191
L1 Best Layers: [1, 50, 1]
Learning Rate: 0.01
Lambda: 1e-05
Validation MSE: 0.0609237477183342
----------
L2 Best Layers: [1, 50, 1]
Learning Rate: 0.01
Lambda: 1e-05
Validation MSE: 0.06092306226491928
----------
Regularization L1
L1 Norm: 14.934857368469238
L2 Norm Squared: 3.500621795654297
--------------
Regularization L2
L1 Norm: 14.954706192016602
L2 Norm Squared: 3.5034897327423096
--------------
Epoch 0, MSE: 0.187894
Epoch 100, MSE: 0.021911
Epoch 200, MSE: 0.019841
Regularization: L1
Sigmoid Validation MSE: 0.0609237477183342
ReLU Validation MSE: 0.018134867772459984
Epoch 0, MSE: 0.187893
Epoch 100, MSE: 0.021907
Epoch 200, MSE: 0.019836
Regularization: L2
Sigmoid Validation MSE: 0.06092306226491928
ReLU Validation MSE: 0.018122538924217224
Epoch 0, MSE: 0.187894
Epoch 100, MSE: 0.021911
Epoch 200, MSE: 0.019841
Regularization: L1
Architecture: [1, 50, 1]
Train MSE: 0.019209815189242363
Validation MSE: 0.018134867772459984
Epoch 0, MSE: 0.181678
Epoch 100, MSE: 0.017775
Epoch 200, MSE: 0.015632
Regularization: L1
Architecture: [1, 50, 50, 1]
Train MSE: 0.014254952780902386
Validation MSE: 0.013873571529984474
Epoch 0, MSE: 0.136170
Epoch 100, MSE: 0.018215
Epoch 200, MSE: 0.012687
Regularization: L1
Architecture: [1, 50, 50, 50, 1]
Train MSE: 0.010334054939448833
Validation MSE: 0.011393817141652107
Epoch 0, MSE: 0.187893
Epoch 100, MSE: 0.021907
Epoch 200, MSE: 0.019836
Regularization: L2
Architecture: [1, 50, 1]
Train MSE: 0.01920347847044468
Validation MSE: 0.018122538924217224
Epoch 0, MSE: 0.181676
Epoch 100, MSE: 0.017836
Epoch 200, MSE: 0.015762
Regularization: L2
Architecture: [1, 50, 50, 1]
Train MSE: 0.014393401332199574
Validation MSE: 0.013939055614173412
Epoch 0, MSE: 0.136168
Epoch 100, MSE: 0.018273
Epoch 200, MSE: 0.012679
Regularization: L2
Architecture: [1, 50, 50, 50, 1]
Train MSE: 0.010319025255739689
Validation MSE: 0.011381657794117928
Epoch 0, MSE: 0.224241
Epoch 100, MSE: 0.187019
Epoch 200, MSE: 0.161570
Regularization: L1
Learning Rate: 0.0001
Train MSE: 0.1435040533542633
Validation MSE: 0.11433769762516022
Epoch 0, MSE: 0.220317
Epoch 100, MSE: 0.087539
Epoch 200, MSE: 0.059331
Regularization: L1
Learning Rate: 0.001
Train MSE: 0.04508095234632492
Validation MSE: 0.03781822696328163
Epoch 0, MSE: 0.187894
Epoch 100, MSE: 0.021911
Epoch 200, MSE: 0.019841
Regularization: L1
Learning Rate: 0.01
Train MSE: 0.019209815189242363
Validation MSE: 0.018134867772459984
Epoch 0, MSE: 0.096730
Epoch 100, MSE: 0.015599
Epoch 200, MSE: 0.011147
Regularization: L1
Learning Rate: 0.1
Train MSE: 0.011797788552939892
Validation MSE: 0.017687899991869926
Epoch 0, MSE: 0.621587
Epoch 100, MSE: 0.156913
Epoch 200, MSE: 0.075996
Regularization: L1
Learning Rate: 0.8
Train MSE: 0.20436665415763855
Validation MSE: 0.17478246986865997
Epoch 0, MSE: 0.224241
Epoch 100, MSE: 0.187018
Epoch 200, MSE: 0.161568
Regularization: L2
Learning Rate: 0.0001
Train MSE: 0.14350204169750214
Validation MSE: 0.11433634907007217
Epoch 0, MSE: 0.220317
Epoch 100, MSE: 0.087537
Epoch 200, MSE: 0.059328
Regularization: L2
Learning Rate: 0.001
Train MSE: 0.04507676884531975
Validation MSE: 0.03781527280807495
Epoch 0, MSE: 0.187893
Epoch 100, MSE: 0.021907
Epoch 200, MSE: 0.019836
Regularization: L2
Learning Rate: 0.01
Train MSE: 0.01920347847044468
Validation MSE: 0.018122538924217224
Epoch 0, MSE: 0.096727
Epoch 100, MSE: 0.015563
Epoch 200, MSE: 0.011115
Regularization: L2
Learning Rate: 0.1
Train MSE: 0.011779652908444405
Validation MSE: 0.017691031098365784
Epoch 0, MSE: 0.621667
Epoch 100, MSE: 0.162328
Epoch 200, MSE: 0.075934
Regularization: L2
Learning Rate: 0.8
Train MSE: 0.17010053992271423
Validation MSE: 0.15792222321033478
Epoch 0, MSE: 0.224241
Epoch 100, MSE: 0.187019
Epoch 200, MSE: 0.161570
Regularization: L1
Learning Rate: 0.0001
Lambda: 1e-05
Validation MSE: 0.11433769762516022
Epoch 0, MSE: 0.224241
Epoch 100, MSE: 0.187025
Epoch 200, MSE: 0.161580
Regularization: L1
Learning Rate: 0.0001
Lambda: 0.0001
Validation MSE: 0.11434464901685715
Epoch 0, MSE: 0.224242
Epoch 100, MSE: 0.187086
Epoch 200, MSE: 0.161677
Regularization: L1
Learning Rate: 0.0001
Lambda: 0.001
Validation MSE: 0.11441724002361298
Epoch 0, MSE: 0.220317
Epoch 100, MSE: 0.087539
Epoch 200, MSE: 0.059331
Regularization: L1
Learning Rate: 0.001
Lambda: 1e-05
Validation MSE: 0.03781822696328163
Epoch 0, MSE: 0.220317
Epoch 100, MSE: 0.087569
Epoch 200, MSE: 0.059375
Regularization: L1
Learning Rate: 0.001
Lambda: 0.0001
Validation MSE: 0.03786120191216469
Epoch 0, MSE: 0.220323
Epoch 100, MSE: 0.087866
Epoch 200, MSE: 0.059845
Regularization: L1
Learning Rate: 0.001
Lambda: 0.001
Validation MSE: 0.038294292986392975
Epoch 0, MSE: 0.187894
Epoch 100, MSE: 0.021911
Epoch 200, MSE: 0.019841
Regularization: L1
Learning Rate: 0.01
Lambda: 1e-05
Validation MSE: 0.018134867772459984
Epoch 0, MSE: 0.187899
Epoch 100, MSE: 0.021977
Epoch 200, MSE: 0.019910
Regularization: L1
Learning Rate: 0.01
Lambda: 0.0001
Validation MSE: 0.018262507393956184
Epoch 0, MSE: 0.187947
Epoch 100, MSE: 0.022617
Epoch 200, MSE: 0.020625
Regularization: L1
Learning Rate: 0.01
Lambda: 0.001
Validation MSE: 0.019631963223218918
Epoch 0, MSE: 0.096730
Epoch 100, MSE: 0.015599
Epoch 200, MSE: 0.011147
Regularization: L1
Learning Rate: 0.1
Lambda: 1e-05
Validation MSE: 0.017687899991869926
Epoch 0, MSE: 0.096760
Epoch 100, MSE: 0.015992
Epoch 200, MSE: 0.011692
Regularization: L1
Learning Rate: 0.1
Lambda: 0.0001
Validation MSE: 0.017112797126173973
Epoch 0, MSE: 0.097064
Epoch 100, MSE: 0.019177
Epoch 200, MSE: 0.017197
Regularization: L1
Learning Rate: 0.1
Lambda: 0.001
Validation MSE: 0.017001274973154068
Epoch 0, MSE: 0.621587
Epoch 100, MSE: 0.156913
Epoch 200, MSE: 0.075996
Regularization: L1
Learning Rate: 0.8
Lambda: 1e-05
Validation MSE: 0.17478246986865997
Epoch 0, MSE: 0.620815
Epoch 100, MSE: 0.150810
Epoch 200, MSE: 0.044945
Regularization: L1
Learning Rate: 0.8
Lambda: 0.0001
Validation MSE: 0.16059806942939758
Epoch 0, MSE: 0.614039
Epoch 100, MSE: 0.303862
Epoch 200, MSE: 0.094019
Regularization: L1
Learning Rate: 0.8
Lambda: 0.001
Validation MSE: 0.22978150844573975
Epoch 0, MSE: 0.224241
Epoch 100, MSE: 0.187018
Epoch 200, MSE: 0.161568
Regularization: L2
Learning Rate: 0.0001
Lambda: 1e-05
Validation MSE: 0.11433634907007217
Epoch 0, MSE: 0.224241
Epoch 100, MSE: 0.187016
Epoch 200, MSE: 0.161566
Regularization: L2
Learning Rate: 0.0001
Lambda: 0.0001
Validation MSE: 0.1143333911895752
Epoch 0, MSE: 0.224241
Epoch 100, MSE: 0.186997
Epoch 200, MSE: 0.161541
Regularization: L2
Learning Rate: 0.0001
Lambda: 0.001
Validation MSE: 0.11430283635854721
Epoch 0, MSE: 0.220317
Epoch 100, MSE: 0.087537
Epoch 200, MSE: 0.059328
Regularization: L2
Learning Rate: 0.001
Lambda: 1e-05
Validation MSE: 0.03781527280807495
Epoch 0, MSE: 0.220316
Epoch 100, MSE: 0.087538
Epoch 200, MSE: 0.059338
Regularization: L2
Learning Rate: 0.001
Lambda: 0.0001
Validation MSE: 0.03782567009329796
Epoch 0, MSE: 0.220314
Epoch 100, MSE: 0.087552
Epoch 200, MSE: 0.059434
Regularization: L2
Learning Rate: 0.001
Lambda: 0.001
Validation MSE: 0.03792590647935867
Epoch 0, MSE: 0.187893
Epoch 100, MSE: 0.021907
Epoch 200, MSE: 0.019836
Regularization: L2
Learning Rate: 0.01
Lambda: 1e-05
Validation MSE: 0.018122538924217224
Epoch 0, MSE: 0.187891
Epoch 100, MSE: 0.021929
Epoch 200, MSE: 0.019857
Regularization: L2
Learning Rate: 0.01
Lambda: 0.0001
Validation MSE: 0.01814364828169346
Epoch 0, MSE: 0.187872
Epoch 100, MSE: 0.022152
Epoch 200, MSE: 0.020080
Regularization: L2
Learning Rate: 0.01
Lambda: 0.001
Validation MSE: 0.01838473416864872
Epoch 0, MSE: 0.096727
Epoch 100, MSE: 0.015563
Epoch 200, MSE: 0.011115
Regularization: L2
Learning Rate: 0.1
Lambda: 1e-05
Validation MSE: 0.017691031098365784
Epoch 0, MSE: 0.096728
Epoch 100, MSE: 0.015708
Epoch 200, MSE: 0.011362
Regularization: L2
Learning Rate: 0.1
Lambda: 0.0001
Validation MSE: 0.01734604872763157
Epoch 0, MSE: 0.096739
Epoch 100, MSE: 0.017049
Epoch 200, MSE: 0.013903
Regularization: L2
Learning Rate: 0.1
Lambda: 0.001
Validation MSE: 0.015565572306513786
Epoch 0, MSE: 0.621667
Epoch 100, MSE: 0.162328
Epoch 200, MSE: 0.075934
Regularization: L2
Learning Rate: 0.8
Lambda: 1e-05
Validation MSE: 0.15792222321033478
Epoch 0, MSE: 0.621617
Epoch 100, MSE: 0.160959
Epoch 200, MSE: 0.076180
Regularization: L2
Learning Rate: 0.8
Lambda: 0.0001
Validation MSE: 0.15877600014209747
Epoch 0, MSE: 0.621112
Epoch 100, MSE: 0.158006
Epoch 200, MSE: 0.077120
Regularization: L2
Learning Rate: 0.8
Lambda: 0.001
Validation MSE: 0.16373266279697418
Regularization: L1
Best Architecture: [1, 50, 1]
Best Learning Rate: 0.1
Best Lambda: 0.001
Best Validation MSE: 0.017001274973154068
Regularization: L2
Best Architecture: [1, 50, 1]
Best Learning Rate: 0.1
Best Lambda: 0.001
Best Validation MSE: 0.015565572306513786
Ridge Lambda: 1e-08
Validation MSE: 0.010608451440930367
Ridge Lambda: 1e-06
Validation MSE: 0.010581647045910358
Ridge Lambda: 1e-05
Validation MSE: 0.010555773973464966
Ridge Lambda: 0.0001
Validation MSE: 0.011287705972790718
Ridge Lambda: 0.001
Validation MSE: 0.012440045364201069
Ridge Lambda: 0.01
Validation MSE: 0.014212320558726788
Best Ridge Lambda: 1e-05
Ridge Validation MSE: 0.010555773973464966
L2 Neural Network Validation MSE: 0.015565572306513786
Lasso Lambda: 0.0001
Validation MSE: 0.013159080408513546
Lasso Lambda: 0.0003
Validation MSE: 0.014003987424075603
Lasso Lambda: 0.001
Validation MSE: 0.01656796596944332
Lasso Lambda: 0.01
Validation MSE: 0.027437565848231316
Best Lasso Lambda: 0.0001
Lasso Validation MSE: 0.013159080408513546
L1 Neural Network Validation MSE: 0.017001274973154068
Final L1 Neural Network Test MSE: 0.027470087632536888
Final L2 Neural Network Test MSE: 0.024960337206721306
Final Ridge Test MSE: 0.14054132997989655
Final Lasso Test MSE: 0.028498895466327667
"""