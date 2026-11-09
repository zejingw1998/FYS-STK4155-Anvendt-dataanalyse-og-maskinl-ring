# FYS-STK4155 – Project 2

**Neural Networks for Regression and Classification**  
University of Oslo, Autumn 2026

**Group members:** Zejing Wang and Zhenwei Liu

## Project description

In this project, we implement feed-forward neural networks using Python and PyTorch tensors, with our own forward propagation and backpropagation calculations. We study neural network performance on two tasks:

- **Regression:** fitting the one-dimensional Runge function and comparing the results with OLS, Ridge and Lasso regression.
- **Classification:** recognizing handwritten digits using the full MNIST dataset and comparing neural networks with logistic regression.

We investigate activation functions, learning rates, optimization methods, network depth and width, and L1/L2 regularization.

## Files

The project report is provided as `Project2.pdf`. The Python programs are in the `Code/` folder:

| File | Description |
| --- | --- |
| `functions.py` | Runge function, activation functions, their derivatives and MSE utilities. |
| `NeuralNetwork.py` | Our feed-forward neural network implementation, including forward propagation, backpropagation and training methods. |
| `Partb.py` | Regression with the Runge function; comparison with OLS, different network architectures and optimization methods. |
| `Partc.py` | Experiments with Sigmoid, ReLU and Leaky ReLU; network depth, width and overfitting. |
| `Partd.py` | L1/L2 regularization, hyperparameter experiments and comparisons with Ridge and Lasso. |
| `Parte.py` | Full MNIST classification, logistic regression baseline, architecture and hyperparameter comparisons, and confusion matrices. |

## Requirements

- Python 3
- NumPy
- PyTorch
- Matplotlib
- Scikit-learn

Install the packages using:

```bash
python -m pip install -r requirements.txt
```

## How to run

Open a terminal in the `Project2` folder. Run the scripts from the `Code` directory so that local imports work correctly:

```bash
cd Code
python Partb.py
python Partc.py
python Partd.py
python Parte.py
```

Each script can also be run separately. The scripts print numerical results and display figures using Matplotlib. Figures are displayed during execution; they are not automatically saved by these scripts.

`Parte.py` checks for `mnist.npz` in the current working directory and downloads it from the TensorFlow-hosted MNIST dataset URL if it is missing. An internet connection is therefore needed for the first MNIST run unless the file is already present.

The experiments use fixed random seeds where specified in the scripts to support reproducibility. Results may still vary slightly depending on software versions and the computing environment.

## Main results

- For Runge regression, neural networks are compared with polynomial regression using training, validation and test mean squared error (MSE).
- For MNIST classification, the final two-hidden-layer neural network with 128 neurons in each hidden layer achieved **97.39% test accuracy**, compared with **91.56%** for the logistic regression baseline.
- Further results, tables, figures and discussion are provided in `Project2.pdf`.

## Report and source code

The complete report, implementation and experimental discussion are part of this Project 2 submission.
