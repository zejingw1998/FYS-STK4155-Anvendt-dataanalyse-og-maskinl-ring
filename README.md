# FYS-STK4155 – Applied Data Analysis and Machine Learning

**University of Oslo (UiO) | Autumn 2026**

This repository contains our projects, reports, Python code and weekly exercises for **FYS-STK4155: Applied Data Analysis and Machine Learning**.

## Contributors

- Zejing Wang
- Zhenwei Liu

## Repository overview

| Folder | Description |
| --- | --- |
| [Project1](Project1/) | Polynomial regression, optimization and statistical model evaluation using the Runge function. |
| [Project2](Project2/) | Feed-forward neural networks for regression and classification, including the full MNIST dataset. |
| [Project3](Project3/) | Reserved for Project 3; materials have not been added yet. |
| [Week_exercises](Week_exercises/) | Weekly exercises and notebooks from the course. |

## Project 1 – Regression Analysis and Optimization for the Runge Function

In Project 1, we studied polynomial regression using the one-dimensional Runge function. We implemented and compared **Ordinary Least Squares (OLS), Ridge and Lasso regression**, and investigated how polynomial degree, noise, data size and regularization affect prediction error.

We also examined:

- Gradient descent and optimization methods: Plain Gradient Descent, Momentum, AdaGrad, RMSprop and Adam.
- Mini-batch Stochastic Gradient Descent (SGD), learning rates and batch sizes.
- Bootstrap resampling and the bias–variance trade-off.
- 5-fold and 10-fold cross-validation for model and hyperparameter selection.

In the final 10-fold cross-validation comparison, OLS and Ridge achieved nearly identical minimum MSE values of approximately **0.01111** and **0.01110**, respectively. Lasso achieved approximately **0.01963**. These results were obtained under the experimental settings described in the report.

**Files:** [Report](Project1/FYS_STK_4155_Project_1_ZejingWang_ZhenweiLiu.pdf) · [Python code](Project1/Code/) · [Figures](Project1/Figures/)

## Project 2 – Neural Networks for Regression and Classification

In Project 2, we developed **feed-forward neural networks** using PyTorch tensors, while implementing forward propagation and backpropagation ourselves instead of relying on a predefined network model.

The project includes two main tasks:

- **Regression:** approximating the Runge function and comparing neural networks with OLS, Ridge and Lasso from Project 1.
- **Classification:** recognizing handwritten digits from the full MNIST dataset and comparing neural networks with a logistic regression baseline.

We investigated Sigmoid, ReLU and Leaky ReLU; different network depths and widths; gradient descent, SGD, Adam and RMSprop; and L1/L2 regularization. We also compared our manually calculated gradients against PyTorch automatic differentiation.

The final MNIST neural network, with two hidden layers of 128 neurons each, achieved **97.39% test accuracy**, compared with **91.56%** for logistic regression.

**Files:** [Report](Project2/Project2.pdf) · [Detailed README](Project2/README.md) · [Python code](Project2/Code/) · [Requirements](Project2/requirements.txt)

### Running Project 2

Install the dependencies from the repository root:

```bash
python -m pip install -r Project2/requirements.txt
```

Run the scripts from the `Project2/Code` directory:

```bash
cd Project2/Code
python Partb.py
python Partc.py
python Partd.py
python Parte.py
```

The scripts can be run separately. They print numerical results and display plots with Matplotlib; the plots are not automatically saved by the scripts. The MNIST script uses `mnist.npz` from its working directory and can download the dataset if it is missing and an internet connection is available.

For the purpose of each script and additional instructions, see the [Project 2 README](Project2/README.md).

## Weekly exercises

The [Week_exercises](Week_exercises/) folder contains course exercises and notebooks, organized by week.

## Reproducibility

The project reports document the methods, numerical experiments, figures and conclusions. Fixed random seeds are used where specified in the scripts; numerical results may vary slightly across software versions or computing environments.
