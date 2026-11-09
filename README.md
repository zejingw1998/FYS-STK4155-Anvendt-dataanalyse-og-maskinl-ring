# FYS-STK4155 – Applied Data Analysis and Machine Learning

**University of Oslo (UiO) · Autumn 2026**

This repository contains our course projects, Python implementations, reports and weekly exercises for **FYS-STK4155: Applied Data Analysis and Machine Learning**.

## Contributors

- Zejing Wang
- Zhenwei Liu

## Repository structure

| Directory | Contents |
| --- | --- |
| [Project1](Project1/) | Polynomial regression and model evaluation using OLS, Ridge and Lasso, including resampling and bias–variance analysis of the Runge function. |
| [Project2](Project2/) | Feed-forward neural networks for regression and MNIST classification, implemented using PyTorch tensors and manual backpropagation. |
| [Project3](Project3/) | Reserved for the third course project; no project materials have been added yet. |
| [Week_exercises](Week_exercises/) | Weekly exercises and notebooks covering course topics. |

## Project 1 – Regression analysis

Project 1 investigates polynomial regression on the Runge function and compares ordinary least squares (OLS), Ridge and Lasso regression, with attention to model complexity, regularization and evaluation.

- [Project 1 report](Project1/FYS_STK_4155_Project_1_ZejingWang_ZhenweiLiu.pdf)
- [Project 1 Python code](Project1/Code/)

## Project 2 – Neural networks for regression and classification

For Project 2, we implemented feed-forward neural networks with our own forward propagation and backpropagation routines. We evaluated different architectures, activation functions, optimization methods and L1/L2 regularization.

The regression experiments compare neural networks with polynomial OLS, Ridge and Lasso models on the Runge function. The classification experiments use the full MNIST handwritten-digit dataset and compare neural networks with a logistic regression baseline.

The final neural network achieved **97.39% test accuracy** on MNIST, compared with **91.56%** for logistic regression.

- [Project 2 report](Project2/Project2.pdf)
- [Project 2 README: setup, scripts and results](Project2/README.md)
- [Project 2 Python code](Project2/Code/)
- [Project 2 dependencies](Project2/requirements.txt)

## Running Project 2

Detailed setup and run instructions are in [Project2/README.md](Project2/README.md). For a quick start, run the following commands from the repository root:

```bash
python -m pip install -r Project2/requirements.txt
cd Project2/Code
python Partb.py
python Partc.py
python Partd.py
python Parte.py
```

Each script can be run separately. The scripts display plots with Matplotlib, and `Parte.py` can download the MNIST dataset on its first run if the dataset is not already present.

## Notes

The reports contain the experimental settings, numerical results, figures and critical discussions. Where specified in the code, fixed random seeds are used to improve reproducibility.

This repository was prepared as coursework for FYS-STK4155 at the University of Oslo.
