# Introduce Pytorch

#Import necessary libraries
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

# Function for Sigmoid 

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# Function for softmax

def softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x), dim=0)

# Function for linear layer
def linear_layer(x, w, b):
    return torch.matmul(x, w) + b

# LOSS FUNCTIONS #

## Function for Mean Square Error - Loss:
def squared_loss(y_hat, y):
    return ((y_hat - y.reshape(y_hat.shape)) ** 2 / 2).mean()

## Function for Cross Entropy - Loss:
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y]).mean()