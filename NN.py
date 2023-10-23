# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 22:05:01 2022

@author: Marcin
"""

import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation function
def sigmoid(X):
    out = 1.0 / (1.0 + np.exp(-X))
    return out

# Dervative of sigmoid funcition
def sigmoid_derivative(X):
    return sigmoid(X) * (1 - sigmoid(X))

# Forward progpagation
def forward_propagation(X, w1, w2, predict=False):   
    # Z - before apply activation function
    # A - after apply activation function 
    
    # Calculate multiplication of input X and first layer weights
    A1 = np.dot(X, w1)
    # Apply sigmoid
    Z1 = sigmoid(A1)
    
    # Add bias and do the same as above
    bias = np.ones(Z1.shape[0]).reshape(-1, 1)
    Z1 = np.concatenate((bias, Z1), axis = 1)
    
    A2 = np.dot(Z1, w2)
    Z2 = sigmoid(A2)
    
    # If precition - just return network prediction (Z2)
    if predict:
        return Z2
    # If not - return all matrices before and after sigmoid
    else:
        return A1, Z1, A2, Z2

# Backpropagation
def backpropagation(A1, X, Z1, Z2, Y):
    # Calculate difference betweend output and desired otput
    out_diff = Z2 - Y
    # Propagete inside of network (from back to front) 
    outDiff = np.dot(Z1.T, out_diff)
    # Calculate dot product of out_diff and weights w2 multiplied by sigmoid derivative of A1
    inside_diff = (out_diff.dot(w2[1:, :].T)) * sigmoid_derivative(A1)
    # Dot product of X and inside_diff
    insideDiff = np.dot(X.T, inside_diff)
    
    return out_diff, insideDiff, outDiff

# Initialize weights
def initialize(input_size, output_size, hidden_units_w1, hidden_w2):
    # Random weights
    w1 = np.random.randn(input_size, hidden_units_w1)
    w2 = np.random.randn(hidden_w2, output_size) 
    
    return w1, w2

# Define input data with bias and output values
X = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
Y = np.array([0, 1, 1, 0]).reshape(-1,1)

# Number of neurons in layers
input_size = X.shape[1]
hidden_units_w1 = 5
hidden_w2 = hidden_units_w1 + 1
output_size = 1

# Initialize random weights
w1, w2 = initialize(input_size, output_size, hidden_units_w1, hidden_w2)

# Define learning rate
learning_rate = 0.08
# Lists for costs (errors)
costs = []
# Desired number of epochs
epochs = 10000
# Y data shape - to weight modification
m = Y.shape[0]

# Training process
for i in range(1, epochs+1):
    # Put out data into forword propagation
    A1, Z1, A2, Z2 = forward_propagation(X, w1, w2)
    # Backpropagation
    out_diff, insideDiff, outDiff = backpropagation(A1, X, Z1, Z2, Y)   
    
    # Modify weights
    w1 = w1 - learning_rate * (1/m) * insideDiff
    w2 = w2 - learning_rate * (1/m) * outDiff
    
    # Costs (differences betweend desired output) - mean 
    c = np.mean(np.abs(out_diff))
    costs.append(c)
    
    if i%100 == 0:
        print('Iteration: %f, cost: %f' % (i, c))
        
print('Completed.')

# Predict:
pred = forward_propagation(X, w1, w2, True)
print('Pred. percentage:')
print(pred)
pred_rounded = np.round(pred)
print('Predictions:')
print(pred_rounded)

# Plot error curve
plt.plot(costs)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Training erroro curve')
