# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 11:27:05 2022

@author: Marcin
"""


import pandas as pd
import numpy as np

# Find k nearest neighbours
def find_kNN(train, y_train, test_sample, k, metric = 'euclidean'):
    distances = []
    y_train_set = list(y_train)
    
    # Calculate distances based on training data and chosen metric
    dist = 0
    for train_point in train.values:
        if metric == 'euclidean':
            dist = euclidean_distance(train_point, test_sample)
        elif metric == 'manhattan':
            dist = manhattan_distance(train_point, test_sample)
        elif metric == 'chebyshev':
            dist = chebyshev_distance(train_point, test_sample)
        distances.append(dist)
    # Sort distances
    distances, y_train_set = zip(*sorted(zip(distances, y_train_set)))
    
    # Get k nearest neighbours
    kNN = []
    for neighbour in range(k):
        kNN.append(y_train_set[neighbour])
    return kNN

# Methot will choose the most frequent element (class) in kNN list
def vote_kNN(kNN):
    return max(set(kNN), key = kNN.count)

# Function will calculate an Euclidean distance
def euclidean_distance(p1, p2):
    dist = 0 
    for i in range(len(p1)):
        dist += (p1[i] - p2[i])**2
    
    return np.sqrt(dist)

# Function will calculate a Manhattan distance
def manhattan_distance(p1, p2):
    dist = 0 
    for i in range(len(p1)):
        dist += abs(p1[i] - p2[i])
    
    return dist

# Function will calculate a Manhattan distance
def chebyshev_distance(p1, p2):
    dist = []
    for i in range(len(p1)):
        dist.append(np.abs(p1[i] - p2[i]))
    
    return max(dist)

# Calculate an accuracy
def calc_accuracy(pred, real):
    acc = 0
    for i in range(len(pred)):
        if(pred[i] == real[i]):
            acc +=1
    acc = acc / len(pred)
    
    return acc

# Min-Max normalization 
def min_max_norm(df, target = 'irisType'):
    # Normalize columns (no target)
        for i in df.columns:
            if i != target:
                df[i] = (df[i] - df[i].min()) / (df[i].max() - df[i].min())



# Load rows from txt file
rows = []
with open('iris.txt','r',encoding = 'latin-1') as file:
    data = file.readlines()
    rows = [row.split('\t') for row in data]
    
iris_type = []
with open('iris_type.txt','r',encoding = 'latin-1') as file:
    data = file.readlines()
    iris_type = [row.split('\t')[0] for row in data]

iris_type[4] = 'irisType'    

# Cnovert data into dataframe (pandas)
df = pd.DataFrame(rows, columns=iris_type).astype(float)

# Normalization
min_max_norm(df)

# Split into train and test data
mask = np.random.rand(len(df)) < 0.75

train_data = df[mask]
test_data = df[~mask]

X_train = train_data.drop(columns=['irisType'])
X_test = test_data.drop(columns=['irisType'])
y_train = train_data['irisType']
y_test = test_data['irisType']

# EUCLIDEAN
# Let's classify X_test data for 3 neighbours 
predictions = []
k = 3
for x_t in X_test.values:
    kNN = find_kNN(X_train, y_train, x_t, k, metric='euclidean')
    predictions.append(vote_kNN(kNN))

# Calculate accuracy

acc_3NN = calc_accuracy(predictions, list(y_test))

print('my k-NN alg. k=3, metric=euclidean accuracy: %f' % acc_3NN)


# MANHATTAN
# Let's classify X_test data for 3 neighbours 
predictions = []
k = 3
for x_t in X_test.values:
    kNN = find_kNN(X_train, y_train, x_t, k, metric='manhattan')
    predictions.append(vote_kNN(kNN))

# Calculate accuracy

acc_3NN = calc_accuracy(predictions, list(y_test))

print('my k-NN alg. k=3, metric=manhattan accuracy: %f' % acc_3NN)

# CHEBYSHEV
# Let's classify X_test data for 3 neighbours 
predictions = []
k = 3
for x_t in X_test.values:
    kNN = find_kNN(X_train, y_train, x_t, k, metric='chebyshev')
    predictions.append(vote_kNN(kNN))

# Calculate accuracy

acc_3NN = calc_accuracy(predictions, list(y_test))

print('my k-NN alg. k=3, metric=chebyshev accuracy: %f' % acc_3NN)


import matplotlib.pyplot as plt

# Plot accuracy for k=1..100, euclidean
acc_steps = []
for k in range(1, 101):
    predictions = []
    for x_t in X_test.values:
        kNN = find_kNN(X_train, y_train, x_t, k, metric='euclidean')
        predictions.append(vote_kNN(kNN))

    acc_steps.append(calc_accuracy(predictions, list(y_test)))

# Plot accuracy
plt.plot(list(range(1,101)), acc_steps)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.title('kNN accuracy, euclidean metric')

# Plot accuracy for k=1..100, manhattan
acc_steps = []
for k in range(1, 101):
    predictions = []
    for x_t in X_test.values:
        kNN = find_kNN(X_train, y_train, x_t, k, metric='manhattan')
        predictions.append(vote_kNN(kNN))

    acc_steps.append(calc_accuracy(predictions, list(y_test)))

# Plot accuracy
plt.plot(list(range(1,101)), acc_steps)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.title('kNN accuracy, manhattan metric')


# Plot accuracy for k=1..100, chebyshev
acc_steps = []
for k in range(1, 101):
    predictions = []
    for x_t in X_test.values:
        kNN = find_kNN(X_train, y_train, x_t, k, metric='chebyshev')
        predictions.append(vote_kNN(kNN))

    acc_steps.append(calc_accuracy(predictions, list(y_test)))

# Plot accuracy
plt.plot(list(range(1,101)), acc_steps)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.title('kNN accuracy, chebyshev metric')

# One vs all classification
def one_vs_all(x_data, y_data, k, metric):
    correct_preds = 0
    for sample_no in range(len(x_data)):
        # Get actual sample (one)
        x_sample = x_data.loc[sample_no]
        y_sample = y_data.loc[sample_no]
        # Drop one and select all
        x_train_all = x_data.drop(index=sample_no)
        y_train_all = y_data.drop(index=sample_no)
        
        # Build a classifier
        kNN_sample = find_kNN(x_train_all, y_train_all, x_sample, k, metric=metric)
        # Make a prediction for one sample
        pred = vote_kNN(kNN_sample)
        # Compare with desired value
        if(pred == y_sample):
            # Increment correct preds
            correct_preds += 1
    
    # Return mean accuracy
    return correct_preds/len(y_data)


# Let's test our models for some k_values - 1vsAll method
for k_value in range(1, 20):
    print('k = %f' % k_value)
    simpleOneVsAll_euclidean = one_vs_all(df.drop(columns=['irisType']), df['irisType'], k=k_value, metric='euclidean')
    print('One vs all method, euclidean accuracy: %f' % simpleOneVsAll_euclidean)
    
    simpleOneVsAll_manhattan = one_vs_all(df.drop(columns=['irisType']), df['irisType'], k=k_value, metric='manhattan')
    print('One vs all method, manhattan accuracy: %f' % simpleOneVsAll_manhattan)
    
    simpleOneVsAll_chebyshev = one_vs_all(df.drop(columns=['irisType']), df['irisType'], k=k_value, metric='chebyshev')
    print('One vs all method, chebyshev accuracy: %f' % simpleOneVsAll_chebyshev)
    
    print('\n')