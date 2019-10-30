# Backprop on the Seeds Dataset
from random import seed
from random import randrange
from random import random
from csv import reader
import csv
from math import exp

# import pandas as pn
# import matplotlib.pyplot as plt

# from sklearn.neural_network import MLPClassifier as perceptron
# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(train_dataset, test_dataset, algorithm, n_folds, *args):
    train_folds = cross_validation_split(train_dataset, n_folds)
    test_folds = cross_validation_split(test_dataset, n_folds)
    scores = list()

    for i in range(len(train_folds)):
        train_set = list(train_folds)
        train_set.remove(train_folds[i])
        train_set = sum(train_set, [])

        test_set = list(test_folds)
        test_set.remove(test_folds[i])
        test_set = sum(test_set, [])

        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in train_folds[i]]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)

        file = open('test/Y_test.csv', 'w', newline='')
        output_writer = csv.writer(file)
        for row in predicted:
            output_writer.writerow([row])
        file.close()

    return scores


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    initial_expected = [0 for i in range(n_outputs)]
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = initial_expected
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return (predictions)


# # Test Backprop on Seeds dataset
#   seed(1)
#   # load and prepare data
#   training_inputs_path_file = 'train/X_train.csv'
#   training_inputs_data_set = load_csv(training_inputs_path_file)
#   for i in range(len(training_inputs_data_set[0])):
#       str_column_to_float(training_inputs_data_set, i)
#
#   training_outputs_file = 'train/Y_train.csv'
#   training_outputs_data_set = load_csv(training_outputs_file)
#   # convert class column to integers
#   str_column_to_int(training_outputs_data_set, 0)

# Test Backprop on Seeds dataset
seed(1)
training_input_reader = csv.reader(open("train/X_train.csv"))
training_output_reader = csv.reader(open("train/Y_train.csv"))
f = open("X_Y_train.csv", "w")
writer = csv.writer(f)

# for row in training_input_reader:
#     writer.writerow(row)
# for row in training_output_reader:
#     writer.writerow(row)
training_input_reader = list(training_input_reader)
training_output_reader = list(training_output_reader)
file_lenght = sum(1 for row in training_input_reader)
for i in range(file_lenght):
    writer.writerow(training_input_reader[i] + training_output_reader[i])
f.close()

# load and prepare data
train_set_file = 'X_Y_train.csv'
train_data_set = load_csv(train_set_file)
for i in range(len(train_data_set[0]) - 1):
    str_column_to_float(train_data_set, i)
# convert class column to integers
str_column_to_int(train_data_set, len(train_data_set[0]) - 1)
# normalize input variables
# minmax = dataset_minmax(train_data_set)
# normalize_dataset(train_data_set, minmax)

test_set_file = 'test/X_test.csv'
test_data_set = load_csv(test_set_file)
for i in range(len(test_data_set[0])):
    str_column_to_float(test_data_set, i)
# normalize input variables
# minmax = dataset_minmax(test_data_set)
# normalize_dataset(test_data_set, minmax)

# evaluate algorithm
n_folds = 2
l_rate = 0.5
n_epoch = 1000
n_hidden = 2
scores = evaluate_algorithm(train_data_set, test_data_set, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
