import numpy as np
import sys
import random
import math

def convertVector(data):

    unique = np.unique(data)
    indices = [i for i in range(len(unique))]
    mapping = dict(zip(unique, indices))
    reverseMap = dict(zip(indices, unique))
    target = []
    
    for i in range(len(data)):
        vector = [0.0 for j in range(len(unique))]
        vector[mapping.get(data[i])] = 1.0
        target.append(vector)
    
    return target, len(unique), reverseMap

def getSigmaVal(x):
    try:
        return 1 / (1 + math.exp(-x))
    except:
        return 0

def readFileContents(fileName):
    file = open(fileName, "r")
    data = []
    output = []
    maxVal = sys.float_info.min

    for row in file:
        temp = row.split()
        length = len(temp)
        intermediate = []

        for i in range(length-1):
            intermediate.append(float(temp[i]))
            if float(temp[i]) > maxVal:
                maxVal = float(temp[i])
        output.append(temp[length-1])
        data.append(intermediate)

    data = [[float(num/maxVal) for num in row] for row in data]

    return data, output


def trainingData(training_file, layers, units_per_layer, rounds):

    trainingInput, output = readFileContents(training_file)
    trainingOutput, numClasses, reverseMap = convertVector(output)
    num_attributes = len(trainingInput[0])    

    unitsInEachLayer = [units_per_layer for i in range(layers)]
    unitsInEachLayer[0] = num_attributes
    unitsInEachLayer[layers-1] = numClasses

    b = [[] for i in range(layers)]
    w = [[] for i in range(layers)]
    for l in range(1, layers):
        b[l] = [random.uniform(-0.05, 0.05) for i in range(unitsInEachLayer[l])]
        w[l] = [[random.uniform(-0.05, 0.05) for j in range(unitsInEachLayer[l-1])] for i in range(unitsInEachLayer[l])]

    learning_rate = 1.0
    for r in range(rounds):
        for n in range(len(trainingInput)):
            z = [[] for i in range(layers)]
            a = [[] for i in range(layers)]

            z[0] = [0 for i in range(num_attributes)]
            for i in range(num_attributes):
                z[0][i] = trainingInput[n][i]
                
            for l in range(1, layers):
                a[l] = [0.0 for i in range(unitsInEachLayer[l])]
                z[l] = [0.0 for i in range(unitsInEachLayer[l])]
                for i in range(unitsInEachLayer[l]):
                    weighted_sum = 0.0
                    for j in range(unitsInEachLayer[l-1]):
                        weighted_sum += (w[l][i][j] * z[l-1][j])
                    a[l][i] = b[l][i] + weighted_sum
                    z[l][i] = getSigmaVal(a[l][i])

            delta =[[] for i in range(layers)]
            delta[layers-1] = [0 for i in range(numClasses)]

            for i in range(numClasses):
                delta[layers-1][i] = (z[layers-1][i] - trainingOutput[n][i]) * z[layers-1][i] * (1.0-z[layers-1][i])
            
            for l in range(layers-2, 0, -1):
                delta[l] = [0 for i in range(unitsInEachLayer[l])]
                for i in range(unitsInEachLayer[l]):
                    sum = 0.0
                    for k in range(unitsInEachLayer[l+1]):
                        sum += (delta[l+1][k] * w[l+1][k][i])
                    delta[l][i] = sum * z[l][i] * (1 - z[l][i])

            for l in range(1, layers):
                for i in range(unitsInEachLayer[l]):
                    b[l][i] -= (learning_rate * delta[l][i])
                    for j in range(unitsInEachLayer[l-1]):
                        w[l][i][j] -= (learning_rate * delta[l][i] * z[l-1][j])
        learning_rate *= 0.98
    return b, w, reverseMap, numClasses, unitsInEachLayer


def test_data(test_file, layers, b, w, reverseMap, numClasses, unitsInEachLayer):
    
    testInput, testOutput = readFileContents(test_file)
    
    num_attributes = len(testInput[0])
    accuracy = 0.0

    for n in range(len(testInput)):
        z = [[] for i in range(layers)]
        a = [[] for i in range(layers)]

        z[0] = [0.0 for i in range(num_attributes)]
        for i in range(num_attributes):
            z[0][i] = testInput[n][i]
        
        for l in range(1, layers):
            a[l] = [0.0 for i in range(unitsInEachLayer[l])]
            z[l] = [0.0 for i in range(unitsInEachLayer[l])]
            for i in range(unitsInEachLayer[l]):
                weighted_sum = 0.0
                for j in range(unitsInEachLayer[l-1]):
                    weighted_sum += (w[l][i][j] * z[l-1][j])
                a[l][i] = b[l][i] + weighted_sum
                z[l][i] = getSigmaVal(a[l][i])

        argmax = []
        maxVal = -1

        for i in range(numClasses):
            if z[layers-1][i] > maxVal:
                maxVal = z[layers-1][i]
                argmax.clear()
                argmax.append(i)
            elif z[layers-1][i] == maxVal:
                argmax.append(i)
    
        predicted = [reverseMap.get(n) for n in argmax]
        true = testOutput[n]
        actual_predicted = predicted[0]

        if len(predicted)==1 and int(predicted[0]) == int(true):
            curr_accuracy = 1.0
        else:
            try:
                index = predicted.index(true)
                actual_predicted = predicted[index]
                curr_accuracy = float(1.0/len(predicted))
            except ValueError:
                curr_accuracy = 0.0
        accuracy += curr_accuracy

        print('ID={:5d}, predicted={:3d}, true={:3d}, accuracy={:4.2f}'.format(n+1, int(actual_predicted), int(true), curr_accuracy))
    print('classification accuracy={:6.4f}'.format(accuracy/len(testInput)))

def neural_network(training_file, test_file, layers, units_per_layer, rounds):
    b, w, reverseMap, numClasses, unitsInEachLayer = trainingData(training_file, layers, units_per_layer, rounds)
    test_data(test_file, layers, b, w, reverseMap, numClasses, unitsInEachLayer)
    
neural_network('pendigits_string_training.txt','pendigits_string_test.txt',3,20,20)
