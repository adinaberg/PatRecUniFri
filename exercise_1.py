#### Exercise 1 Pattern Recognition spring 2018 ####

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import time


def loadData(filename):
    data = genfromtxt(filename, delimiter = ',')
    dataLabels = data[:,0] # labels of each sample
    dataFeatures = np.delete(data, (0), axis = 1)
    return (dataLabels, dataFeatures)

def show(imgArray):
    img = []
    line = []
    for l in range(0,27):
        line = imgArray[l*28:(l+1)*28]
        img.append(line)
    plt.imshow(img, cmap = 'gray_r')
    return img

def euclideanDist(img1, img2):
    return sum(np.abs(img1 - img2))

# find nearest neighbour to img given training set and returns classification of image
def NNEuclidean(img, trainFeatures, trainLabels):
    minDistance = np.Inf
    label = np.NaN
    for i in range(0,len(trainFeatures)):
        currDistance = euclideanDist(img, trainFeatures[i])
        if currDistance < minDistance:
            minDistance = currDistance
            label = trainLabels[i]
    return (label, minDistance)


def KNNEuclideanSimple(img, trainFeatures, trainLabels, k):
    labels = [np.NaN] * k
    minDistances = [np.Inf] * k # keep distances to nearest neighbours in sorted list
    
    for i in range(0,len(trainFeatures)):
        currDistance = euclideanDist(img, trainFeatures[i])
        index = 0
        if currDistance < minDistances[0]: # one of minDistances should be updated
            while (index < k - 1 and currDistance < minDistances[index + 1]):
                index += 1
            minDistances[index] = currDistance
            labels[index] = trainLabels[i]
    
    labelFreqs = [0] * 10
    for label in labels:
        labelFreqs[int(label)] += 1
    
    KNNlabel = np.NaN
    maxFreq = 0
    for i in range(0,10):
        if labelFreqs[i] > maxFreq:
            maxFreq = labelFreqs[i]
        elif labelFreqs[i] == maxFreq: # we have a tie
            KNNlabel = labels[k-1] # assign 1-NN label
            break
        KNNlabel = labelFreqs.index(maxFreq) # kanske snabbare att skriva över KNNlabel varje gång en bättre hittats istället för att söka i listan i slutet. Testa
                
    return (minDistances, labels, KNNlabel, maxFreq)
    ### fram hit inte säker

def KNNEuclidean(trainFeatures, trainLabels, testFeatures, testLabels, k):
    errors = 0
    for i in range(0,len(testFeatures)):
        minDistances, labels, KNNlabel, maxFreq = KNNEuclideanSimple(testFeatures[i], trainFeatures, trainLabels, k)
        print('classified sample nr ', i)
        print('true label: ', testLabels[i], ' labels:', labels, 'maxFreq: ', maxFreq)
        if not KNNlabel == testLabels[i]:
            errors += 1
    errorRate = errors/len(testLabels)
    return(errorRate)
# may be put into a function instead...
#testLabels, testFeatures = loadData('test.csv')
#trainLabels, trainFeatures = loadData('train.csv')

show(testFeatures[120])

start = time.time()
#a, minDistance = NNEuclidean(testFeatures[10], trainFeatures, trainLabels)
#minDistances, labels, KNNlabel=  KNNEuclideanSimple(testFeatures[120], trainFeatures, trainLabels, 10)
errorRate = KNNEuclidean(trainFeatures[0:100], trainLabels[0:100], testFeatures[0:20], testLabels[0:20], 7)
end = time.time()
elapsedTime = end - start
print('elapsedTime:' ,elapsedTime)

print('errorRate:', errorRate)


print("exercise_1 done")











