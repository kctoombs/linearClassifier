#!/usr/bin/env python3
import sys
import math
import numpy as np

trainData = []         #Holds the data from the training set
numExamples = []       #Holds number of examples per class 
centroids = []         #Holds the centroid of each class

trainingSet = sys.argv[1]  #Extract training data file from command line
with open(trainingSet) as f:
    firstLine = f.readline().strip().split()
    dimension = firstLine[0]   #First num in line 1 is dimensionality
    
    if(len(firstLine) != 4):
        sys.exit("Incorrect number of classes. Exiting.")

    #Extract the number of examples for each class
    for num in firstLine[1:]:
        numExamples.append(int(num))

    #Add the rest of the data to data[]
    for line in f.readlines():
        #Turn each point into a list of numbers
        trainData.append([float(num) for num in line.split()])


index = 0
for n in numExamples:
    #Compute the centroid of each data set
    centroid = np.mean(trainData[index:index+n], 0)
    centroids.append(centroid)
    index += n
#centroids[0] is centroid of A, centroids[1] is centroid of B, centroids[1] is centroid of C
#print(centroids)

#These are vectors of size dimensionality
middleAB = centroids[0] - centroids[1]
middleAC = centroids[0] - centroids[2]
middleBC = centroids[1] - centroids[2]
#print(middleBC)

#These are single floating point numbers
boundaryAB = (middleAB.dot((centroids[0] + centroids[1]).T))/2.0
boundaryAC = (middleAC.dot((centroids[0] + centroids[2]).T))/2.0
boundaryBC = (middleBC.dot((centroids[1] + centroids[2]).T))/2.0
#print(boundaryBC)
'''
print(boundaryAB)
print(boundaryAC)
print(boundaryBC)
'''

def computeClass(data, middleAB, middleAC, middleBC, boundaryAB, boundaryAC, boundaryBC):
    #Class A vs. Class B
    computedAB = (middleAB.dot([float(i) for i in data])) - boundaryAB
    if(computedAB >= 0):
        #Class A wins, compare against Class C
        computedAC = (middleAC.dot([float(i) for i in data])) - boundaryAC
        if(computedAC >= 0):
            return "A"
        else:
            return "C"

    else:
        #Class B wins, compare against Class C
        computedBC = (middleBC.dot([float(i) for i in data])) - boundaryBC
        if(computedBC >= 0):
            return "B"
        else:
            return "C"

testData = []           #Holds the data from the testing file
numTestExamples = []    #Holds number of examples per class

testingSet = sys.argv[2]  #Extract testing data file from command line
with open(testingSet) as f:
    testFirstLine = f.readline().strip().split()
    testDimension = testFirstLine[0]   #First num in line 1 is dimensionality
    
    if(len(testFirstLine) != 4):
        sys.exit("Incorrect number of classes. Exiting.")

    #Extract the number of examples for each class
    for num in testFirstLine[1:]:
        numTestExamples.append(int(num))
        
    #Add the rest of the data to data[]
    for line in f.readlines():
    #Turn each point into a list of numbers
        testData.append([float(num) for num in line.split()])

TPA = 0       #True Positives class A
TPB = 0       #True Positives class B
TPC = 0       #True Positives class C
correct = 0   #Number of correct classifications
incorrect = 0 #Number of incorrect classifications

#Contingency Table. 3x3 all entries initialized to 0
cTable = [[0.0 for x in range(3)] for y in range(3)]


for i in range(0, numTestExamples[0]):
    result = computeClass(testData[i], middleAB, middleAC, middleBC,
                          boundaryAB, boundaryAC, boundaryBC)
    #print(testData[i])
    #print(i)
    if(result == "A"):
        #True positive
        TPA +=1
        correct += 1
        cTable[0][0] += 1

    elif(result == "B"):
        incorrect += 1
        cTable[1][0] += 1

    else:
        incorrect += 1
        cTable[2][0] += 1

for i in range(numTestExamples[0], numTestExamples[0]+numTestExamples[1]):
    result = computeClass(testData[i], middleAB, middleAC, middleBC,
                          boundaryAB, boundaryAC, boundaryBC)
    #print(i)
        
    if(result == "B"):
        #True positive
        TPB += 1
        correct += 1
        cTable[1][1] += 1
        
    elif(result == "A"):
        incorrect += 1
        cTable[0][1] += 1

    else:
        incorrect += 1
        cTable[2][1] += 1

for i in range(numTestExamples[0]+numTestExamples[1],
               numTestExamples[0] + numTestExamples[1] + numTestExamples[2]):
    result = computeClass(testData[i], middleAB, middleAC, middleBC,
                          boundaryAB, boundaryAC, boundaryBC)
    #print(testData[i])
    #print(i)
    if(result == "C"):
        #True positive
        TPC += 1
        correct += 1
        cTable[2][2] += 1

    elif(result == "A"):
        incorrect += 1
        cTable[0][2] += 1

    else:
        incorrect += 1
        cTable[1][2] += 1


#print(cTable)

#Compute True Positive Rate
TPRateA = float(cTable[0][0]/numTestExamples[0])
TPRateB = float(cTable[1][1]/numTestExamples[1])
TPRateC = float(cTable[2][2]/numTestExamples[2])
TPRate = float((TPRateA + TPRateB + TPRateC)/3.0)

#Compute False Positive Rate
FPRateA = float((cTable[0][1] + cTable[0][2])/(numTestExamples[1] + numTestExamples[2]))
FPRateB = float((cTable[1][0] + cTable[1][2])/(numTestExamples[0] + numTestExamples[2]))
FPRateC = float((cTable[2][0] + cTable[2][1])/(numTestExamples[0] + numTestExamples[1]))
FPRate = float((FPRateA + FPRateB + FPRateC)/3.0)

#Compute Error Rate
eRateA = float((cTable[0][1] + cTable[0][2] + cTable[1][0] + cTable[2][0])/
               (numTestExamples[0] + numTestExamples[1] + numTestExamples[2]))
eRateB = float((cTable[1][0] + cTable[1][2] + cTable[0][1] + cTable[2][1])/
               (numTestExamples[0] + numTestExamples[1] + numTestExamples[2]))
eRateC = float((cTable[2][0] + cTable[2][1] + cTable[0][2] + cTable[1][2])/
               (numTestExamples[0] + numTestExamples[1] + numTestExamples[2]))
eRate = float((eRateA + eRateB + eRateC)/3.0)

#Compute Accuracy
accuracy = float(1 - eRate)

#Compute Precision
precisionA = float((cTable[0][0])/(cTable[0][0] + cTable[0][1] + cTable[0][2]))
precisionB = float((cTable[1][1])/(cTable[1][1] + cTable[1][0] + cTable[1][2]))
precisionC = float((cTable[2][2])/(cTable[2][2] + cTable[2][0] + cTable[2][1]))
precision= float((precisionA + precisionB + precisionC)/3.0)

print("True positive rate = %.2f" %(TPRate))
print("False positive rate = %.2f" %(FPRate))
print("Error rate = %.2f" %(eRate))
print("Accuracy = %.2f" %(accuracy))
print("Precision = %.2f" %(precision))


