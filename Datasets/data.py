import torch
import numpy as np
from random import randint
from torch.utils.data import DataLoader, random_split

def makeData(filename, dataSize, PUFSample):
    dataSet = []
    length = PUFSample.length
    for _ in range(dataSize):
        C = np.asarray([randint(0, 1) for _ in range(length)])
        phi = PUFSample.transform(C)
        R = PUFSample.getResponse(phi=phi)
        dataline = np.hstack((phi, R)).tolist()
        dataSet.append(dataline)
    dataSet = np.asarray(dataSet)
    np.savetxt(filename, dataSet, fmt='%d', delimiter=',')
    
def loadData(filename, batch_size=32):
    dataSet = np.loadtxt(filename, delimiter=',')
    trainSet = []
    for dataline in dataSet:
        [phi, R] = np.split(dataline, [-1])
        phi = torch.from_numpy(phi).to(torch.float32)
        R = torch.from_numpy(R).to(torch.float32)
        trainSet.append((phi, R))
    dataSize = dataSet.shape[0]
    trainSize = int(dataSize * 0.8)
    validSize = int(dataSize * 0.1)
    testSize = dataSize - trainSize - validSize
    trainSet, validSet, testSet = random_split(trainSet, [trainSize, validSize, testSize])
    trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True)
    validLoader = DataLoader(validSet, batch_size=batch_size, shuffle=True)
    testLoader = DataLoader(testSet, batch_size=batch_size, shuffle=True)
    return trainLoader, validLoader, testLoader
