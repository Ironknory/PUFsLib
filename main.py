import torch
from PUFs import *
from Attacks import *
from random import randint
from torch.utils.data import DataLoader, random_split

def testLRonAPUF():
    PUFLength = 32
    batch_size = 32

    dataSize = int(1e4)
    trainSize = int(0.8 * dataSize)
    testSize = dataSize - trainSize
    dataSet = []

    PUFSample = APUF.randomSample(PUFLength)
    for _ in range(dataSize):
        C = torch.tensor([randint(0, 1) for _ in range(PUFLength)])
        R = PUFSample.getResponse(C)
        dataSet.append((C, R))

    trainSet, testSet = random_split(dataSet, [trainSize, testSize])
    trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True)
    testLoader = DataLoader(testSet, batch_size=batch_size, shuffle=True)

    randomSample = APUF.randomSample(PUFLength)
    attackMethod = LR(trainLoader, testLoader)
    ansModel, accuracy = attackMethod.onAPUF(randomSample)
    print("LR on APUF")
    print("length =", PUFLength, "accuracy =", accuracy)

def testMLPonXORAPUF():
    PUFNumber = 3
    PUFLength = 32
    batch_size = 2

    dataSize = int(1e5)
    trainSize = int(0.8 * dataSize)
    validSize = int(0.1 * dataSize)
    testSize = dataSize - trainSize - validSize
    dataSet = []

    PUFSample = XORAPUF.randomSample(PUFNumber, PUFLength)
    for _ in range(dataSize):
        C = torch.tensor([randint(0, 1) for _ in range(PUFLength)])
        R = PUFSample.getResponse(C)
        dataSet.append((C, R))

    trainSet, validSet, testSet = random_split(dataSet, [trainSize, validSize, testSize])
    trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True)
    validLoader = DataLoader(validSet, batch_size=batch_size, shuffle=True)
    testLoader = DataLoader(testSet, batch_size=batch_size, shuffle=True)

    randomSample = XORAPUF.randomSample(PUFNumber, PUFLength)
    attackMethod = MLP(trainLoader, validLoader, testLoader)
    ansModel, accuracy = attackMethod.onXORAPUF(randomSample)
    print("MLP on XORAPUF")
    print("number =", PUFNumber, "length =", PUFLength, "accuracy =", accuracy)


if __name__ == "__main__":
    # testLRonAPUF()
    testMLPonXORAPUF()
