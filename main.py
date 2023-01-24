import torch
from PUFs import *
from Attacks import *
from random import randint
from torch.utils.data import DataLoader, random_split

if __name__ == "__main__":
    PUFLength = 32

    batch_size = 32

    dataSize = int(1e5)
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
    print(accuracy)
    