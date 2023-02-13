import torch
import numpy as np
from PUFs import *
from Attacks import *
from Datasets.data import makeData, loadData
from random import randint
from torch.utils.data import DataLoader, random_split

def testLRonAPUF():
    PUFLength = 32
    PUFSample = APUF.randomSample(PUFLength)
    filename = "./Datasets/32_APUF_1k.csv"
    # makeData(filename, int(1e3), PUFSample)
    
    trainLoader, validLoader, testLoader = loadData(filename)
    randomSample = APUF.randomSample(PUFLength)
    attackMethod = LR(trainLoader, validLoader, testLoader)
    ansWeight, accuracy = attackMethod.onAPUF(randomSample)
    print("LR on APUF")
    print("length =", PUFLength, "accuracy =", accuracy)

def testMLPonXORAPUF():
    PUFNumber = 3
    PUFLength = 32
    batch_size = 128
    
    PUFSample = XORAPUF.randomSample(PUFNumber, PUFLength)
    filename = "./Datasets/3_32_XORAPUF_20k.csv"
    # makeData(filename, int(2e4), PUFSample)
    
    trainLoader, validLoader, testLoader = loadData(filename, batch_size=batch_size)
    randomSample = XORAPUF.randomSample(PUFNumber, PUFLength)
    attackMethod = MLP(trainLoader, validLoader, testLoader)
    ansModel, accuracy = attackMethod.onXORAPUF(randomSample)
    print("MLP on XORAPUF")
    print("number =", PUFNumber, "length =", PUFLength, "accuracy =", accuracy)


if __name__ == "__main__":
    testLRonAPUF()
    testMLPonXORAPUF()
