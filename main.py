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
    PUFNumber = 7
    PUFLength = 64
    batch_size = 4096
    
    PUFSample = XORAPUF.randomSample(PUFNumber, PUFLength)
    filename = "./Datasets/" + str(PUFNumber) + "_" + str(PUFLength) + "_XORAPUF_1m.csv"
    # makeData(filename, int(1e6), PUFSample)
    
    trainLoader, validLoader, testLoader = loadData(filename, batch_size=batch_size)
    randomSample = XORAPUF.randomSample(PUFNumber, PUFLength)
    attackMethod = MLP(trainLoader, validLoader, testLoader, epochs=500)
    ansModel, accuracy = attackMethod.onXORAPUF(randomSample)
    print("MLP on XORAPUF")
    print("number =", PUFNumber, "length =", PUFLength, "accuracy =", accuracy)

def testECP_TRNonXORAPUF():
    PUFNumber = 4
    PUFLength = 64
    batch_size = 4096
    
    PUFSample = XORAPUF.randomSample(PUFNumber, PUFLength)
    filename = "./Datasets/4_64_XORAPUF_80k.csv"
    # makeData(filename, int(8e4), PUFSample)
    
    trainLoader, validLoader, testLoader = loadData(filename, batch_size=batch_size)
    randomSample = XORAPUF.randomSample(PUFNumber, PUFLength)
    attackMethod = ECP_TRN(trainLoader, validLoader, testLoader, epochs=200)
    ansWeight, accuracy = attackMethod.onXORAPUF(randomSample)
    print("ECP_TRN on XORAPUF")
    print("number =", PUFNumber, "length =", PUFLength, "accuracy =", accuracy)

    
if __name__ == "__main__":
    # testLRonAPUF()
    testMLPonXORAPUF()
    # testECP_TRNonXORAPUF()
