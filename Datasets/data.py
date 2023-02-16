import torch
import numpy as np
import random
# from torch.utils.data import DataLoader, random_split

def simpleSplit(data, nums):
    totCount = len(data)
    plc = 0
    splitResult = []
    for i in range(len(nums)):
        splitResult.append(data[plc:plc+nums[i]])
        plc += nums[i]
    return splitResult

class simpleDataLoader:
    def __init__(self, Xlable, Ylable, batch_size=1024, shuffle=True, drop_last=True, to_cuda=True):
        assert Xlable.shape[0] == Ylable.shape[0], "Data Shape Error"
        self.Xlable = torch.from_numpy(Xlable).to('cuda', torch.float32)
        self.Ylable = torch.from_numpy(Ylable).to('cuda', torch.float32)
        self.size = self.Xlable.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index = [(x, x + batch_size) for x in range(0, self.size, self.batch_size)]
        lastIndex = self.index[-1][1]
        if drop_last is False and lastIndex < self.size:
            lastIndex = self.index[-1][1]
            self.index.append((lastIndex, self.size))
    
    def __iter__(self):
        self.pos = -1
        if self.shuffle is True:
            random.shuffle(self.index)
        return self

    def __next__(self):
        self.pos += 1
        if self.pos >= len(self.index):
            raise StopIteration
        (l, r) = self.index[self.pos]
        return self.Xlable[l:r], self.Ylable[l:r]


def makeData(filename, dataSize, PUFSample):
    dataSet = []
    length = PUFSample.length
    for _ in range(dataSize):
        C = np.asarray([random.randint(0, 1) for _ in range(length)])
        phi = PUFSample.transform(C)
        R = PUFSample.getResponse(phi=phi)
        dataline = np.hstack((phi, R)).tolist()
        dataSet.append(dataline)
    dataSet = np.asarray(dataSet)
    np.savetxt(filename, dataSet, fmt='%d', delimiter=',')
    
def loadData(filename, batch_size=32, num_workers=8):
    dataSet = np.loadtxt(filename, delimiter=',')
    np.random.shuffle(dataSet)

    dataSize = dataSet.shape[0]
    trainSize = int(dataSize * 0.9)
    validSize = int(dataSize * 0.01)
    testSize = dataSize - trainSize - validSize

    Xlable, Ylable = dataSet[:,:-1], dataSet[:,-1:]
    [trainSetX, validSetX, testSetX] = simpleSplit(Xlable, [trainSize, validSize, testSize])
    [trainSetY, validSetY, testSetY] = simpleSplit(Ylable, [trainSize, validSize, testSize])
    '''
    trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    validLoader = DataLoader(validSet, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    testLoader = DataLoader(testSet, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    '''
    trainLoader = simpleDataLoader(trainSetX, trainSetY, batch_size=batch_size, shuffle=True)
    validLoader = simpleDataLoader(validSetX, validSetY, batch_size=batch_size, shuffle=True)
    testLoader = simpleDataLoader(testSetX, testSetY, batch_size=batch_size, shuffle=True)
    return trainLoader, validLoader, testLoader
