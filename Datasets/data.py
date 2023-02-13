import torch
import numpy
from torch.utils.data import DataLoader, random_split

def saveData(filename, datasize, PUFSample):
    dataset = []
    length = PUFSample.length
    with open(filename, "w") as file:
        for _ in range(datasize):
            C = torch.tensor([randint(0, 1) for _ in range(length)])
            phi = PUFSample.transform(C).item()
            response = PUFSample.getRespnse(C).item()
            
            dataset.append((phi, R))
