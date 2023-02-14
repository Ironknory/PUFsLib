import torch
import torch.nn as nn
import torch.nn.functional as F
from .LR import LR, transform2D

class MLPModel(nn.Module):
    def __init__(self, sizes, activateFunc='tanh'):
        super(MLPModel, self).__init__()
        self.activateFunc = nn.Tanh
        
        self.layers = nn.Sequential()
        self.layers.add_module("Flatten", nn.Flatten())
        # input -> 64/32 -> 2^n/2 -> 2^n -> 2^n/2 -> 1
        for i in range(len(sizes) - 2):
            layerName = "Liner" + str(i + 1)
            self.layers.add_module(layerName, nn.Linear(sizes[i], sizes[i + 1]))
            layerName = "Activate" + str(i + 1)
            self.layers.add_module(layerName, self.activateFunc())
        
        layerName = "Liner" + str(len(sizes))
        self.layers.add_module(layerName, nn.Linear(sizes[-2], sizes[-1]))
        layerName = "Activate" + str(len(sizes))
        self.layers.add_module(layerName, nn.Sigmoid())

    def forward(self, x):
        return self.layers(x)


class MLP(LR):
    def __init__(self, trainLoader, validLoader, testLoader, lr=0.001, epochs=100, momentum=0.9):
        super(MLP, self).__init__(trainLoader, validLoader, testLoader, lr, epochs, momentum)

    def onXORAPUF(self, PUFSample):
        number = PUFSample.number
        sizes = [PUFSample.length + 1, 2 ** (number - 1), 2 ** number, 2 ** (number - 1), 1]
        model = MLPModel(sizes, activateFunc='tanh')
        print(model)
        
        device = torch.device('cuda')
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        for i in range(self.epochs):
            model.train()
            for (phi, R) in self.trainLoader:
                phi, R = phi.to(device), R.to(device)
                response = model(phi)
                loss = F.binary_cross_entropy(response, R)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            accCount = 0
            for (phi, R) in self.validLoader:
                phi, R = phi.to(device), R.to(device)
                response = torch.round(model(phi))
                accCount += torch.sum(response == R).item()
            print("Epoch =", i, "Valid Accuracy =", accCount / len(self.validLoader.dataset.indices))
        
        model.eval()
        accCount = 0
        for (phi, R) in self.testLoader:
            phi, R = phi.to(device), R.to(device)
            response = torch.round(model(phi))
            accCount += torch.sum(response == R).item()
        return model, accCount / len(self.testLoader.dataset.indices)
    