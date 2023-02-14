import torch
from math import sqrt
import torch.nn.functional as F

from .LR import LR


class ECP_TRN(LR):
    def __init__(self, trainLoader, validLoader, testLoader, lr=0.001, epochs=100, momentum=0.9):
        super(ECP_TRN, self).__init__(trainLoader, validLoader, testLoader, lr, epochs, momentum)
        
    def onXORAPUF(self, PUFSample, rank=10):
        device = torch.device('cuda')
        weight = []
        for _ in range(rank):
            eachWeight = torch.normal(0, sqrt(0.05), size=PUFSample.weight.shape)
            eachWeight = eachWeight.to(device).requires_grad_(True)
            weight.append(eachWeight)
        optimizer = torch.optim.Adam(weight, lr=self.lr)
        
        for epochs in range(self.epochs):
            for (phi, R) in self.trainLoader:
                phi, R = phi.to(device), R.to(device)
                delta = torch.zeros_like(R).to(device).requires_grad_(True)
                for i in range(rank):
                    eachDelta = torch.sum((weight[i][0] * phi), dim=-1, keepdim=True)
                    for j in range(1, PUFSample.number):
                        eachDelta = eachDelta * torch.sum((weight[i][j] * phi), dim=-1, keepdim=True)
                    delta = delta + eachDelta
                response = torch.sigmoid(-delta)
                loss = F.binary_cross_entropy(response, R)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            with torch.no_grad():
                accCount = 0
                for (phi, R) in self.validLoader:
                    phi, R = phi.to(device), R.to(device)
                    delta = torch.zeros_like(R).to(device).requires_grad_(False)
                    for i in range(rank):
                        eachDelta = torch.sum((weight[i][0] * phi), dim=-1, keepdim=True)
                        for j in range(1, PUFSample.number):
                            eachDelta = eachDelta * torch.sum((weight[i][j] * phi), dim=-1, keepdim=True)
                        delta = delta + eachDelta
                    response = torch.round(torch.sigmoid(-delta))
                    accCount += torch.sum(response == R).item()
                print("Epoch =", epochs, "Valid Accuracy =", accCount / len(self.validLoader.dataset.indices))
        
        accCount = 0
        for (phi, R) in self.testLoader:
            phi, R = phi.to(device), R.to(device)
            delta = torch.zeros_like(R).to(device).requires_grad_(False)
            for i in range(rank):
                eachDelta = torch.sum((weight[i][0] * phi), dim=-1, keepdim=True)
                for j in range(1, PUFSample.number):
                    eachDelta = eachDelta * torch.sum((weight[i][j] * phi), dim=-1, keepdim=True)
                delta = delta + eachDelta
            response = torch.round(torch.sigmoid(-delta))
            accCount += torch.sum(response == R).item()
        ansWeight = []
        for i in range(rank):
            ansWeight.append(weight[i].clone().detach())
        return ansWeight, accCount / len(self.testLoader.dataset.indices)
    