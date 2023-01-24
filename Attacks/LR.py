import torch
import torch.nn.functional as F

import sys
sys.path.append("..")
from PUFs import *

class LR:
    def __init__(self, trainLoader, testLoader, lr=1, momentum=0):
        self.trainLoader, self.testLoader = trainLoader, testLoader
        self.lr = lr
        self.momentum=momentum

    def transform(challenge):
        phi = torch.ones(size=(challenge.shape[0], challenge.shape[1] + 1))
        for i in range(phi.shape[0]):
            for j in range(phi.shape[1] - 2, -1, -1):
               phi[i][j] = phi[i][j + 1] * (1 - 2 * challenge[i][j])
        return phi

    def onAPUF(self, PUFSample):
        weight = PUFSample.weight.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([weight], lr=self.lr, momentum=self.momentum)
        for (C, R) in self.trainLoader:
            phi = LR.transform(C)
            delta = torch.sum(weight * phi, dim=1, keepdim=True)
            response  = torch.sigmoid(-delta)
            R = R.to(torch.float32)
            loss = F.binary_cross_entropy(response, R)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        ansModel = APUF(weight.shape[-1] - 1, weight.clone().detach())
        accCount = 0
        for (C, R) in self.testLoader:
            phi = LR.transform(C)
            delta = torch.sum(weight * phi, dim=1, keepdim=True)
            response = torch.zeros_like(delta)
            for i in range(delta.shape[0]):
                if delta[i] < 0:
                    response[i] = 1
            accCount += torch.sum(response == R).item()
        return ansModel, accCount / len(self.testLoader.dataset.indices)

         
            



            


