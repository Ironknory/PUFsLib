import torch
import torch.nn.functional as F

def transform2D(challenge):
    phi = torch.ones(size=(challenge.shape[0], challenge.shape[1] + 1))
    phi = phi.to(challenge.device)
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1] - 2, -1, -1):
            phi[i][j] = phi[i][j + 1] * (1 - 2 * challenge[i][j])
    return phi

class LR:
    def __init__(self, trainLoader, validLoader, testLoader, lr=0.1, epochs=20, momentum=0):
        self.trainLoader, self.validLoader, self.testLoader = trainLoader, validLoader, testLoader
        self.lr = lr
        self.epochs = epochs
        self.momentum = momentum

    def onAPUF(self, PUFSample):
        device = torch.device('cuda')
        
        weight = torch.tensor(PUFSample.weight).to(device).requires_grad_(True)
        optimizer = torch.optim.Adam([weight], lr=self.lr)
        for i in range(self.epochs):
            for (phi, R) in self.trainLoader:
                phi, R = phi.to(device), R.to(device)
                delta = torch.sum(weight * phi, dim=1, keepdim=True)
                response  = torch.sigmoid(-delta).to(torch.float32)
                loss = F.binary_cross_entropy(response, R)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            with torch.no_grad():
                accCount = 0
                for (phi, R) in self.validLoader:
                    phi, R = phi.to(device), R.to(device)
                    delta = torch.sum(weight * phi, dim=1, keepdim=True)
                    response  = torch.round(torch.sigmoid(-delta))
                    accCount += torch.sum(response == R).item()
                print("Epoch =", i, "Valid Accuracy =", accCount / len(self.validLoader.dataset.indices))

        accCount = 0
        for (phi, R) in self.testLoader:
            phi, R = phi.to(device), R.to(device)
            delta = torch.sum(weight * phi, dim=1, keepdim=True)
            response = torch.zeros_like(delta)
            for i in range(delta.shape[0]):
                if delta[i] < 0:
                    response[i] = 1
            accCount += torch.sum(response == R).item()
        ansWeight = weight.clone().detach()
        return ansWeight, accCount / len(self.testLoader.dataset.indices)
    