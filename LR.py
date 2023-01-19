import torch

class LR:
    def __init__(self, CLoader, RLoader, batch=128, lr=0.1):
        self.DataC, self.DataR = CLoader, RLoader
        self.lr = lr
        self.batch = batch

    def onAPUF(self, PUFSample):
        weight = torch.tensor(PUFSample.weight, requires_grad=True)

