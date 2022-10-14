from re import L
import torch
from random import randint
from math import sqrt

class APUF:
    def transform(challenge):
        challenge = torch.cat((challenge, torch.tensor([1])), 0)
        for i in range(challenge.shape[0] - 2, -1, -1):
            challenge[i] = challenge[i + 1] * (1 - 2 * challenge[i])
        return challenge


    def __init__(self, challenge, weight, transformed=False):
        self.challenge = challenge
        self.weight = weight
        if not transformed:
            self.challenge = APUF.transform(self.challenge)
        assert self.challenge.shape[-1] == self.weight.shape[-1], "Shape Error"
            
    def forward(self):
        ans = torch.sum(self.challenge * self.weight)
        return ans
    
    def getResponse(self):
        ans = self.forward()
        response = torch.ones(size=ans.shape)
        if ans >= 0:
            response = torch.zeros(size=ans.shape)
        return response
    
    def randomSample(length):
        challenge = torch.tensor([randint(0, 1) for _ in range(length)])
        weight = torch.normal(0, sqrt(0.05), size=(length + 1, ))
        return APUF(challenge, weight, transformed=False)

class XORAPUF(APUF):
    def __init__(self, challenge, weight, transformed=True):
        self.challenge = challenge
        self.weight = weight
        if not transformed:
            for i in range(self.challenge.shape[0]): # expend line by line will make the tensor not rectangular
                self.challenge[i] = XORAPUF.transform(self.challenge[i])
        assert self.challenge.shape[-1] == self.weight.shape[-1], "Shape Error"
        
    def forward(self):
        ans = torch.sum(self.challenge * self.weight, dim=1)
        return ans
    
    def getResponse(self):
        ans = self.forward()
        response = torch.ones(size=ans.shape)
        for i in range(response.shape[0]):
            if ans[i] >= 0:
                response[i] = 0
        return response
    
    def randomSample(n, length):
        challenge = torch.tensor([[randint(0, 1) for _ in range(length)] for _ in range(n)])
        weight = torch.normal(0, sqrt(0.05), size=(n, length + 1))
        return XORAPUF(challenge, weight, transformed=False)
            
    
        
    


APUFSample = APUF.randomSample(128)
print(APUFSample.forward(), APUFSample.getResponse())

XORAPUFSample = XORAPUF.randomSample(2, 128)
print(XORAPUFSample.forward(), XORAPUFSample.getResponse())

