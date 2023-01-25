import torch
from math import sqrt
from random import randint
from .APUF import APUF

class XORAPUF(APUF):
    # XOR APUF
    # Attributes:
    #   number: the number of XORs
    #   length: the length of PUF
    #   noise: noise level of PUF, normal distributed
    #   weight: the delay of each stage, wiith the length of 64 + 1 or 128 + 1
    # Methods:
    #   __init__: init a object when use PUF(args)
    #   transform: trans challenge to phi
    #   getDelta: calc the time delay of the PUF with/without noise
    #   getResponse: get the response of the PUF with/without noise
    #   ramdomSample: inital a PUF class with random challenge and weight(normal distribution)

    def __init__(self, number, length, weight, noise_level=0):
        super(XORAPUF, self).__init__(length, weight, noise_level)
        self.number = number
        assert self.weight.shape[0] == self.number, "Shape Error"
        assert self.weight.shape[-1] == self.length + 1, "Shape Error"
    
    def getDelta(self, phi, noisy=False):
        noise = torch.zeros(size=self.weight.shape)
        if noisy is True:
            noise = torch.normal(0, self.noise_level, size=self.weight.shape)
        delta = torch.sum(phi * (self.weight + noise), dim=1, keepdim=True)
        return delta

    def getResponse(self, challenge, noisy=False):
        phi = XORAPUF.transform(challenge)
        assert phi.shape[0] == self.length + 1, "Shape Error"
        delta = self.getDelta(phi, noisy)
        response = torch.tensor([0])
        for eachDelta in delta:
           if eachDelta < 0:
                response[0] ^= 1
        return response
            
    def randomSample(number=3, length=32, noise_level=0):
        weight = torch.normal(0, sqrt(0.05), size=(number, length + 1))
        return XORAPUF(number, length, weight, noise_level)

'''
# TEST
number = 3
length = 32
XORAPUFSample = XORAPUF.randomSample(number, length, noise_level=0.1)

print("Within different challenge")
for _ in range(5):
    challenge = torch.tensor([randint(0, 1) for _ in range(length)])
    print("Response =", XORAPUFSample.getResponse(challenge))

challenge = torch.tensor([randint(0, 1) for _ in range(length)])
print("Within noise, where standard response is", XORAPUFSample.getResponse(challenge))
for _ in range(10):
    print("Response =", XORAPUFSample.getResponse(challenge, noisy=True))
'''
