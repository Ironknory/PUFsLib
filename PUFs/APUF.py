import torch
from math import sqrt
from random import randint

class APUF:
    # Arbiter PUF
    # Attributes:
    #   length: the length of APUF
    #   noise: noise level of APUF, normal distributed
    #   weight: the delay of each stage, wiith the length of 64 + 1 or 128 + 1
    # Methods:
    #   __init__: init a object when use APUF(args)
    #   transform: trans challenge to phi
    #   getDelta: calc the time delay of the PUF with/without noise
    #   getResponse: get the response of the PUF with/without noise
    #   ramdomSample: inital a APUF class with random challenge and weight(normal distribution)

    def __init__(self, length, weight, noise_level=0):
        self.length = length
        self.weight = weight
        self.noise_level = noise_level
        assert self.weight.shape[0] == self.length + 1, "Shape Error"
    
    def transform(challenge):
        phi = torch.ones(size=(challenge.shape[0] + 1,))
        for i in range(phi.shape[0] - 2, -1, -1):
            phi[i] = phi[i + 1] * (1 - 2 * challenge[i])
        return phi

    def getDelta(self, phi, noisy=False):
        noise = torch.zeros(size=self.weight.shape)
        if noisy is True:
            noise = torch.normal(0, self.noise_level, size=self.weight.shape)
        delta = torch.sum(phi * (self.weight + noise), dim=0, keepdim=True)
        return delta

    def getResponse(self, challenge, noisy=False):
        phi = APUF.transform(challenge)
        assert phi.shape[0] == self.length + 1, "Shape Error"
        delta = self.getDelta(phi, noisy)
        response = torch.tensor([0])
        if delta < 0:
            response[0] = 1
        return response
            
    def randomSample(length=32, noise_level=0):
        weight = torch.normal(0, sqrt(0.05), size=(length + 1,))
        return APUF(length, weight, noise_level)

'''
# TEST
length = 32
APUFSample = APUF.randomSample(length, noise_level=0.1)

print("Within different challenge")
for _ in range(5):
    challenge = torch.tensor([randint(0, 1) for _ in range(length)])
    print("Response =", APUFSample.getResponse(challenge))

challenge = torch.tensor([randint(0, 1) for _ in range(length)])
print("Within noise, where standard response is", APUFSample.getResponse(challenge))
for _ in range(10):
    print("Response =", APUFSample.getResponse(challenge, noisy=True))
'''
