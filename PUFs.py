import torch
from random import randint
from math import sqrt

class APUF:
    # Arbiter PUF
    # attribute:
    #   challenge: the challenge bits, with the length of 64 or 128
    #   weight: the delay of each stage, wiith the length of 64 + 1 or 128 + 1
    # method:
    #   transform: transform challenge to PHI
    #   forward: calculate the total time delay of the PUF
    #   getResponse: get the response by forward mothod
    #   randomSample: inital a APUF class with random challenge(ramdon 0/1) and weight(normal distribution)
    
    def transform(challenge):
        challenge = torch.cat((challenge, torch.ones(size=(1,))), 0)
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
        ans = torch.sum(self.challenge * self.weight, dim=0, keepdim=True)
        return ans
    
    def getResponse(self):
        ans = self.forward()
        response = torch.tensor([0])
        for i in range(ans.shape[0]):
            if ans[i] >= 0:
                response ^= 0
            else:
                response ^= 1
        return response
    
    def randomSample(length):
        challenge = torch.tensor([randint(0, 1) for _ in range(length)])
        weight = torch.normal(0, sqrt(0.05), size=(length + 1, ))
        return APUF(challenge, weight, transformed=False)

class XORAPUF(APUF):
    def forward(self):
        ans = torch.sum(self.challenge * self.weight, dim=1, keepdim=True)
        return ans

    def randomSample(n, length):
        challenge = torch.tensor([randint(0, 1) for _ in range(length)])
        weight = torch.normal(0, sqrt(0.05), size=(n, length + 1))
        return XORAPUF(challenge, weight, transformed=False)

class iPUF(XORAPUF):
    # Interpose PUF
    # attribute:
    #   weightX, weightY: the weight of x-xor APUF and y-xor APUF
    #   interplace: the place that the response of x-xor APUF would interpose
    # method:
    #   flipChallenge: calculate the challenge after being interposed
    #       intsert the bit of challenge[interplace], flip challenge[0~interplace] if interbit is 1
    
    def flipChallenge(interbit, interplace, challenge):
        tmplist = challenge.tolist()
        tmplist.insert(interplace, tmplist[interplace])
        challenge = torch.tensor(tmplist)
        if interbit != 0:
            for i in range(interplace + 1):
                challenge[i] = -challenge[i]
        return challenge

    def __init__(self, challenge, weightX, weightY, interplace, transformed=False):
        self.challenge = challenge
        self.weightX, self.weightY = weightX, weightY
        self.interplace = interplace
        if not transformed:
            self.challenge = iPUF.transform(self.challenge)
        assert self.challenge.shape[-1] == self.weightX.shape[-1], "x-XOR APUF Shape Error"
        assert self.challenge.shape[-1] == self.weightY.shape[-1] - 1, "y-XOR APUF Shape Error"
        assert self.interplace >= 0 and self.interplace < self.challenge.shape[-1], "Interpose place Error"
     
    def forwardX(self):
        ans = torch.sum(self.challenge * self.weightX, dim=1, keepdim=True)
        return ans
    
    def forwardY(self, interbit):
        challenge = iPUF.flipChallenge(interbit, self.interplace, self.challenge)
        ans = torch.sum(challenge * self.weightY, dim=1, keepdim=True)
        return ans
    
    def forward(self):
        ansX = self.forwardX()
        interbit = 0
        for i in range(ansX.shape[0]):
            if ansX[i] >= 0:
                interbit ^= 0
            else:
                interbit ^= 1
        return self.forwardY(interbit)
        
    def randomSample(x, y, length):
        challenge = torch.tensor([randint(0, 1) for _ in range(length)])
        weightX = torch.normal(0, sqrt(0.05), size=(x, length + 1))
        weightY = torch.normal(0, sqrt(0.05), size=(y, length + 2))
        return iPUF(challenge, weightX, weightY, interplace=length//2, transformed=False)
        
class MPUF(APUF):
    def __init__(self, challenge, weightR, weightD, transformed=False):
        self.challenge = challenge
        self.weightR, self.weightD = weightR, weightD
        if not transformed:
            self.challenge = MPUF.transform(self.challenge)
        assert self.challenge.shape[-1] == self.weightD.shape[-1], "Data APUF Shape Error"
        assert self.challenge.shape[-1] == self.weightR.shape[-1], "Select APUF Shape Error"
        assert self.weightD.shape[0] == 2 ** self.weightR.shape[0], "D&R not Match"
    
    def forwardR(self):
        ans = torch.sum(self.challenge * self.weightR, dim=1, keepdim=True)
        return ans
    
    def forwardD(self):
        ans = torch.sum(self.challenge * self.weightD, dim=1, keepdim=True)
        return ans
    
    def getSelect(self):
        ans = self.forwardR()
        select = 0
        for i in range(ans.shape[0] - 1, -1, -1):
            if ans[i] >= 0:
                select = (select << 1) | 0
            else:
                select = (select << 1) | 1
        return select
    
    def getResponse(self):
        select = self.getSelect()
        ans = self.forwardD()
        response = torch.tensor([1])
        if ans[select] >= 0:
            response[0] = 0
        return response
    
    def randomSample(s, length):
        challenge = torch.tensor([randint(0, 1) for _ in range(length)])
        weightR = torch.normal(0, sqrt(0.05), size=(s, length + 1))
        weightD = torch.normal(0, sqrt(0.05), size=(2 ** s, length + 1))
        return MPUF(challenge, weightR, weightD, transformed=False)
    
APUFSample = APUF.randomSample(128)
print("Ans =", APUFSample.forward())
print("Response =", APUFSample.getResponse())

XORAPUFSample = XORAPUF.randomSample(2, 128)
print("Ans =", XORAPUFSample.forward())
print("Response =", XORAPUFSample.getResponse())

iPUFSample = iPUF.randomSample(2, 2, 128)
print("Ans =", iPUFSample.forward())
print("Response =", iPUFSample.getResponse())

MPUFSample = MPUF.randomSample(2, 128)
print("AnsR =", MPUFSample.forwardR())
print("AnsD =", MPUFSample.forwardD())
print("Select =", MPUFSample.getSelect())
print("Response =", MPUFSample.getResponse())