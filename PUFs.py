import torch
from random import randint
from math import sqrt

class APUF:
    # Arbiter PUF
    # attribute:
    #   challenge: the challenge bits, with the length of 64 or 128
    #   phi: phi, with the length of 64 or 128
    #   weight: the delay of each stage, wiith the length of 64 + 1 or 128 + 1
    # method:
    #   initChallenge: init or change challenge and phi
    #   forward: calculate the total time delay of the PUF
    #   getResponse: get the response by forward mothod
    #   randomSample: inital a APUF class with random challenge(ramdon 0/1) and weight(normal distribution)

    def initChallenge(self, challenge):
        self.challenge = challenge
        self.phi = torch.cat((challenge, torch.ones(size=(1,))), 0)
        for i in range(self.phi.shape[0] - 2, -1, -1):
            self.phi[i] = self.phi[i + 1] * (1 - 2 * self.phi[i])

    def __init__(self, challenge, weight):
        self.initChallenge(challenge)
        self.weight = weight
        assert self.phi.shape[-1] == self.weight.shape[-1], "Shape Error"
    
    def forward(self):
        ans = torch.sum(self.phi * self.weight, dim=0, keepdim=True)
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
        return APUF(challenge, weight)

class XORAPUF(APUF):
    def forward(self):
        ans = torch.sum(self.phi * self.weight, dim=1, keepdim=True)
        return ans

    def randomSample(n, length):
        challenge = torch.tensor([randint(0, 1) for _ in range(length)])
        weight = torch.normal(0, sqrt(0.05), size=(n, length + 1))
        return XORAPUF(challenge, weight)

class iPUF(XORAPUF):
    # Interpose PUF
    # attribute:
    #   weightX, weightY: the weight of x-xor APUF and y-xor APUF
    #   interplace: the place that the response of x-xor APUF would interpose
    # method:
    #   flipChallenge: calculate the challenge after being interposed
    #       intsert the bit of challenge[interplace], flip challenge[0~interplace] if interbit is 1
    
    def flipPhi(interbit, interplace, phi):
        tmplist = phi.tolist()
        tmplist.insert(interplace, tmplist[interplace])
        phi = torch.tensor(tmplist)
        if interbit != 0:
            for i in range(interplace + 1):
                phi[i] = -phi[i]
        return phi

    def __init__(self, challenge, weightX, weightY, interplace):
        self.initChallenge(challenge)
        self.weightX, self.weightY = weightX, weightY
        self.interplace = interplace
        assert self.phi.shape[-1] == self.weightX.shape[-1], "x-XOR APUF Shape Error"
        assert self.phi.shape[-1] == self.weightY.shape[-1] - 1, "y-XOR APUF Shape Error"
        assert self.interplace >= 0 and self.interplace < self.challenge.shape[-1], "Interpose place Error"
     
    def forwardX(self):
        ans = torch.sum(self.phi * self.weightX, dim=1, keepdim=True)
        return ans
    
    def forwardY(self, interbit):
        phiY = iPUF.flipPhi(interbit, self.interplace, self.phi)
        ans = torch.sum(phiY * self.weightY, dim=1, keepdim=True)
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
        return iPUF(challenge, weightX, weightY, interplace=length//2)
        
class MPUF(APUF):
    def __init__(self, challenge, weightR, weightD):
        self.initChallenge(challenge)
        self.weightR, self.weightD = weightR, weightD
        assert self.phi.shape[-1] == self.weightD.shape[-1], "Data APUF Shape Error"
        assert self.phi.shape[-1] == self.weightR.shape[-1], "Select APUF Shape Error"
        assert self.weightD.shape[0] == 2 ** self.weightR.shape[0], "D&R not Match"
    
    def forwardR(self):
        ansR = torch.sum(self.phi * self.weightR, dim=1, keepdim=True)
        return ansR
    
    def forwardD(self):
        ansD = torch.sum(self.phi * self.weightD, dim=1, keepdim=True)
        return ansD
    
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
        return MPUF(challenge, weightR, weightD)

length = 128
challenge = torch.tensor([randint(0, 1) for _ in range(128)])

APUFSample = APUF.randomSample(length)
print("Ans =", APUFSample.forward())
print("Response =", APUFSample.getResponse())
APUFSample.initChallenge(challenge)
print("Ans =", APUFSample.forward())
print("Response =", APUFSample.getResponse())

XORAPUFSample = XORAPUF.randomSample(2, length)
print("Ans =", XORAPUFSample.forward())
print("Response =", XORAPUFSample.getResponse())
XORAPUFSample.initChallenge(challenge)
print("Ans =", XORAPUFSample.forward())
print("Response =", XORAPUFSample.getResponse())

iPUFSample = iPUF.randomSample(2, 2, length)
print("Ans =", iPUFSample.forward())
print("Response =", iPUFSample.getResponse())
iPUFSample.initChallenge(challenge)
print("Ans =", iPUFSample.forward())
print("Response =", iPUFSample.getResponse())


MPUFSample = MPUF.randomSample(2, length)
print("AnsR =", MPUFSample.forwardR())
print("AnsD =", MPUFSample.forwardD())
print("Select =", MPUFSample.getSelect())
print("Response =", MPUFSample.getResponse())
MPUFSample.initChallenge(challenge)
print("AnsR =", MPUFSample.forwardR())
print("AnsD =", MPUFSample.forwardD())
print("Select =", MPUFSample.getSelect())
print("Response =", MPUFSample.getResponse())

