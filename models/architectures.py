import torch.nn as nn
import torch.nn.functional as F
from .shift_bn import ShiftBN

class StudentBlock(nn.Module):
    def __init__(self, ic, oc, s=1):
        super().__init__()
        self.c1 = nn.Conv2d(ic, oc, 3, s, 1, bias=False)
        self.b1 = ShiftBN(oc)
        self.c2 = nn.Conv2d(oc, oc, 3, 1, 1, bias=False)
        self.b2 = ShiftBN(oc)
        self.sc = nn.Sequential() if (s==1 and ic==oc) else nn.Sequential(nn.Conv2d(ic, oc, 1, s, bias=False), ShiftBN(oc))
        
    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(out)) + self.sc(x))

class IndraBitResNet(nn.Module):
    """
    Parametrized APoT ResNet. 
    base_width=16 -> 270K params
    base_width=28 -> 830K params
    base_width=40 -> 1.7M params
    """
    def __init__(self, base_width=16):
        super().__init__()
        w1, w2, w3 = base_width, base_width*2, base_width*4
        
        self.prep = nn.Conv2d(3, w1, 3, 1, 1, bias=False)
        self.bn0 = nn.BatchNorm2d(w1)
        self.l1 = nn.Sequential(StudentBlock(w1, w1), StudentBlock(w1, w1), StudentBlock(w1, w1))
        self.l2 = nn.Sequential(StudentBlock(w1, w2, 2), StudentBlock(w2, w2), StudentBlock(w2, w2))
        self.l3 = nn.Sequential(StudentBlock(w2, w3, 2), StudentBlock(w3, w3), StudentBlock(w3, w3))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(w3, 10)

    def forward(self, x):
        x = F.relu(self.bn0(self.prep(x)))
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return self.fc(self.pool(x).view(x.size(0), -1))
