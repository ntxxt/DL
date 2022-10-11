from asyncio import trsock
from time import sleep
from wsgiref import headers
import torch
from torch import nn
import torch.nn.functional as F

#basic self-attention
x = torch.rand(2,3,4)  #random tensor with shape (b,t,k) #batch 
#print(x.size()) #torch.Size([2, 3, 4])
#print(x.transpose(1,2).size()) #torch.Size([2, 4, 3])
raw_weights = torch.bmm(x, x.transpose(1,2))
#print(raw_weights.size()) #torch.Size([2, 3, 3])
weights = F.softmax(raw_weights, dim=2)
#print(weights.size()) #torch.Size([2, 3, 3])
y = torch.bmm(weights, x)
#print(y.size()) #torch.Size([2, 3, 4]) # (batch, x, a) bmm (batch, a, y) -> (batch, x, y)

'''
x = torch.rand(3,4,1)
y = torch.rand(3,1,5)
print(torch.bmm(x, y).size())
'''

class SelfAttention(nn.Module):
    def __init__(self, k, heads=8):
        super.__init__()
        self.k, self.heads = k, heads
        
        self.tokeys = nn.Linear(k, k * heads, bias=False)
        self.toqueries = nn.Linear(k, k * heads, bias=False)
        self.tovalues = nn.Linear(k, k * heads, bias=False)
        self.unifyheads = nn.Linear(heads * k , k)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads

        queries = self.toqueries(x).view(b,t ,h, k)
        keys = self.tokeys(x).view(b, t, h, k)
        values = self.tovalues(x).view(b,t,h,k)

        keys = keys.transpose(1,2).contiguous().view(b*h, t, k) / (k ** (1/4))
        queries = queries.transpose(1,2).contiguous().view(b*h, t, k) / (k ** (1/4))
        values = values.transpose(1,2).contiguous().view(b*h, t, k)

        dot = torch.bmm(queries, keys.transpose(1,2))
        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).view(b,h,t,k)
        out = out.transpose(1,2).contiguous().view(b,t,h*k)
        return self.unifyheads(out)


a = SelfAttention(k=8)
x = torch.rand(8,8,8)
a(k=x, heads=8)