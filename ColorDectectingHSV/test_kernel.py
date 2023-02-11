import torch
import torch.nn as nn
input = torch.ones(2,3)
n = nn.Conv1d(2,1,2,dilation=1)
out = n(input)
print(input)