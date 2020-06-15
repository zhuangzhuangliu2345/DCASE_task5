import torch
import torch.nn.functional as F

# 
def binary_cross_entropy(output, target):
    return F.binary_cross_entropy(output, target)
def binary_cross_entropy1(output, target):
    l=[]
    for i in range(len(output)):
        l.append(F.binary_cross_entropy(output[i], target[i]))
    return l