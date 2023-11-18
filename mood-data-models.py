import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

class LinearModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.soft = nn.Softmax(dim=0)

    def forward(self, X):
        out = self.linear(X)
        return out

class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size = 60):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=0)

    def forward(self, X):
        out = self.relu(self.linear1(X))
        out = self.linear2(out)
        return out
