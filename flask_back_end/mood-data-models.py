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
        out = X.view(X.shape[0],-1)
        out = self.linear(out)
        return out

class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size = 60):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.soft = nn.Softmax(dim=0)

    def forward(self, X):
        out = self.tanh(self.linear1(X))
        out = self.linear2(out)
        return out

class SimpleCNN(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv1d(num_channels, 8, 5)
        self.conv2 = nn.Conv1d(8, 8, 3)
        self.max_pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(968, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, X):
        out = self.relu(self.conv1(X))
        out = self.relu(self.conv2(out))
        out = self.dropout(out)
        out = self.max_pool(out)
        out = out.view(out.shape[0],-1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out
        

#1d cnn based on paper here: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6453988/
class WaveletCNN(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(WaveletCNN, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 32, 3)
        self.norm1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 34, 4)
        self.norm2 = nn.BatchNorm1d(34)
        self.max_pool = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(34, 64, 3)
        self.norm3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 64, 4)
        self.norm4 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(3520, 550)
        self.norm5 = nn.BatchNorm1d(550)
        self.dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(550, 250)
        self.norm6 = nn.BatchNorm1d(250)
        self.fc3 = nn.Linear(250, 100)
        self.norm7 = nn.BatchNorm1d(100)
        self.fc4 = nn.Linear(100, 25)
        self.norm8 = nn.BatchNorm1d(25)
        self.fc5 = nn.Linear(25, num_classes)
    
    def forward(self, X):
        out = self.norm1(self.conv1(X))
        out = self.norm2(self.conv2(out))
        out = self.max_pool(out)
        out = self.norm3(self.conv3(out))
        out = self.norm4(self.conv4(out))
        out = self.max_pool(out)
        out = out.view(out.shape[0],-1)
        out = self.norm5(self.fc1(out))
        out = self.dropout(out)
        out = self.norm6(self.fc2(out))
        out = self.dropout(out)
        out = self.norm7(self.fc3(out))
        out = self.dropout(out)
        out = self.norm8(self.fc4(out))
        out = self.dropout(out)
        out = self.fc5(out)
        return out

