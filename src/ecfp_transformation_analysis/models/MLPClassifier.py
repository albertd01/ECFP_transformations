import torch.nn as nn
import torch.nn.functional as F
import torch

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=128, output_dim=1):
        #torch.manual_seed(1312)
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    