import torch
import torch.nn as nn
import torch.nn.functional as F

class TravelTimePredictor(nn.Module):
    def __init__(self):
        super(TravelTimePredictor, self).__init__()
        self.fc1 = nn.Linear(in_features=10, out_features=64) # Adjust in_features based on your final input size
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Linear output for regression
        return x

# Instantiate the model
model = TravelTimePredictor()

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
