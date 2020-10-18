import torch
from torch import nn
import torch.nn.functional as F 

class SKUModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        #input 3 X 200 X 200 iamges
        self.conv1 = nn.Conv2d(3, 20, kernel_size=(3,3), stride=1, padding=(1,1))
        # input 20 X 200 X 200
        # Apply maxpool
        # input 20 X 100 X 100
        self.conv2 = nn.Conv2d(20, 50, kernel_size=(3,3), stride=1, padding=(1,1))
        # input 50 X 100 X 100
        # Apply maxpool
        # input 50 X 50 X 50
        self.fc1 = nn.Linear(50*50*50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, xb):
        x = F.relu(self.conv1(xb))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)
        x = x.reshape(-1, 50*50*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
