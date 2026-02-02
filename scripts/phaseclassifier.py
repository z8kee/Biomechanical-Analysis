import numpy as np, torch.nn as nn, torch.nn.functional as F, torch, pandas as pd
from torch.utils.data import Dataset, DataLoader

class PoseDataset(Dataset): # Create dataset objects for train and val splits
    def __init__(self, dataframe, npy_dir):
        self.dataframe = dataframe
        self.npy_dir = npy_dir
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        label = row['label']
        x = np.load(f"{self.npy_dir}/sample{row['sample_id']}.npy")
        y = label
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    

class PhaseClassifier(nn.Module):
    def __init__(self, num_classes=5, input_channels=132, window_size=30):
        super(PhaseClassifier, self).__init__()
        #Feat extrcttion, 1st conv layer looks at raw coordinates, output channels expands to 64
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        #2nd conv layer further extracts features and looks for pattern, output channels expands to 128
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        #3rd conv layer further extracts features, output channels expands to 256
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        #pooling reduces time dimensionality from 30 -> 15 -> 7 -> 3 by integer division of kernel_size each time
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.5)

        self.fc_input_dimension= 256 * (window_size // 8)  #after 3 pooling layers, window size reduced by factor of 8

        self.fc1 = nn.Linear(self.fc_input_dimension, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        #x shape: (batch, 30, 132), should be (batch, 132, 30)

        x = x.permute(0,2,1)

        #b1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        #b2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        #b3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        #flatten [batch, features, time] -> [batch, features * time]
        x = x.flatten(start_dim=1)

        #dense layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x) # no need for softmax, as CrossEntropyLoss does that

        return x