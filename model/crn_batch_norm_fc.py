import torch.nn as nn
import torch.nn.functional as F
import torch

class CRNN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """
    def __init__(self):
        super(CRNN, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 6), stride=(1, 2), padding=(1, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 6), stride=(1, 2), padding=(1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 6), stride=(1, 2), padding=(1, 0), bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 6), stride=(1, 2), padding=(1, 0), bias=False)
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 6), stride=(1, 2), padding=(1, 0), bias=False)
        self.bn5 = nn.BatchNorm2d(num_features=256)

        # LSTM
        # for 512 samples / frame
        self.LSTM1 = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=2, batch_first=True, bidirectional=True)

        # FC layer
        self.Linear = nn.Linear(in_features=2048, out_features=257)


    def forward(self, x):
        # conv
        # (B, in_c, T, F)
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x5 = F.relu(self.bn5(self.conv5(x4)))
        # reshape
        out5 = x5.permute(0, 2, 1, 3)
        out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)
        # lstm
        lstm, (hn, cn) = self.LSTM1(out5)


        # FC layer
        output = F.relu(self.Linear(lstm))
        res5 = output.reshape(output.size()[0], output.size()[1], 257, 1)
        res5 = res5.permute(0, 3, 1, 2)

        return res5.squeeze()
