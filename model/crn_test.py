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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0)) # (batch_size, num_channels, height, width)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0))
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0))
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0))
        self.bn5 = nn.BatchNorm2d(num_features=256)

        # LSTM
        self.LSTM1 = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=2, batch_first=True)

        # Decoder
        self.pad1 = nn.ZeroPad2d((0, 0, 0, 1))
        self.convT1 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0))
        self.bnT1 = nn.BatchNorm2d(num_features=128)
        self.pad2 = nn.ZeroPad2d((0, 0, 0, 1))
        self.convT2 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0))
        self.bnT2 = nn.BatchNorm2d(num_features=64)
        self.pad3 = nn.ZeroPad2d((0, 0, 0, 1))
        self.convT3 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0))
        self.bnT3 = nn.BatchNorm2d(num_features=32)
        self.pad4 = nn.ZeroPad2d((0, 0, 0, 1))
        self.convT4 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0), output_padding=(0, 1))  # output_padding
        self.bnT4 = nn.BatchNorm2d(num_features=16)
        self.pad5 = nn.ZeroPad2d((0, 0, 0, 1))
        self.convT5 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0))
        self.bnT5 = nn.BatchNorm2d(num_features=1)

    def forward(self, x):
        # conv
        # (B, in_c, T, F)
        x.unsqueeze_(1)
        x1 = F.elu(self.bn1(self.conv1(x)[:, :, :-1, :]))
        x2 = F.elu(self.bn2(self.conv2(x1)[:, :, :-1, :]))
        x3 = F.elu(self.bn3(self.conv3(x2)[:, :, :-1, :]))
        x4 = F.elu(self.bn4(self.conv4(x3)[:, :, :-1, :]))
        x5 = F.elu(self.bn5(self.conv5(x4)[:, :, :-1, :]))

        # reshape
        out5 = x5.permute(0, 2, 1, 3)
        out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)
        # lstm
        lstm, (hn, cn) = self.LSTM1(out5)

        # reshape
        output = lstm.reshape(lstm.size()[0], lstm.size()[1], 256, -1)
        output = output.permute(0, 2, 1, 3)
        # ConvTrans
        res = torch.cat((output, x5), 1)
        res1 = F.elu(self.bnT1(self.convT1(self.pad1(res))))
        res1 = torch.cat((res1, x4), 1)
        res2 = F.elu(self.bnT2(self.convT2(self.pad2(res1))))
        res2 = torch.cat((res2, x3), 1)
        res3 = F.elu(self.bnT3(self.convT3(self.pad3(res2))))
        res3 = torch.cat((res3, x2), 1)
        res4 = F.elu(self.bnT4(self.convT4(self.pad4(res3))))
        res4 = torch.cat((res4, x1), 1)
        # (B, o_c, T. F)
        res5 = F.relu(self.bnT5(self.convT5(self.pad5(res4))))
        return res5.squeeze()
