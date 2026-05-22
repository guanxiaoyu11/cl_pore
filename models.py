import torch
import torch.nn as nn
from torch.autograd import Variable
import torch


class Self_Attn_Spatial(nn.Module):
    """
    func: Self attention Spatial Layer 
    inputs:
        in_dim: 
        out_dim: 
    """

    def __init__(self, in_dim, out_dim):
        super(Self_Attn_Spatial, self).__init__()
        self.chanel_in = in_dim
        self.out_dim = out_dim

        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, length = x.size()

        
        proj_query = self.query_conv(x).view(m_batchsize, -1, length).permute(0, 2, 1)

        
        proj_key = self.key_conv(x).view(m_batchsize, -1, length)

        
        energy = torch.bmm(proj_query, proj_key)  # transpose check

        
        attention = self.softmax(energy)  # B X N X N

        proj_value = self.value_conv(x).view(m_batchsize, -1, length)  # B X C X N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B X C X N
        out = out.view(m_batchsize, C,length)  # B X C X W X H

        
        out = self.gamma * out + x  # B X C X W X H

        # return out, attention

        return out

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, padding=2, dilation=1, stride=1):
        super().__init__()

        if bn:
            self.basic_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 7, padding=padding, dilation=dilation, stride=stride),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )
        else:
            self.basic_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 5, padding=padding, dilation=dilation, stride=stride),
                nn.ReLU()
            )

    def forward(self, x):
        return self.basic_block(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(2, 4, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.convBlock = nn.Sequential(
            Block(1, 8, False),
            # Self_Attn_Spatial(8, 16),
            nn.MaxPool1d(kernel_size=2),
            Block(8, 16, False),
            # Self_Attn_Spatial(16, 32),
            nn.MaxPool1d(kernel_size=2),
            Block(16, 32, False),
            # Self_Attn_Spatial(32, 64),
            nn.MaxPool1d(kernel_size=2),
            Block(32, 64, False),
            # Self_Attn_Spatial(64, 64),
            nn.MaxPool1d(kernel_size=2),
            Block(64, 128, False),
            nn.MaxPool1d(kernel_size=2),
            Block(128, 256, False),
            nn.MaxPool1d(kernel_size=2),
            Block(256, 512, False),
            nn.MaxPool1d(kernel_size=2),
            Block(512, 256, False),
            nn.MaxPool1d(kernel_size=1),
            Block(256, 128, False),
            nn.MaxPool1d(kernel_size=1),
            Block(128, 64, False),
            nn.MaxPool1d(kernel_size=1),
            Block(64, 32, False),
            nn.MaxPool1d(kernel_size=1),
            Block(32, 16, False),
            nn.MaxPool1d(kernel_size=1),

            # Block(512, 1024, False),
            # nn.MaxPool1d(kernel_size=2)
        )
        self.classification = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(16, 16)
            # nn.Linear(1024, 7)
        )

    def forward(self, x,x1):
        x = self.convBlock(x)
        x1 = self.convBlock(x1)
        # x = Self_Attn_Spatial().to(torch.device('cuda:0'))(x)
        x, _ = torch.max(x, dim=2)
        x1, _ = torch.max(x1, dim=2)

        return self.classification(x),self.classification(x1),x,x1



def preprocessing(batch):
    batch_size, _, n_peaks = batch.shape
    processed_batch = batch.clone().view(batch_size, n_peaks)
    # TO DO: rewrite without loop
    for x in processed_batch:
        x[x < 1e-4] = 0
        pos = (x != 0)
        x[pos] = torch.log10(x[pos])
        x[pos] = x[pos] - torch.min(x[pos])
        x[pos] = x[pos] / torch.max(x[pos])
    return processed_batch.view(batch_size, 1, n_peaks)


class Integrator(nn.Module):
    def __init__(self, length=256):
        super().__init__()

        self.starter = nn.Sequential(
            Block(2, 16),
            Block(16, 20),
            nn.AvgPool1d(kernel_size=2),
            Block(20, 24),
            nn.AvgPool1d(kernel_size=2),
            Block(24, 28),
            nn.AvgPool1d(kernel_size=2),
            Block(28, 32)
        )

        self.pass_down1 = nn.Sequential(
            nn.AvgPool1d(kernel_size=2),
            Block(32, 48)
        )

        self.pass_down2 = nn.Sequential(
            nn.AvgPool1d(kernel_size=2),
            Block(48, 64)
        )

        self.code = nn.Sequential(
            nn.AvgPool1d(kernel_size=2),
            Block(64, 96),
            nn.Upsample(scale_factor=2),
            Block(96, 64)
        )

        self.pass_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Block(128, 64)
        )

        self.pass_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Block(112, 48)
        )

        self.finisher = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Block(80, 64),
            nn.Upsample(scale_factor=2),
            Block(64, 32),
            nn.Upsample(scale_factor=2),
            Block(32, 16),
            nn.Conv1d(16, 2, 1, padding=0)
        )

    def forward(self, x):
        x = torch.cat((x, preprocessing(x)), dim=1)
        starter = self.starter(x)
        pass1 = self.pass_down1(starter)
        pass2 = self.pass_down2(pass1)
        x = self.code(pass2)
        x = torch.cat((x, pass2), dim=1)
        x = self.pass_up2(x)
        x = torch.cat((x, pass1), dim=1)
        x = self.pass_up1(x)
        x = torch.cat((x, starter), dim=1)
        x = self.finisher(x)
        return x

