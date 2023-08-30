import torch
import torch.nn as nn
import Utils

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(InceptionBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channel = out_channel
        # 1x1 convolution branch
        
        self.conv1x1        =   nn.Conv2d(self.in_channels, self.out_channel, kernel_size=1)
        
    
        # 1x1 conv followed by 2x2 conv branch
        self.conv2x2_reduce =   nn.Conv2d(self.in_channels, self.out_channel, kernel_size=1)
        self.conv2x2        =   nn.Conv2d(self.out_channel, self.out_channel, kernel_size=2, padding='same')


        # 1x1 conv followed by 5x5 conv branch
        self.conv5x5_reduce =   nn.Conv2d(self.in_channels, self.out_channel, kernel_size=1)
        self.conv5x5        =   nn.Conv2d(self.out_channel, self.out_channel, kernel_size=5, padding='same')
        
        # 1x1 conv followed by 8x8 conv branch
        self.conv8x8_reduce =   nn.Conv2d(self.in_channels, self.out_channel, kernel_size=1)
        self.conv8x8        =   nn.Conv2d(self.out_channel, self.out_channel, kernel_size=8, padding='same')

        # 3x3 pooling followed by 1x1 conv branch
        self.pool           =   nn.MaxPool2d(kernel_size=3, stride=1, padding = 1)
        self.conv1x1_pool   =   nn.Conv2d(self.in_channels,  self.out_channel, kernel_size=1)

        self.head           =   model = nn.Sequential(
                                                        nn.Conv2d(50,100, kernel_size=(4,16),stride=2, padding=0),
                                                        nn.ReLU(),
                                                        nn.Conv2d(100,100, kernel_size=(4,16),stride=3, padding=0),
                                                        nn.ReLU(),
                                                    )
    def GAP(self, x):
        return torch.mean(x, dim=[2, 3])    
         
    def forward(self, x):
        out1x1 = self.conv1x1(x)
        
        out2x2 = self.conv2x2_reduce(x)
        out2x2 = self.conv2x2(out2x2)
        
        out5x5 = self.conv5x5_reduce(x)
        out5x5 = self.conv5x5(out5x5)

        out8x8 = self.conv8x8_reduce(x)
        out8x8 = self.conv8x8(out8x8)
        
        out1x1_pool = self.pool(x)
        out1x1_pool = self.conv1x1_pool(out1x1_pool)
        
        out = torch.cat([out1x1, out2x2, out5x5,out8x8, out1x1_pool], dim=1)

        out = self.GAP(self.head(out))
        return out