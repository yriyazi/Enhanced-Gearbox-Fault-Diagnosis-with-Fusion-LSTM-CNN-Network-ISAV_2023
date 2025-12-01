import torch
import torch.nn as nn
import Utils

class InceptionBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channel: int):
        """
        Initialize the InceptionBlock with various branches for different convolutions.
        
        Args:
            in_channels (int): Number of input channels.
            out_channel (int): Number of output channels.
        """
        super(InceptionBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channel = out_channel

        # 1x1 convolution branch
        self.conv1x1        =   nn.Sequential(
                                            nn.Conv2d(self.in_channels, self.out_channel, kernel_size=1),
                                            nn.ReLU(inplace=True))
        
    
        # 1x1 conv followed by 3x3 conv branch
        self.conv3x3        =   nn.Sequential(
                                            nn.BatchNorm2d(self.in_channels),
                                            nn.Conv2d(self.in_channels, self.out_channel, kernel_size=1),
                                            # nn.Dropout2d(),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, padding='same'),
                                            nn.ReLU(inplace=True))


        # 1x1 conv followed by 5x5 conv branch
        self.conv5x5        =   nn.Sequential(
                                            nn.BatchNorm2d(self.in_channels),
                                            nn.Conv2d(self.in_channels, self.out_channel, kernel_size=1),
                                            # nn.Dropout2d(),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(self.out_channel, self.out_channel, kernel_size=5, padding='same'),
                                            nn.ReLU(inplace=True))

        
        # 1x1 conv followed by 9x9 conv branch
        self.conv9x9        =   nn.Sequential(
                                                nn.BatchNorm2d(self.in_channels),
                                                nn.Conv2d(self.in_channels, self.out_channel, kernel_size=1),
                                                # nn.Dropout2d(),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(self.out_channel, self.out_channel, kernel_size=9, padding='same'),
                                                nn.ReLU(inplace=True))
        # 3x3 pooling followed by 1x1 conv branch
        self.conv1x1_pool   =   nn.Sequential(
                                              nn.BatchNorm2d(self.in_channels),
                                              nn.MaxPool2d(kernel_size=3, stride=1, padding = 1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(self.in_channels,  self.out_channel, kernel_size=1),
                                              nn.ReLU(inplace=True),)
        
        self.head           =   nn.Sequential(  
                                                # nn.BatchNorm2d(self.in_channels),
                                                nn.Conv2d(self.out_channel*5, Utils.CNN_outFeature, kernel_size=(4,16),stride=2, padding=0),
                                                # nn.Dropout2d(),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(Utils.CNN_outFeature,    Utils.CNN_outFeature, kernel_size=(4,16),stride=3, padding=0),
                                                nn.ReLU(inplace=True),
                                            )
    def GAP(self, x: torch.Tensor) -> torch.Tensor:
        """
        Global Average Pooling (GAP) operation.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Result of GAP.
        """
        return torch.mean(x, dim=[2, 3])    
         
         
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the InceptionBlock.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            Tensor: Output tensor.
        """
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)
        out9x9 = self.conv9x9(x)
        out1x1_pool = self.conv1x1_pool(x)
        
        out = torch.cat([out1x1, out3x3, out5x5,out9x9, out1x1_pool], dim=1)

        out = self.GAP(self.head(out))
        return out