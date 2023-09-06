# Import necessary libraries and modules
import Utils
import torch
import pywt
import torch.nn as nn
import numpy as np
from .cnn_net import InceptionBlock
from .lstm_net import LSTMModel

# Define a class for Continuous Wavelet Transform (CWT)
class _CWT():
    def __init__(self) -> None:
        # Initialize CWT with a range of scales
        self.scales = np.arange(1, Utils.Scales)

    # Perform forward pass for CWT
    def forward(self, x):
        coefficients, _ = pywt.cwt(x, self.scales, Utils.wavelet)
        if Utils.Coefficient_Real:
            coefficients = np.abs(coefficients)

        return coefficients

# Define a neural network structure that combines CNN, LSTM, CWT, and MLP
class Structure_CNN_RNN(nn.Module):
    def __init__(self,):
        super(Structure_CNN_RNN, self).__init__()
        
        # Initialize LSTM and CNN modules
        self.LSTM = LSTMModel(Utils.input_horizon).to(device=Utils.Device)
        self.CNN = InceptionBlock(Utils.CNN_inChannel, Utils.CNN_outChannel).to(device=Utils.Device)
        
        # Initialize CWT and MLP
        self.CWT = _CWT()
        _out = Utils.LSTM_outFeature + Utils.CNN_outFeature
        self.Classifier = nn.Sequential(
            nn.Linear(_out, _out // 5),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(_out // 5, 10),
            # nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(10, 2),
        )

    # Perform forward pass through the network
    def forward(self, x: np.array):
        coefficients = self.CWT.forward(x)

        _L = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device=Utils.Device)
        _L_out = self.LSTM(_L)

        _C = torch.tensor(coefficients, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device=Utils.Device)
        _C_out = self.CNN(_C)

        out = torch.cat([_C_out, _L_out], dim=1)
        out = self.Classifier(out)

        return out

# Define a neural network structure that consists of only CNN and CWT
class Structure_CNN(nn.Module):
    def __init__(self,):
        super(Structure_CNN, self).__init__()
        
        # Initialize CNN and CWT modules
        self.CNN = InceptionBlock(Utils.CNN_inChannel, Utils.CNN_outChannel).to(device=Utils.Device)
        self.CWT = _CWT()
        
        # Initialize MLP
        _out = Utils.CNN_outFeature
        self.Classifier = nn.Sequential(
            nn.Linear(_out, _out // 5),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(_out // 5, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
        )

    # Perform forward pass through the network
    def forward(self, x: np.array):
        coefficients = self.CWT.forward(x)

        _C = torch.tensor(coefficients, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device=Utils.Device)
        _C_out = self.CNN(_C)
        out = self.Classifier(_C_out)

        return out

# Define a neural network structure that consists of only LSTM
class Structure_RNN(nn.Module):
    def __init__(self,):
        super(Structure_RNN, self).__init__()
        
        # Initialize LSTM module
        self.LSTM = LSTMModel(Utils.input_horizon).to(device=Utils.Device)
        
        # Initialize MLP
        _out = Utils.LSTM_outFeature
        self.Classifier = nn.Sequential(
            nn.Linear(_out, _out // 5),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(_out // 5, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
        )

    # Perform forward pass through the network
    def forward(self, x: np.array):
        _L = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device=Utils.Device)
        _L_out = self.LSTM(_L)
        out = self.Classifier(_L_out)
        return out
    
##############################################################################
#################################For debug####################################
##############################################################################

class Simp_Conv(nn.Module):
    def __init__(self,
                 ):
        super(Simp_Conv, self).__init__()
        self.in_channels = 1
        self.Conv = torch.nn.Sequential(
                                        nn.BatchNorm2d(self.in_channels),
                                        nn.Conv2d(self.in_channels, 50, kernel_size=3),
                                        nn.ReLU(),
                                        nn.Conv2d(50, 50, kernel_size=5,padding = 0,stride=2),
                                        nn.ReLU(),
                                        nn.Conv2d(50, 100, kernel_size=5,padding = 0,stride=2),
                                        nn.ReLU(),
                                        nn.Conv2d(100, 100, kernel_size=5,padding = 0,stride=2),
                                        nn.ReLU(),)

        self.fc = torch.nn.Sequential(
                                        nn.Linear(100,50),
                                        nn.Dropout(p=0.5),
                                        nn.ReLU(),
                                        nn.Linear(50,2),
                                        )

        self.scales        =     np.arange(1, Utils.Scales)
        
    def _CWT(self,  x:np.array):
        coefficients, _     =    pywt.cwt(x, self.scales, Utils.wavelet)
        if Utils.Coefficient_Real:
            coefficients    =    np.abs(coefficients)

        return coefficients


    def forward(self,
                data,
                ):
        coefficients    =    self._CWT(data)

        coefficients = torch.tensor(coefficients, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device = Utils.Device)
        
        out = self.Conv(coefficients)
        # out = self.fc(out.mean(dim=-1).view(self.ssshape))
        out = out.mean(dim=-1).mean(dim=-1)

        out = self.fc(out)
        return out#.view(1,2)
    


from    torchvision.models          import  resnet50, ResNet50_Weights
from    torchvision.transforms      import  Resize
import                         pywt


class ResNetFeatureExtractor(nn.Module):
    def __init__(self,
                 cut    = -1,
                 ):
        super(ResNetFeatureExtractor, self).__init__()
        self.resnet  = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
        # Remove the classification layer (the last fully connected layer)
        # and pooling layer
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:cut])
        self.ssshape = 2048
        self.fc = torch.nn.Sequential(nn.Linear(self.ssshape,500),
                                        # nn.Dropout(p=0.5),
                                        nn.ReLU(),
                                        nn.Linear(500,20),
                                        # nn.Dropout(p=0.5),
                                        nn.ReLU(),
                                        nn.Linear(20,2),)

        self.scales        =     np.arange(1, Utils.Scales)
        
        # for name, param in self.resnet.named_parameters():
        #     if name.startswith('0') or name.startswith('1') or name.startswith('4') or name.startswith('5'):
        #         param.requires_grad = False
        # for param in self.resnet.parameters():
        #         param.requires_grad = False
                
    def _CWT(self,  x:np.array):
        coefficients, _     =    pywt.cwt(x, self.scales, Utils.wavelet)
        if Utils.Coefficient_Real:
            coefficients    =    np.abs(coefficients)

        return coefficients


    def forward(self,
                data,
                ):
        coefficients    =    self._CWT(data)

        coefficients = np.repeat(coefficients[:, :, np.newaxis], 3, axis=2)
        coefficients = torch.tensor(coefficients, dtype=torch.float32).permute(-1,0,1).unsqueeze(0).to(device = Utils.Device)
        
        out = self.resnet(coefficients)
        out = self.fc(out.mean(dim=-1).view(self.ssshape))

        return out.view(1,2)
    def check(self,):
        # Verify requires_grad for each parameter
        for name, param in self.resnet.named_parameters():
            print(name, param.requires_grad)