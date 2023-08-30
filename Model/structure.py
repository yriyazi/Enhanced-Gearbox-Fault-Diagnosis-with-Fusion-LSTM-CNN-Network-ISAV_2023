import                         Utils
import                         torch
import                         pywt
import                         torch.nn          as   nn
import                         numpy             as   np
from    .cnn_net    import     InceptionBlock
from    .lstm_net   import     LSTMModel



class Structure(nn.Module):

    def __init__(self,):
        super(Structure, self).__init__()
        self.LSTM          =     LSTMModel(Utils.input_horizon                 ).to(device = Utils.Device)
        self.CNN           =     InceptionBlock(Utils.CNN_inChannel,Utils.CNN_outChannel   ).to(device = Utils.Device)
        # CWT
        self.scales        =     np.arange(1, Utils.Scales)
        # MLP
        self.Classifier    =     nn.Linear(Utils.LSTM_outFeature*2,1)

        self.gig = nn.Sigmoid()

    def _CWT(self,  x:np.array):
        coefficients, _     =    pywt.cwt(x, self.scales, Utils.wavelet)
        if Utils.Coefficient_Real:
            coefficients    =    np.abs(coefficients)

        return coefficients



    def forward(self, x:np.array):
        coefficients    =    self._CWT(x)

        _L      =    torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device = Utils.Device)
        _L_out  =    self.LSTM(_L)

        _C      =    torch.tensor(coefficients, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device = Utils.Device)
        _C_out  =    inception_block = self.CNN(_C)

        out     =    torch.cat([_C_out,_L_out], dim=1)
        out     =    self.Classifier(out)

        return  self.gig(out)