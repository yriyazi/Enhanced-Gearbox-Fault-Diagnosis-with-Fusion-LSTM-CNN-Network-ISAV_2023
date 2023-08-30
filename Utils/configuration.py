import yaml

# Load config file
with open('config.yaml') as f:
    config     =    yaml.load(f, Loader=yaml.FullLoader)

# Access hyperparameters
Device              =    config['Device']
num_epochs          =    config['num_epochs']
seed                =    config['seed']

#datasets
samples_per_test    =    config['Dataset']['samples_per_test']
Total_test          =    config['Dataset']['Total_test']
test_Duration       =    config['Dataset']['test_Duration']

# Access model architecture parameters
input_horizon       =   config['model']['input_horizon']
# model-LSTM
LSTM_outFeature     =    config['model']['LSTM_outFeature']
LSTM_NumLayer       =    config['model']['LSTM_NumLayer']
LSTM_hidden_size    =    config['model']['LSTM_hidden_size']
# model-CNN
CNN_inChannel       =    config['model']['CNN_inChannel']
CNN_outChannel      =    config['model']['CNN_outChannel']
CNN_outFeature      =    config['model']['CNN_outFeature']  
# CWT
Scales              =    config['CWT']['Scales']
wavelet_lis         =    config['CWT']['wavelet_lis']
wavelet             =    wavelet_lis[config['CWT']['wavelet']]
Coefficient_Real    =    config['CWT']['Coefficient_Real']
