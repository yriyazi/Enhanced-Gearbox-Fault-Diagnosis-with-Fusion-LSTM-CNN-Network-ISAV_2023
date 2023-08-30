import yaml

# Load config file
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Access hyperparameters
Device              = config['Device']
num_epochs          = config['num_epochs']
seed                = config['seed']

#datasets
samples_per_test    = config['datasets']['samples_per_test']
Total_test          = config['datasets']['Total_test']
test_Duration       = config['datasets']['test_Duration']

# Access model architecture parameters
input_horizon       = config['model']['input_horizon']
# model-LSTM
LSTM_outFeature     = config['model']['LSTM_outFeature']
LSTM_NumLayer       = config['model']['LSTM_NumLayer']
LSTM_hidden_size    = config['model']['LSTM_hidden_size']


# CWT
Scales              = config['CWT']['Scales']
wavelet_lis         = config['CWT']['wavelet_lis']
wavelet             = wavelet_lis[config['CWT']['wavelet']]
Coefficient_Real    = config['CWT']['Coefficient_Real']
