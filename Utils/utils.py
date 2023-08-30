import  torch
import  random
import  torch.nn            as      nn
import  numpy               as      np


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
