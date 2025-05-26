import random
import numpy as np
import torch
import pickle

# Set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Save model state and configuration
def save_model(state_dict, gnn_config, dir):
    pickle.dump(gnn_config, open('{}.config'.format(dir), 'wb'))
    torch.save({k:v.cpu() for k, v in state_dict.items()}, '{}.ckpt'.format(dir))
