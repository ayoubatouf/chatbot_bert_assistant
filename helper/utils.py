import json
import random
import numpy as np
import torch


def load_json_data(filename):
    with open(filename) as f:
        file_content = json.load(f)
    return file_content


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
