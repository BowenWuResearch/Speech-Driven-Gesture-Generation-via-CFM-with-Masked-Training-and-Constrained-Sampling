def seed_everything(seed):
    import torch
    import numpy
    import random
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)