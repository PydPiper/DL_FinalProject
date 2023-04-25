import os
import random
# -------------------------------------------------------------------------------------------------------
# Import pip libs
# -------------------------------------------------------------------------------------------------------
import torch
import torchvision
from torch import nn
from torch.optim import Adam
from matplotlib import pyplot as plt
import numpy as np

# -------------------------------------------------------------------------------------------------------
# Import source modules
# -------------------------------------------------------------------------------------------------------
import utils_cold as utils
from corruptions import snow


if __name__ == '__main__':
    # image to corrupt; a 224x224x3
    severity = np.random.randint(1, 6)
    snow_layer = snow(np.ones((24, 24, 3)), severity)
    snow_tensor_kernel = utils.img_pil_to_tensor(snow_layer)[0]

