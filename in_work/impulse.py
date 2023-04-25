# -------------------------------------------------------------------------------------------------------
# Import standard libs
# -------------------------------------------------------------------------------------------------------
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
import skimage as sk

# -------------------------------------------------------------------------------------------------------
# Import source modules
# -------------------------------------------------------------------------------------------------------
import utils_cold as utils

# -------------------------------------------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------------------------------------------
os.chdir(os.path.dirname(__file__))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------------------------------------------------------------------
# Custom Diffusion Functions (custom diffusion functions go here)
# -------------------------------------------------------------------------------------------------------

def beta_scheduler(steps=300, start=0.0001, end=0.02, beta_type='linear'):
    # note step=1e-4 and end=0.2 is from the original DDPM paper: https://arxiv.org/abs/2102.09672
    # there is also cosine, sigmoid and other types to implement
    if beta_type == 'linear':
        rv = torch.linspace(start, end, steps).to(DEVICE)
    else:
        raise ValueError(f'Incorrect beta scheduler type={beta_type}')
    return rv

# -------------------------------------------------------------------------------------------------------
# Model Training Functions (do not rename functions here, they are used by trainer)
# -------------------------------------------------------------------------------------------------------

def forward_diffusion_sample(x_0, t):
    """takes an original img and returns a noised version of it at any given time step "t"
    :param x_0: _description_
    :type x_0: _type_
    :param t: _description_
    :type t: _type_
    :return: _description_
    :rtype: _type_
    """
    output = torch.tensor(sk.util.random_noise(np.array(x_0.cpu()), mode='s&p', amount=.06))
    return output.to(x_0.dtype).to(DEVICE), None


@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    # REVISED FOR THE COLD DIFFUSION SAMPLING WHICH SHOULD BE BETTER WHEN SWITCHING TO DETERMINISTIC
    if t == 0:
        return x
    else:
        x_pred = model(x, t)
        if isinstance(t, int):
            t_int = t
        else:
            t_int = t.item()
        output1, _ = forward_diffusion_sample(x_pred, t_int)
        output2, _ = forward_diffusion_sample(x_pred, t_int-1)
        if SAMPLING_METHOD == "naive":
            return output2
        else:
            return x - output1 + output2

# -------------------------------------------------------------------------------------------------------
# Star of Process
# -------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # reproducability seeding
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    # -------------------------------------------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------------------------------------------
    DIFFUSION_NAME = 'impulse'
    DATASET = 'MNIST' # MNIST CIFAR10 CelebA
    IMG_SIZE = 24 # resize img to smaller than original helps with training (MNIST is already 24x24 though)
    TRAIN = True # True will train a new model and save it in ../trained_model/ otherwise it will try to load one if it exist
    SHOW_PLOTS = False
    # -------------------------------------------------------------------------------------------------------
    # Hyperparameter Tuning
    # -------------------------------------------------------------------------------------------------------
    T = 50 # (for gaussian this is called beta time steps)
    BATCH_SIZE = 64 # batch size to process the imgs, larger the batch the more avging happens for gradient training updates
    LEARNING_RATE = 2e-5
    EPOCHS = 10
    SAMPLING_METHOD = "AGLO2"
    # -------------------------------------------------------------------------------------------------------
    # Diffusion Global Parameters
    # -------------------------------------------------------------------------------------------------------
    # Define beta schedule
    # NOTE: T is also the size of beta
    BETA = beta_scheduler(steps=T, end=0.1)
    # NOTE: alpha.shape == beta.shape, but alpha is a slow decrease from 1 to 1-beta_end 
    ALPHA = 1. - BETA
    # alpha_bar is the product_sum of alpha (ie: alphas[0], alphas[0]*alphas[1], alphas[0]*alphas[1]*alphas[2],...)
    ALPHA_BAR = torch.cumprod(ALPHA, axis=0)
    # alpha_bar_prev is simply just the start from 1 to alpha_bar[-1]
    ALPHA_BAR_PREV = torch.cat((torch.tensor([1.]).to(DEVICE), ALPHA_BAR[:-1]))
    # used in x_t calc
    SQRT_ALPHA_BAR = torch.sqrt(ALPHA_BAR)
    SQRT_ONE_MINUS_ALPHA_BAR = torch.sqrt(1. - ALPHA_BAR)
    # used in backward pass
    SQRT_RECIP_ALPHA = torch.sqrt(1.0 / ALPHA)
    POSTERIOR_VARIANCE = BETA * (1. - ALPHA_BAR_PREV) / (1. - ALPHA_BAR)

    # -------------------------------------------------------------------------------------------------------
    # Start of Process
    # -------------------------------------------------------------------------------------------------------
    os.makedirs(f'../results/{DATASET}/{DIFFUSION_NAME}', exist_ok=True)

    data_train, data_valid = utils.load_data(DATASET, IMG_SIZE, BATCH_SIZE)
    # NOTE: [0] for 0th sample, this returns the x,y as a tuple, we want the img only so again [0], the shape will be [channel, height, width]
    IMG_CHANNELS = data_train.dataset[0][0].shape[0]
    # -------------------------------------------------------------------------------------------------------
    # CREATE GAUSSIAN MASK
    # -------------------------------------------------------------------------------------------------------
    GAUSSIAN_MASK = torch.zeros((T, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)).to(DEVICE)
    variance = 1
    for t in range(T):
        for c in range(IMG_CHANNELS):
            sigma = 1
            x = torch.arange(-IMG_SIZE // 2 + 1, IMG_SIZE // 2 + 1, dtype=torch.float32).to(DEVICE)
            y = torch.arange(-IMG_SIZE // 2 + 1, IMG_SIZE // 2 + 1, dtype=torch.float32).to(DEVICE)
            y = y[:, None]
            kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * variance))
            kernel = 1 - kernel / kernel.max()
            if t == 0:
                GAUSSIAN_MASK[t, c, :] = kernel
            else:
                GAUSSIAN_MASK[t, c, :] = kernel*GAUSSIAN_MASK[t-1, c, :]
        variance += 0.1

    # show sample imgs from dataset
    utils.visualize_input_imgs(data_train, 3, DATASET, DIFFUSION_NAME, SHOW_PLOTS)

    # show what fwd diffusion looks like
    utils.simluate_forward_diffusion(data_train, forward_diffusion_sample, max_time=T, n_imgs=5, show_n_steps=10, 
                                     dataset=DATASET, diffusion_name=DIFFUSION_NAME, show_plots=SHOW_PLOTS)




    # run training
    model = utils.Unet(IMG_CHANNELS)
    model.to(DEVICE)
    SAVED_MODEL_FILENAME = f'../results/{DATASET}/{DIFFUSION_NAME}/{DIFFUSION_NAME}_{DATASET}.model'
    if TRAIN:
        model = utils.train(model, LEARNING_RATE, EPOCHS, BATCH_SIZE, data_train, data_valid, T, IMG_CHANNELS, IMG_SIZE, DATASET,
          DIFFUSION_NAME, SHOW_PLOTS, sample_timestep, SAVED_MODEL_FILENAME, forward_diffusion_sample)
    elif os.path.exists(SAVED_MODEL_FILENAME):
        model = utils.load_model(SAVED_MODEL_FILENAME, IMG_CHANNELS)
    else:
        raise FileNotFoundError('Missing training model')