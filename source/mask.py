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

# -------------------------------------------------------------------------------------------------------
# Import source modules
# -------------------------------------------------------------------------------------------------------
import utils

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

    # t can come in for n_batch size, need to reshape to multiply
    if not isinstance(t, int):
        t = t.reshape((t.shape[0],1,1,1))
    else:
        t = torch.full(size=(x_0.shape[0],1,1,1), fill_value=t)

    # using "reparameterization" we can calcluate any noised sample without iterating over the previous n-samples
    # x_t = sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*noise
    x_0 = x_0.to(DEVICE)
    # rand_like is nothing special just a normal distribution thats the same size as the input
    batch, channel, height, width = x_0.shape
    percent_mask = t / T
    i_count_to_mask = (torch.floor((height * width) * percent_mask)).int()
    noised_x = x_0.clone().detach()
    noised_x = noised_x.reshape((batch, channel, height * width))

    # test 1: noise is the same masked portion as the img masking (full noise masking back to x_0)
    noise = torch.zeros_like(noised_x)
    for i in range(batch):
        noised_x[i,:,:i_count_to_mask[i]] = 1
        noise[i,:,:i_count_to_mask[i]] = 1
    noised_x = noised_x.reshape((batch, channel, height, width))
    noised_x = noised_x * BETA[t]
    noise = noise.reshape((batch, channel, height, width))
    noise = noise

    # test 1.5: noise is the same masked portion as the img masking (delta noise masking back to x_t-1)
    # noise = torch.zeros_like(noised_x)
    # i_count_to_mask_prev = i_count_to_mask - 1
    # i_count_to_mask_prev[i_count_to_mask_prev < 0] = 0
    # for i in range(batch):
    #     noised_x[i,:,:i_count_to_mask[i]] = 1
    #     noise[i,:,i_count_to_mask_prev[i]:i_count_to_mask[i]] = 1
    # noised_x = noised_x.reshape((batch, channel, height, width))
    # noised_x = noised_x * BETA[t]
    # noise = noise.reshape((batch, channel, height, width))
    # noise = noise

    # test 1.75: noise is the same masked portion as the img masking (delta img masking back to x_t-1)
    noised_x_prev = noised_x.clone().detach()
    percent_mask_prev = (t-1) / T
    i_count_to_mask_prev = (torch.floor((height * width) * percent_mask_prev)).int()
    i_count_to_mask_prev[i_count_to_mask_prev < 0] = 0
    for i in range(batch):
        noised_x[i,:,:i_count_to_mask[i]] = 1
        noised_x_prev[i,:,:i_count_to_mask_prev[i]] = 1
    noised_x = noised_x.reshape((batch, channel, height, width))
    noised_x_prev = noised_x_prev.reshape((batch, channel, height, width))
    noise = noised_x - noised_x_prev 

    # test 2: noise is delta of original
    # for i in range(batch):
    #     noised_x[i,:,:i_count_to_mask[i]] = 1
    # noised_x = noised_x.reshape((batch, channel, height, width))
    # noised_x = noised_x * BETA[t]
    # noise = noised_x - x_0 


    # test 3: noise is full 1s (this makes the most sense thats most similar to gaussian noise with noise being simply the gaussian)
    # for i in range(batch):
    #     noised_x[i,:,:i_count_to_mask[i]] = 1
    # noised_x = noised_x.reshape((batch, channel, height, width))
    # noised_x = noised_x * BETA[t]
    # noise = torch.ones_like(noised_x)


    # test 4
    # for i in range(batch):
    #     noised_x[i,:,:i_count_to_mask[i]] = 1
    # noised_x = noised_x.reshape((batch, channel, height, width))
    # noise = torch.randn_like(x_0)
    # # mean + variance
    # noised_x = SQRT_ALPHA_BAR[t] * noised_x + SQRT_ONE_MINUS_ALPHA_BAR[t] * noise

    # mean + variance
    return noised_x, noise


@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    
    pred_noise = model(x, t)
    batch, channel, height, width = x.shape
    pred_noise = pred_noise.reshape((batch, channel, height * width))
    percent_mask = t / T
    i_count_to_mask = (torch.floor((height * width) * percent_mask)).int()
    pred_noise[:,:,:i_count_to_mask] = 0
    pred_noise = pred_noise.reshape((batch, channel, height, width))
    x_prev = x - pred_noise

    # x_prev = SQRT_RECIP_ALPHA[t] * (x - BETA[t] * pred_noise / SQRT_ONE_MINUS_ALPHA_BAR[t])

    return x_prev    





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
    DIFFUSION_NAME = 'mask'
    DATASET = 'MNIST' # MNIST CIFAR10 CelebA
    IMG_SIZE = 24 # resize img to smaller than original helps with training (MNIST is already 24x24 though)
    TRAIN = True # True will train a new model and save it in ../trained_model/ otherwise it will try to load one if it exist
    SHOW_PLOTS = False
    
    # -------------------------------------------------------------------------------------------------------
    # Hyperparameter Tuning
    # -------------------------------------------------------------------------------------------------------
    T = 300 # (for gaussian this is called beta time steps)
    BATCH_SIZE = 128 # batch size to process the imgs, larger the batch the more avging happens for gradient training updates
    LEARNING_RATE = 0.001
    EPOCHS = 10

    # -------------------------------------------------------------------------------------------------------
    # Diffusion Global Parameters
    # -------------------------------------------------------------------------------------------------------
    # Define beta schedule
    # NOTE: T is also the size of beta
    BETA = beta_scheduler(steps=T)
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