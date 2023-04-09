import torch
import torchvision
from torch import nn
from matplotlib import pyplot as plt
import os
import numpy as np

# -------------------------------------------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------------------------------------------
os.chdir(os.path.dirname(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -------------------------------------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------------------------------------
def plot_img(imgs, cols=5):

    if len(imgs.shape) < 3:
        # this is a single img
        imgs = imgs.unsqueeze(dim=0)
    n_imgs = imgs.shape[0]

    plt.figure(figsize=(10,10))
    for i, img in enumerate(imgs):
        plt.subplot(int(np.ceil(n_imgs/cols)), cols, i+1)
        plt.imshow(img)
    plt.show()

def img_to_tensor():
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE))

# -------------------------------------------------------------------------------------------------------
# Diffusion Functions
# -------------------------------------------------------------------------------------------------------

def beta_scheduler(step=0.0001, start=0.0001, end=0.02, beta_type='linear'):
    # note step=1e-4 and end=0.2 is from the original DDPM paper: https://arxiv.org/abs/2102.09672
    # there is also cosine, sigmoid and other types to implement
    if beta_type == 'linear':
        rv = torch.linspace(start, end, step)
    else:
        raise ValueError(f'Incorrect beta scheduler type={beta_type}')
    return rv

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(device)

def forward_diffusion_sample(x_0, t):
    """takes an original img and returns a noised version of it at any given time step "t"

    :param x_0: _description_
    :type x_0: _type_
    :param t: _description_
    :type t: _type_
    :return: _description_
    :rtype: _type_
    """

    # using "reparameterization" we can calcluate any noised sample without iterating over the previous n-samples
    # x_t = sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*noise
    x_0 = x_0.to(device)

    # rand_like is nothing special just a normal distribution thats the same size as the input
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alpha_bar, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alpha_bar, t, x_0.shape)
    # mean + variance
    return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

# -------------------------------------------------------------------------------------------------------
# Diffusion Global Parameters
# -------------------------------------------------------------------------------------------------------

# Define beta schedule
# NOTE: T is also the size of beta
T = 300
beta = beta_scheduler(step=T)

# NOTE: alpha.shape == beta.shape, but alpha is a slow decrease from 1 to 1-beta_end 
alpha = 1. - beta
# alpha_bar is the product_sum of alpha (ie: alphas[0], alphas[0]*alphas[1], alphas[0]*alphas[1]*alphas[2],...)
alpha_bar = torch.cumprod(alpha, axis=0)
# alpha_bar_prev is simply just the start from 1 to alpha_bar[-1]
alpha_bar_prev = torch.cat((torch.tensor([1.]), alpha_bar[:-1]))
# used in x_t calc
sqrt_alpha_bar = torch.sqrt(alpha_bar)
sqrt_one_minus_alpha_bar = torch.sqrt(1. - alpha_bar)

# used in backward pass
posterior_variance = beta * (1. - alpha_bar_prev) / (1. - alpha_bar)


if __name__ == '__main__':

    # load MNIST dataset (hand written imgs)
    data_train = torchvision.datasets.MNIST('../data/', download=True, train=True).data
    data_test = torchvision.datasets.MNIST('../data/', download=True).data
    # visualize first 20 imgs
    plot_img(data_train[:20])