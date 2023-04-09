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

    plt.figure()
    for i, img in enumerate(imgs):
        ax = plt.subplot(int(np.ceil(n_imgs/cols)), cols, i+1)
        ax.set_axis_off()
        plt.imshow(img)
    plt.show()
    plt.close()

def visualize_input_imgs(first_n_imgs=20):
    data_train = torchvision.datasets.MNIST('../data/', download=True, train=True).data
    plot_img(data_train[:first_n_imgs])

def visualize_tensor_img(img):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda data: (data + 1) / 2), # recover all -1 to 0 values then div/2 to get it back to 0-1
        # torchvision.transforms.Lambda(lambda data: data.permute(1, 2, 0)), # swap from (channel, height, width) to (height, width, channel), note this is needed when using plt.imshow
        torchvision.transforms.Lambda(lambda data: (data.to('cpu') * 255.).numpy().astype(np.int8)), # bring it back from 0-1 to RGB 0-255 scale
        # transforms.ToPILImage(), # note this is only needed when using plt.imshow
    ])

    # if img is in batch form, take the 1st of batch
    if len(img.shape) == 4:
        img = img[0]
    img = transforms(img)
    plot_img(img)

def load_data():
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
        torchvision.transforms.ToTensor(), # scales from RGB 0-255 to 0-1
        torchvision.transforms.Lambda(lambda data: (data*2) - 1) # shift data to be -1 to 1
    ])
    data_train = torchvision.datasets.MNIST('../data/', download=True, train=True, transform=transforms)
    data_test = torchvision.datasets.MNIST('../data/', download=True, transform=transforms)
    data = torch.utils.data.ConcatDataset((data_train, data_test))
    dataloader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    return dataloader


# -------------------------------------------------------------------------------------------------------
# Diffusion Functions
# -------------------------------------------------------------------------------------------------------

def beta_scheduler(steps=300, start=0.0001, end=0.02, beta_type='linear'):
    # note step=1e-4 and end=0.2 is from the original DDPM paper: https://arxiv.org/abs/2102.09672
    # there is also cosine, sigmoid and other types to implement
    if beta_type == 'linear':
        rv = torch.linspace(start, end, steps)
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

def simluate_forward_diffusion(dataloader):
    image = next(iter(dataloader))[0]

    num_images = 10
    stepsize = int(T/num_images)

    imgs = []
    for idx in range(0, T, stepsize):
        t = torch.Tensor([idx]).type(torch.int64)
        img, noise = forward_diffusion_sample(image, t)
        imgs.append(img)
    imgs = torch.concat(imgs,dim=1)
    visualize_tensor_img(imgs)


if __name__ == '__main__':
    # -------------------------------------------------------------------------------------------------------
    # Hyperparameter Tuning
    # -------------------------------------------------------------------------------------------------------
    T = 300 # beta time steps
    IMG_SIZE = 24 # resize img to smaller than original helps with training (MNIST is already 24x24 though)
    BATCH_SIZE = 128 # batch size to process the imgs, larger the batch the more avging happens for gradient training updates


    # -------------------------------------------------------------------------------------------------------
    # Diffusion Global Parameters
    # -------------------------------------------------------------------------------------------------------
    # Define beta schedule
    # NOTE: T is also the size of beta
    beta = beta_scheduler(steps=T)
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

    # -------------------------------------------------------------------------------------------------------
    # Start of Process
    # -------------------------------------------------------------------------------------------------------
    # uncomment to see the input imgs
    visualize_input_imgs()

    dataloader = load_data()
    simluate_forward_diffusion(dataloader)
