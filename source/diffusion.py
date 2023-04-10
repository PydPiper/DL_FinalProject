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

def visualize_input_imgs(dataloader, n_imgs=1):
    iter_dataloader = iter(dataloader)
    plt.figure()
    for i in range(n_imgs):
        # NOTE: calling next() returns a [tensor(x), tensor(y)]
        # where x.shape = [batch size, channel, height, width]
        #       y.shape = [batch size]
        imgs = next(iter_dataloader)[0][0:n_imgs]
        for img in imgs:
            img = img_tensor_to_pil(img)
            ax = plt.subplot(1, n_imgs, i + 1)
            ax.set_axis_off()
            plt.imshow(img)
    plt.show()
    plt.close()

def img_tensor_to_pil(img):
    if len(img.shape) > 3:
        raise ValueError(f'Input img.shape={img.shape}, img.shape must be [channel, height, width]')
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda data: data.permute(1, 2, 0)), # swap from (channel, height, width) to (height, width, channel), note this is needed when using plt.imshow
        torchvision.transforms.Lambda(lambda data: (data / torch.abs(data).max() + 1) / 2), # recover all -1 to 0 values
        torchvision.transforms.Lambda(lambda data: (data.to('cpu') * 255.).numpy()), # bring it back from 0-1 to RGB 0-255 scale
        torchvision.transforms.ToPILImage(), # note this is only needed when using plt.imshow
    ])
    return transforms(img)

def visualize_tensor_img(img):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda data: (data / torch.abs(data).max() + 1) / 2),
        # torchvision.transforms.Lambda(lambda data: (data + 1) / 2), # recover all -1 to 0 values then div/2 to get it back to 0-1
        # torchvision.transforms.Lambda(lambda data: data.permute(1, 2, 0)), # swap from (channel, height, width) to (height, width, channel), note this is needed when using plt.imshow
        torchvision.transforms.Lambda(lambda data: (data.to('cpu') * 255.).numpy()), # bring it back from 0-1 to RGB 0-255 scale
        torchvision.transforms.ToPILImage(), # note this is only needed when using plt.imshow
    ])

    # if img is in batch form, take the 1st of batch
    if len(img.shape) == 4:
        img = img[0]
    img = transforms(img[0])
    # img will be in shape (n_imgs, h, c), so to plot them all in a row have cols=n_imgs
    plt.figure()
    plt.imshow(img)
    plt.show()
    # plot_img(img, cols=img.shape[0])

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
        rv = torch.linspace(start, end, steps).to(device)
    else:
        raise ValueError(f'Incorrect beta scheduler type={beta_type}')
    return rv

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
    # mean + variance
    return SQRT_ALPHA_BAR[t] * x_0 + SQRT_ONE_MINUS_ALPHA_BAR[t] * noise, noise

def simluate_forward_diffusion(dataloader, n_imgs=1, show_n_steps=5):

    iter_dataloader = iter(dataloader)
    imgs = next(iter_dataloader)[0][:n_imgs]

    stepsize = int(T/show_n_steps)
    plt.figure()
    for col_i, t in enumerate(range(0, T, stepsize)):
        # NOTE: calling next() returns a [tensor(x), tensor(y)]
        # where x.shape = [batch size, channel, height, width]
        #       y.shape = [batch size]
        imgs, noises = forward_diffusion_sample(imgs, t)
        for row_i, img in enumerate(imgs):
            img = img_tensor_to_pil(img)
            ax = plt.subplot(n_imgs, show_n_steps, row_i*show_n_steps + col_i + 1)
            ax.set_axis_off()
            plt.imshow(img)
    plt.show()
    plt.close()


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
    BETA = beta_scheduler(steps=T)
    # NOTE: alpha.shape == beta.shape, but alpha is a slow decrease from 1 to 1-beta_end 
    ALPHA = 1. - BETA
    # alpha_bar is the product_sum of alpha (ie: alphas[0], alphas[0]*alphas[1], alphas[0]*alphas[1]*alphas[2],...)
    ALPHA_BAR = torch.cumprod(ALPHA, axis=0)
    # alpha_bar_prev is simply just the start from 1 to alpha_bar[-1]
    ALPHA_BAR_PREV = torch.cat((torch.tensor([1.]).to(device), ALPHA_BAR[:-1]))
    # used in x_t calc
    SQRT_ALPHA_BAR = torch.sqrt(ALPHA_BAR)
    SQRT_ONE_MINUS_ALPHA_BAR = torch.sqrt(1. - ALPHA_BAR)
    # used in backward pass
    POSTERIOR_VARIANCE = BETA * (1. - ALPHA_BAR_PREV) / (1. - ALPHA_BAR)

    # -------------------------------------------------------------------------------------------------------
    # Start of Process
    # -------------------------------------------------------------------------------------------------------
    # uncomment to see the input imgs

    dataloader = load_data()
    # visualize_input_imgs(dataloader, 5)

    simluate_forward_diffusion(dataloader, n_imgs=3, show_n_steps=10)
