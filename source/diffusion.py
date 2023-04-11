import torch
import torchvision
from torch import nn
from torch.optim import Adam
from matplotlib import pyplot as plt
import os
import numpy as np
import random
import skimage as sk
from .corrupt_image import *

# -------------------------------------------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------------------------------------------
os.chdir(os.path.dirname(__file__))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
            plt.imshow(img, cmap='gray')
    plt.show()
    plt.close()

def img_tensor_to_pil(img):
    if len(img.shape) > 3:
        raise ValueError(f'Input img.shape={img.shape}, img.shape must be [channel, height, width]')
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda data: data.permute(1, 2, 0)), # swap from (channel, height, width) to (height, width, channel), note this is needed when using plt.imshow
        torchvision.transforms.Lambda(lambda data: (data / torch.abs(data).max() + 1) / 2), # recover all -1 to 0 values
        torchvision.transforms.Lambda(lambda data: (data.to('cpu') * 255.).numpy().astype(np.uint64)), # bring it back from 0-1 to RGB 0-255 scale
        # torchvision.transforms.ToPILImage(), # note this is only needed when using plt.imshow
    ])
    return transforms(img)


def img_tensors_to_pil(imgs):
    if len(imgs.shape) > 4:
        raise ValueError(f'Input img.shape={imgs.shape}, img.shape must be [n, channel, height, width]')
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda data: data.permute(0, 2, 3, 1)), # swap from (channel, height, width) to (height, width, channel), note this is needed when using plt.imshow
        torchvision.transforms.Lambda(lambda data: (data / torch.abs(data).max() + 1) / 2), # recover all -1 to 0 values
        torchvision.transforms.Lambda(lambda data: (data.to('cpu') * 255.).numpy().astype(np.uint64)), # bring it back from 0-1 to RGB 0-255 scale
        # torchvision.transforms.ToPILImage(), # note this is only needed when using plt.imshow
    ])
    return transforms(imgs)



def load_data(dataset='MNIST'):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
        torchvision.transforms.ToTensor(), # scales from RGB 0-255 to 0-1
        torchvision.transforms.Lambda(lambda data: (data*2) - 1) # shift data to be -1 to 1
    ])
    data_train = getattr(torchvision.datasets, dataset)('../data/', download=True, train=True, transform=transforms)
    # data_train = torch.utils.data.Subset(data_train, torch.arange(0,10e3))
    data_train = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    data_test = getattr(torchvision.datasets, dataset)('../data/', download=True, train=False, transform=transforms)
    data_test = torch.utils.data.DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    return data_train, data_test


# -------------------------------------------------------------------------------------------------------
# Diffusion Functions
# -------------------------------------------------------------------------------------------------------

def beta_scheduler(steps=300, start=0.0001, end=0.02, beta_type='linear'):
    # note step=1e-4 and end=0.2 is from the original DDPM paper: https://arxiv.org/abs/2102.09672
    # there is also cosine, sigmoid and other types to implement
    if beta_type == 'linear':
        rv = torch.linspace(start, end, steps).to(DEVICE)
    else:
        raise ValueError(f'Incorrect beta scheduler type={beta_type}')
    return rv

def forward_diffusion_sample(x_0, t, corruption='gaussian'):
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

    # using "reparameterization" we can calcluate any noised sample without iterating over the previous n-samples
    # x_t = sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*noise
    x_0 = x_0.to(DEVICE)
    # rand_like is nothing special just a normal distribution thats the same size as the input
    noise = torch.randn_like(x_0)
    # mean + variance
    corrupt_imgs = SQRT_ALPHA_BAR[t] * x_0 + SQRT_ONE_MINUS_ALPHA_BAR[t] * noise
    return corrupt_imgs, noise


def fwd_dif_impulse_pil(imgs):
    # assume a vector of images are coming in with [num_imgs, height, width, ch]
    c_range = [.03, .06, .09, 0.17, 0.27]
    c = 0.17
    for i, img in enumerate(imgs):
        x = sk.util.random_noise(np.array(img) / 255., mode='s&p', amount=c)
        imgs[i] = np.clip(x, 0, 1) * 255

    return imgs

def simluate_forward_diffusion(dataloader, n_imgs=1, show_n_steps=5):
    noise_model = "gaussian"
    iter_dataloader = iter(dataloader)
    imgs = next(iter_dataloader)[0][:n_imgs]
    stepsize = int(T/show_n_steps)
    plt.figure()
    if noise_model == "gaussian":
        for col_i, t in enumerate(range(0, T, stepsize)):
            # NOTE: calling next() returns a [tensor(x), tensor(y)]
            # where x.shape = [batch size, channel, height, width]
            #       y.shape = [batch size]
            imgs, noises = forward_diffusion_sample(imgs, t)
            for row_i, img in enumerate(imgs):
                img = img_tensor_to_pil(img)
                ax = plt.subplot(n_imgs, show_n_steps, row_i*show_n_steps + col_i + 1)
                ax.set_axis_off()
                plt.imshow(img, cmap='gray') # NOTE: matplotlib makes grayscale color by default unless you call out cmap=gray
    else:
        # try with an impulse noise corruption as a test
        # todo - make modular
        # convert images from [n, ch, height, width] tensor to PIL [n, height, width, ch]
        imgs = img_tensors_to_pil(imgs)
        for col_i, t in enumerate(range(0, T, stepsize)):
            # NOTE: calling next() returns a [tensor(x), tensor(y)]
            # where x.shape = [batch size, channel, height, width]
            #       y.shape = [batch size]
            imgs = fwd_dif_impulse_pil(imgs)
            for row_i, img in enumerate(imgs):
                # img = img_tensor_to_pil(img)
                ax = plt.subplot(n_imgs, show_n_steps, row_i*show_n_steps + col_i + 1)
                ax.set_axis_off()
                plt.imshow(img, cmap='gray')


    plt.show()
    plt.close()

# -------------------------------------------------------------------------------------------------------
# Diffusion Models
# -------------------------------------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, dim_in_ch, dim_out_ch, dim_time_emb, up_sample=False):
        super().__init__()
        self.linear =  nn.Linear(dim_time_emb, dim_out_ch)
        if up_sample:
            self.conv1 = nn.Conv2d(2*dim_in_ch, dim_out_ch, kernel_size=3, padding=1)
            self.transform = nn.ConvTranspose2d(dim_out_ch, dim_out_ch, kernel_size=5)
        else:
            self.conv1 = nn.Conv2d(dim_in_ch, dim_out_ch, kernel_size=3, padding=1)
            self.transform = nn.Conv2d(dim_out_ch, dim_out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim_out_ch, dim_out_ch, kernel_size=3)
        self.norm = nn.BatchNorm2d(dim_out_ch)
        self.relu  = nn.ReLU()

    def forward(self, x, t):
        # First conv
        h = self.norm(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.linear(t))
        # Extend last 2 dimensions so we can add h + time_emb
        time_emb = time_emb.reshape(time_emb.shape + (1,1))
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.norm(self.relu(self.conv2(h)))
        # Down or Upsample
        h = self.transform(h)
        return h


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=DEVICE) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Unet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = IMG_CHANNELS
        initial_projection_kernel = 3 # 7
        initial_projection_padding = 1 # 3
        starting_channels = 64
        self.n_channels = n_channels = 5
        down_channels = [starting_channels*multi for multi in [2**i for i in range(n_channels)]]
        up_channels = down_channels[::-1]
        out_dim = 1 
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], kernel_size=initial_projection_kernel, padding=initial_projection_padding)

        # Downsample
        # note this is index to n-1 because we are keying fwd 1 for input_dim, output_dim
        self.downs = nn.ModuleList([])
        for i in range(len(down_channels)-1):
            self.downs.append(Block(down_channels[i], down_channels[i+1], time_emb_dim))
        self.pool = nn.MaxPool2d(2, stride=2)

        # Upsample
        self.ups = nn.ModuleList([])
        for i in range(len(up_channels)-1):
            self.ups.append(Block(up_channels[i], up_channels[i+1], time_emb_dim, up_sample=True))

        self.output = nn.Conv2d(up_channels[-1], image_channels, out_dim)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for i, down in enumerate(self.downs):
            x = down(x, t)
            residual_inputs.append(x)
            # if i != self.n_channels:
            #     x = self.pool(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        return self.output(x)

def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t)
    noise_pred = model(x_noisy, t)
    return torch.nn.functional.l1_loss(noise, noise_pred)

@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    
    # Call model (current image - noise prediction)
    model_mean = SQRT_RECIP_ALPHA[t] * (
        x - BETA[t] * model(x, t) / SQRT_ONE_MINUS_ALPHA_BAR[t]
    )
    
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(POSTERIOR_VARIANCE[t]) * noise 

@torch.no_grad()
def sample_plot_image(n_imgs=1, show_n_steps=10):
    # Sample noise
    imgs = torch.randn((n_imgs, IMG_CHANNELS, IMG_SIZE, IMG_SIZE), device=DEVICE)

    stepsize = int(T/show_n_steps)
    plt.figure()
    for col_i, t in enumerate(range(0, T, stepsize)):
        t = torch.full((1,), col_i, device=DEVICE, dtype=torch.long)
        imgs = sample_timestep(imgs, t)
        for row_i, img in enumerate(imgs):
            img = img_tensor_to_pil(img)
            ax = plt.subplot(n_imgs, show_n_steps, row_i*show_n_steps + col_i + 1)
            ax.set_axis_off()
            plt.imshow(img, cmap='gray') # NOTE: matplotlib makes grayscale color by default unless you call out cmap=gray
    plt.show()
    plt.close()


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
    DATASET = 'CIFAR10' # MNIST CIFAR10 CelebA
    IMG_SIZE = 24 # resize img to smaller than original helps with training (MNIST is already 24x24 though)

    # -------------------------------------------------------------------------------------------------------
    # Hyperparameter Tuning
    # -------------------------------------------------------------------------------------------------------

    T = 300 # beta time steps
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
    # uncomment to see the input imgs

    data_train, data_test = load_data(DATASET)
    # NOTE: when the img is rgb the shape is [samples, height, width, channels] else its [samples, height, width]
    IMG_CHANNELS = data_train.dataset.data.shape[-1] if len(data_train.dataset.data.shape) == 4 else 1
    # visualize_input_imgs(dataloader, 3)

    # simluate_forward_diffusion(dataloader, n_imgs=5, show_n_steps=10)

    model = Unet()
    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=0.001)
    epochs = 100 # Try more!

    for epoch in range(epochs):
        for step, batch in enumerate(data_train):
            x, y = batch
            optimizer.zero_grad()

            t = torch.randint(0, T, (BATCH_SIZE,), device=DEVICE).long()
            loss = get_loss(model, x, t)
            loss.backward()
            optimizer.step()
            if step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                sample_plot_image()
