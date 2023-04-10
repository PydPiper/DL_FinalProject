import torch
import torchvision
from torch import nn
from torch.optim import Adam
from matplotlib import pyplot as plt
import os
import numpy as np
import random

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


def load_data(dataset='MNIST'):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
        torchvision.transforms.ToTensor(), # scales from RGB 0-255 to 0-1
        torchvision.transforms.Lambda(lambda data: (data*2) - 1) # shift data to be -1 to 1
    ])
    data_train = getattr(torchvision.datasets, dataset)('../data/', download=True, train=True, transform=transforms)
    data_test = getattr(torchvision.datasets, dataset)('../data/', download=True, transform=transforms)
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

    # t can come in for n_batch size, need to reshape to multiply
    if not isinstance(t, int):
        t = t.reshape((t.shape[0],1,1,1))

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
            plt.imshow(img, cmap='gray') # NOTE: matplotlib makes grayscale color by default unless you call out cmap=gray
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
            self.transform = nn.ConvTranspose2d(dim_out_ch, dim_out_ch, kernel_size=4, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(dim_in_ch, dim_out_ch, kernel_size=3, padding=1)
            self.transform = nn.Conv2d(dim_out_ch, dim_out_ch, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(dim_out_ch, dim_out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(dim_out_ch)
        self.norm2 = nn.BatchNorm2d(dim_out_ch)
        self.relu  = nn.ReLU()

    def forward(self, x, t):
        # First conv
        h = self.norm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.linear(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.norm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = IMG_CHANNELS
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 1 
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], kernel_size=3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up_sample=True) \
                    for i in range(len(up_channels)-1)])

        self.output = nn.Conv2d(up_channels[-1], 3, out_dim)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
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
def sample_plot_image():
    # Sample noise
    img = torch.randn((1, 3, IMG_SIZE, IMG_SIZE), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        if i % stepsize == 0:
            plt.subplot(1, num_images, i/stepsize+1)
            img = img_tensor_to_pil(img.detach().cpu())
            plt.imshow(img)
    plt.show()    



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
    ALPHA_BAR_PREV = torch.cat((torch.tensor([1.]).to(device), ALPHA_BAR[:-1]))
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

    dataloader = load_data(DATASET)
    # NOTE: when the img is rgb the shape is [samples, height, width, channels] else its [samples, height, width]
    IMG_CHANNELS = dataloader.dataset.datasets[0].data.shape[-1] if len(dataloader.dataset.datasets[0].data.shape) == 4 else 1
    visualize_input_imgs(dataloader, 3)

    simluate_forward_diffusion(dataloader, n_imgs=5, show_n_steps=10)

    # model = SimpleUnet()
    # model.to(device)
    # optimizer = Adam(model.parameters(), lr=0.001)
    # epochs = 100 # Try more!

    # for epoch in range(epochs):
    #     for step, batch in enumerate(dataloader):
    #         optimizer.zero_grad()

    #         t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
    #         loss = get_loss(model, batch[0], t)
    #         loss.backward()
    #         optimizer.step()

    #         if epoch % 5 == 0 and step == 0:
    #             print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
    #             sample_plot_image()
