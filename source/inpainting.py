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
import utils as utils

# -------------------------------------------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------------------------------------------
os.chdir(os.path.dirname(__file__))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------------------------------------------------------------------
# Custom Diffusion Functions (custom diffusion functions go here)
# -------------------------------------------------------------------------------------------------------

def cold_diffusion_mask():
    # Currently only works with gaussian
    gaussian_mask = torch.ones((T, IMG_CHANNELS, IMG_SIZE, IMG_SIZE))
    # max-1 because variance starts at 1, T-1 because we start nosing from t=1 not t=0
    max_variance = GAUSSIAN_MASKING_PARAMETERS[MASK_MODE]['max_gaussian_variance']
    min_variance = GAUSSIAN_MASKING_PARAMETERS[MASK_MODE]['min_gaussian_variance']
    variance_step_size = (max_variance-min_variance) / (T-1)
    variance = min_variance
    # start 1 so that we dont add noise to t=0
    indices = torch.arange(0, IMG_SIZE, dtype=torch.float32)
    xgrid, ygrid = torch.meshgrid(indices, indices, indexing='ij')
    for t in range(1, T):
        if MASK_MODE == "vertical":
            if DEGRADATION_FUNCTION == "gaussian":
                kernel = torch.exp(-(xgrid ** 2) / (2 * variance))
            elif DEGRADATION_FUNCTION == "sinusoidal":
                kernel = 1 - torch.sin(xgrid * np.pi / IMG_SIZE)
            else:  # testing high frequency sinusoid  # TODO combine sinusoids together and add freq. term
                kernel = 1 - (0.5*torch.sin(xgrid * np.pi/IMG_SIZE*8)+0.5)
        else:
            if DEGRADATION_FUNCTION == "gaussian":
                kernel = torch.exp(-((xgrid-IMG_SIZE/2) ** 2 + (ygrid-IMG_SIZE/2) ** 2) / (2 * variance))
            elif DEGRADATION_FUNCTION == "sinusoidal":  # sinusoidal
                kernel = 1 - torch.sin(xgrid*np.pi / IMG_SIZE) * torch.sin(ygrid*np.pi / IMG_SIZE)
            else:  # testing high frequency sinusoid  # TODO combine sinusoids together and add freq. term
                kernel = 1 - (0.5*torch.sin(xgrid * np.pi / IMG_SIZE*8) * torch.sin(ygrid * np.pi / IMG_SIZE*8)+0.5)

        kernel = 1 - kernel / kernel.max()  # normalize

        for c in range(IMG_CHANNELS):
            gaussian_mask[t, c, :] = kernel*gaussian_mask[t-1, c, :]
        variance += variance_step_size

    return gaussian_mask.to(DEVICE)

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
    n, c, h, w = x_0.shape
    noise = torch.zeros_like(x_0).to(DEVICE)
    for i in range(n):
        if isinstance(t, int):
            t_int = t
        else:
            t_int = t[i].item()
        noise[i] = GAUSSIAN_MASK[t_int]
    output = x_0.to(DEVICE)*noise
    return output, noise


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
    # MASK_MODE:
    # -------------------------------------------------------------------------------------------------------
    MASK_MODE = "radial"  # 'vertical' or 'radial'
    DEGRADATION_FUNCTION = "gaussian"  # 'gaussian', 'sinusoidal' or 'hf_sinusoidal'
    INVERSE_MASK = True  # True or False  # TODO implement an inverse feature
    DATASET = 'CelebA' # MNIST CIFAR10 CelebA
    SAMPLING_METHOD = "naive" # naive algo2
    DIFFUSION_NAME = f'inpainting_{DEGRADATION_FUNCTION}_{MASK_MODE}_{SAMPLING_METHOD}'
    IMG_SIZE = 64 # resize img to smaller than original helps with training (MNIST is already 24x24 though)
    TRAIN = True # True will train a new model and save it in ../trained_model/ otherwise it will try to load one if it exist
    SHOW_PLOTS = False
    # -------------------------------------------------------------------------------------------------------
    # Hyperparameter Tuning
    # -------------------------------------------------------------------------------------------------------
    T = 50 # (for gaussian this is called beta time steps)
    # NOTE: increasing beyond 64 breaks the restoration!!!
    BATCH_SIZE = 32 # batch size to process the imgs, larger the batch the more avging happens for gradient training updates
    LEARNING_RATE = 2e-5
    EPOCHS = 10
    GAUSSIAN_MASKING_PARAMETERS = {
        'vertical':
            {'max_gaussian_variance': IMG_SIZE*2,
             'min_gaussian_variance': 1},
        'radial':
            {'max_gaussian_variance': IMG_SIZE/2 if IMG_SIZE == 24 else IMG_SIZE*2, # the max variance the model should have at time step T, roughtly want IMG_SIZE/2
             'min_gaussian_variance': 1}
    }
    # -------------------------------------------------------------------------------------------------------
    # Start of Process
    # -------------------------------------------------------------------------------------------------------
    os.makedirs(f'../results/{DATASET}/{DIFFUSION_NAME}', exist_ok=True)

    data_train, data_valid = utils.load_data(DATASET, IMG_SIZE, BATCH_SIZE)
    # NOTE: [0] for 0th sample, this returns the x,y as a tuple, we want the img only so again [0], the shape will be [channel, height, width]
    IMG_CHANNELS = data_train.dataset[0][0].shape[0]

    # -------------------------------------------------------------------------------------------------------
    # Diffusion Global Parameters
    # -------------------------------------------------------------------------------------------------------
    # NOTE: Currently Cold Diffusion scheduler is linear, but it could be changed to ease the degredation at a different rate by using BETA/ALPHA
    # Define beta schedule
    # NOTE: T is also the size of beta
    BETA = utils.beta_scheduler(steps=T, end=0.1)
    # NOTE: alpha.shape == beta.shape, but alpha is a slow decrease from 1 to 1-beta_end 
    ALPHA = 1. - BETA

    GAUSSIAN_MASK = cold_diffusion_mask()

    # -------------------------------------------------------------------------------------------------------
    # Visualize Imgs and Fwd Diffusion
    # -------------------------------------------------------------------------------------------------------
    # show sample imgs from dataset
    utils.visualize_input_imgs(data_train, 3, DATASET, DIFFUSION_NAME, SHOW_PLOTS)

    # show what fwd diffusion looks like
    utils.simluate_forward_diffusion(data_train, forward_diffusion_sample, max_time=T, n_imgs=5, show_n_steps=10, 
                                     dataset=DATASET, diffusion_name=DIFFUSION_NAME, show_plots=SHOW_PLOTS)

    # -------------------------------------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------------------------------------
    SAVED_MODEL_FILENAME = f'../results/{DATASET}/{DIFFUSION_NAME}/{DIFFUSION_NAME}_{DATASET}.model'
    if TRAIN:
        model = utils.Unet(IMG_CHANNELS, IMG_SIZE)
        model.to(DEVICE)
        model = utils.train(model, LEARNING_RATE, EPOCHS, BATCH_SIZE, data_train, data_valid, T, DATASET,
          DIFFUSION_NAME, SHOW_PLOTS, sample_timestep, SAVED_MODEL_FILENAME, forward_diffusion_sample)
    elif os.path.exists(SAVED_MODEL_FILENAME):
        model = utils.load_model(SAVED_MODEL_FILENAME, IMG_CHANNELS)
        utils.sample_plot_model_image(forward_diffusion_sample, sample_timestep, data_train, T, 5, 10, f'{EPOCHS-1}', DATASET, DIFFUSION_NAME, SHOW_PLOTS)
    else:
        raise FileNotFoundError('Missing training model')