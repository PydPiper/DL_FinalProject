# -------------------------------------------------------------------------------------------------------
# Import standard libs
# -------------------------------------------------------------------------------------------------------
import os
import random
import time
# -------------------------------------------------------------------------------------------------------
# Import pip libs
# -------------------------------------------------------------------------------------------------------
import torch
import torchvision
from torch import nn
from torch.optim import Adam
import ignite
from pytorch_fid.inception import InceptionV3
from matplotlib import pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import PIL


# -------------------------------------------------------------------------------------------------------
# Ploting
# -------------------------------------------------------------------------------------------------------
def visualize_input_imgs(dataloader, n_imgs=1, dataset='MNIST', diffusion_name='noise', show_plots=False):
    iter_dataloader = iter(dataloader)
    fig = plt.figure()
    for i in range(n_imgs):
        # NOTE: calling next() returns a [tensor(x), tensor(y)]
        # where x.shape = [batch size, channel, height, width]
        #       y.shape = [batch size]
        imgs = next(iter_dataloader)[0][0:n_imgs]
        for img in imgs:
            img = img_tensor_to_pil(img)
            ax = fig.add_subplot(1, n_imgs, i + 1)
            ax.set_axis_off()
            ax.imshow(img, cmap='gray')
    fig.savefig(f'../results/{dataset}/{diffusion_name}/sample.png')
    if show_plots:
        fig.show()
    plt.close()

def img_tensor_to_pil(img):
    if len(img.shape) > 5:
        raise ValueError(f'Input img.shape={img.shape}, img.shape must be [channel, height, width]')
    elif len(img.shape) == 4:
        # assume img is in the form [n, ch, height, width] --> [n, height, width, ch]
        sequence = (0, 2, 3, 1)
    else:
        # assume img is in the form [ch, height, width] --> [height, width, ch]
        sequence = (1, 2, 0)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda data: data.permute(sequence)), # note this is needed when using plt.imshow
        torchvision.transforms.Lambda(lambda data: (data / torch.abs(data).max() + 1) / 2), # recover all -1 to 0 values
        torchvision.transforms.Lambda(lambda data: (data.to('cpu') * 255.).numpy().astype(np.uint64)), # bring it back from 0-1 to RGB 0-255 scale
        # torchvision.transforms.ToPILImage(), # note this is only needed when using plt.imshow
    ])
    return transforms(img)

@torch.no_grad()
def sample_plot_model_image(forward_diffusion_sample, sample_timestep, data, max_t=300, n_imgs=1, show_n_steps=10, epoch='0',
                            dataset='MNIST', diffusion_name='noise', show_plots=False, validation=False):
    # Create Random Sample 
    # imgs = torch.randn((n_imgs, img_channels, img_size, img_size), device=DEVICE)

    # get imgs out of dataset to do simulated fwd pass
    imgs_0 = data.dataset[0][0].unsqueeze(0).to(DEVICE)
    for i in range(1, n_imgs):
        img = data.dataset[i][0].unsqueeze(0).to(DEVICE)
        imgs_0 = torch.cat((imgs_0, img), dim=0)

    stepsize = int(max_t/show_n_steps)
    fig = plt.figure(figsize=(24,6))
    col_i = 0

    for col_i, t in enumerate(range(0, max_t, stepsize)):
        imgs, noises = forward_diffusion_sample(imgs_0, t)
        for row_i, img in enumerate(imgs):
            img = img_tensor_to_pil(img)
            ax = fig.add_subplot(n_imgs, show_n_steps*2, row_i*show_n_steps*2 + col_i + 1)
            ax.set_axis_off()
            ax.imshow(img, cmap='gray') # NOTE: matplotlib makes grayscale color by default unless you call out cmap=gray


    # Encode actual sample (note [0] is 1st sample, then [0] is for imgs values)
    imgs, noise = forward_diffusion_sample(data.dataset[0][0].unsqueeze(0).to(DEVICE), max_t-1)
    for i in range(1, n_imgs):
        img, noise = forward_diffusion_sample(data.dataset[i][0].unsqueeze(0).to(DEVICE), max_t-1)
        imgs = torch.cat((imgs, img), dim=0)

    rmse_noise = calc_error_rmse(imgs_0, imgs)
    fid_noise = calc_error_fid(imgs_0, imgs)
    ssim_noise = calc_error_ssim(imgs_0, imgs)


    col_i = 0
    for t in range(0, max_t)[::-1]:
        t = torch.full((1,), t, device=DEVICE, dtype=torch.long)
        imgs = sample_timestep(imgs, t)
        if t % stepsize == 0:
            for row_i, img in enumerate(imgs):
                img = img_tensor_to_pil(img)
                ax = fig.add_subplot(n_imgs, show_n_steps*2, show_n_steps + row_i*show_n_steps*2 + col_i + 1)
                ax.set_axis_off()
                ax.imshow(img, cmap='gray') # NOTE: matplotlib makes grayscale color by default unless you call out cmap=gray
            col_i += 1

    # NOTE: these are only computed for the set of n_imgs
    rmse = calc_error_rmse(imgs_0, imgs)
    fid = calc_error_fid(imgs_0, imgs)
    ssim = calc_error_ssim(imgs_0, imgs)

    flag = 'valid' if validation else 'train' 
    error_text = f'{flag} noise/diffused => rsme: {rmse_noise:4f}/{rmse:4f} | fid: {fid_noise:4f}/{fid:4f}, ssim: {ssim_noise:4f}/{ssim:4f}'
    with open(f'../results/{dataset}/{diffusion_name}/{flag}_{diffusion_name}_error.txt', 'w') as f:
        f.write(error_text)
    print(error_text)

    fig.savefig(f'../results/{dataset}/{diffusion_name}/{flag}_{diffusion_name}_{epoch}.png')
    if show_plots:
        fig.show()
    plt.close()

def plot_learning_curve(train_loss, valid_loss, dataset='MNIST', diffusion_name='noise', show_plots=False):

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f'{diffusion_name.title()} - Learning Curve')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    x = np.arange(1, len(train_loss)+1)
    ax.plot(x, train_loss, label='train')
    ax.plot(x, valid_loss, label='valid')
    ax.legend()
    ax.grid()
    fig.savefig(f'../results/{dataset}/{diffusion_name}/learning_curve.png')
    if show_plots:
        fig.show()
    plt.close()


# -------------------------------------------------------------------------------------------------------
# Data load
# -------------------------------------------------------------------------------------------------------
def load_data(dataset='MNIST', img_size=32, batch_size=128):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(), # scales from RGB 0-255 to 0-1
        torchvision.transforms.Lambda(lambda data: (data*2) - 1) # shift data to be -1 to 1
    ])
    if dataset != 'CelebA':
        data_train = getattr(torchvision.datasets, dataset)('../data/', download=True, train=True, transform=transforms)
        # data_train = torch.utils.data.Subset(data_train, torch.arange(0,10e3))
        data_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True)

        data_valid = getattr(torchvision.datasets, dataset)('../data/', download=True, train=False, transform=transforms)
        data_valid = torch.utils.data.DataLoader(data_valid, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        # NOTE: for some reason torch CalebA doesnt take "train" as an arg instead its uses split
        data_train = getattr(torchvision.datasets, dataset)('../data/', download=True, split='train', transform=transforms)
        # data_train = torch.utils.data.Subset(data_train, torch.arange(0,10e3))
        data_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True)

        data_valid = getattr(torchvision.datasets, dataset)('../data/', download=True, split='valid', transform=transforms)
        data_valid = torch.utils.data.DataLoader(data_valid, batch_size=batch_size, shuffle=True, drop_last=True)
    return data_train, data_valid

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


def simluate_forward_diffusion(dataloader, forward_diffusion_sample, max_time=300, n_imgs=1, show_n_steps=5, dataset='MNIST', diffusion_name='noise', show_plots=False):

    iter_dataloader = iter(dataloader)
    # key 0 means that you are pulling out the x values (ie. imgs), key 1 would be the ground truth values "y"
    imgs = next(iter_dataloader)[0][:n_imgs]

    stepsize = int(max_time/show_n_steps)
    fig = plt.figure()
    for col_i, t in enumerate(range(0, max_time, stepsize)):
        # NOTE: calling next() returns a [tensor(x), tensor(y)]
        # where x.shape = [batch size, channel, height, width]
        #       y.shape = [batch size]
        imgs, noises = forward_diffusion_sample(imgs, t)
        for row_i, img in enumerate(imgs):
            img = img_tensor_to_pil(img)
            ax = fig.add_subplot(n_imgs, show_n_steps, row_i*show_n_steps + col_i + 1)
            ax.set_axis_off()
            ax.imshow(img, cmap='gray') # NOTE: matplotlib makes grayscale color by default unless you call out cmap=gray
    fig.savefig(f'../results/{dataset}/{diffusion_name}/fwd_diff.png')
    if show_plots:
        fig.show()
    plt.close()

# -------------------------------------------------------------------------------------------------------
# Unet Model
# -------------------------------------------------------------------------------------------------------
def load_model(filename, img_channels, img_size):
    model = Unet(img_channels, img_size)
    # load model parameters
    model.load_state_dict(torch.load(filename))
    # must used model.eval() when using BatchNorm or Dropout
    model = model.to(DEVICE)
    # model.eval() # TODO: for some reason, it appears to be backward for us than everything that i read online, the only way to use TRAIN=False is to comment this out and let the model compute batch norm
    return model

class Encode(nn.Module):
    def __init__(self, dim_in_ch, dim_out_ch, dim_time_emb):
        super().__init__()
        self.linear =  nn.Linear(dim_time_emb, dim_out_ch)
        # (h + 2*  padding - dilation*(kernel_size - 1) - 1)/stride + 1
        self.conv1 = nn.Conv2d(dim_in_ch, dim_out_ch, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(dim_out_ch, dim_out_ch, kernel_size=3, padding='same')
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
        return h


class Decode(nn.Module):
    def __init__(self, dim_in_ch, dim_out_ch, dim_time_emb, up_kernel, up_stride, up_padding):
        super().__init__()
        self.linear =  nn.Linear(dim_time_emb, dim_out_ch)
        # self.deconv = nn.Upsample(scale_factor=2, mode='nearest')
        # (H −1)×stride−2×padding+dilation×(kernel_size−1)+output_padding+1
        self.deconv = nn.ConvTranspose2d(dim_in_ch, dim_in_ch, kernel_size=up_kernel, stride=up_stride, padding=up_padding)
        self.conv1 = nn.Conv2d(dim_in_ch*2, dim_out_ch, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(dim_out_ch, dim_out_ch, kernel_size=3, padding='same')
        self.norm = nn.BatchNorm2d(dim_out_ch)
        self.relu  = nn.ReLU()

    def forward(self, x, x_encode, t):
        # Upsample
        h = self.deconv(x)
        # cat input with encode input
        # print(x.shape, h.shape, x_encode.shape)
        x = torch.cat((h, x_encode), dim=1)           
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
    def __init__(self, img_channels=3, img_size=24):
        super().__init__()
        image_channels = img_channels
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
            self.downs.append(Encode(down_channels[i], down_channels[i+1], time_emb_dim))
        self.pool = nn.MaxPool2d(2)

        # Upsample
        self.ups = nn.ModuleList([])
        up_params = {
            24: [
                # k, s, p
                [3, 1, 0],
                [4, 2, 1],
                [4, 2, 1],
                [4, 2, 1],
                ],
            64: [
                [3, 3, 2],
                [4, 2, 1],
                [4, 2, 1],
                [4, 2, 1],
            ]
        }[img_size]
        for i in range(len(up_channels)-1):
            self.ups.append(Decode(up_channels[i], up_channels[i+1], time_emb_dim, *up_params[i]))

        self.output = nn.Conv2d(up_channels[-1], image_channels, out_dim)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        encode_outputs = []
        for i, down in enumerate(self.downs):
            x = down(x, t)
            if i != self.n_channels-1:
                encode_outputs.append(x) 
                # for every down the img hxw is halved so 24 > 12 > 6 > 3 > 1 or 64 > 32 > 16 > 8 > 4
                x = self.pool(x)

        for up in self.ups:
            encode_output = encode_outputs.pop()
            x = up(x, encode_output, t)
        return self.output(x)

def get_loss(model, x_0, t, forward_diffusion_sample, diffusion_name):
    x_noisy, noise = forward_diffusion_sample(x_0, t)
    if diffusion_name == 'noise':
        noise_pred = model(x_noisy, t)
        loss = torch.nn.functional.l1_loss(noise, noise_pred)
    else:
        img_prev_pred = model(x_noisy, t)
        loss = torch.nn.functional.l1_loss(x_0.to(DEVICE), img_prev_pred)
    return loss

def train(model, learning_rate, epochs, batch_size, data_train, data_valid, max_t, 
          dataset, diffusion_name, show_plots, sample_timestep, saved_model_filename, forward_diffusion_sample):

    optimizer = Adam(model.parameters(), lr=learning_rate)

    log_train_loss = []
    log_valid_loss = []
    running_batch_train_loss = 0
    start_time = time.time()
    for epoch in range(epochs):
        for step, batch in enumerate(data_train):
            x, y = batch
            optimizer.zero_grad()

            t = torch.randint(0, max_t, (batch_size,), device=DEVICE).long()
            loss = get_loss(model, x, t, forward_diffusion_sample, diffusion_name)
            running_batch_train_loss += loss.to('cpu').item()

            loss.backward()
            optimizer.step()
            if step == 0:
                # on 1st epoch there is only 1 loss sample 
                avg_train_loss = running_batch_train_loss / len(data_train) if epoch != 0 else running_batch_train_loss / 1
                log_train_loss.append(avg_train_loss)
                running_batch_train_loss = 0

                # run validation set
                running_batch_valid_loss = 0
                for valid_step, valid_batch in enumerate(data_valid):
                    x_valid, y_valid = valid_batch
                    loss = get_loss(model, x_valid, t, forward_diffusion_sample, diffusion_name)
                    running_batch_valid_loss += loss.to('cpu').item()
                avg_valid_loss = running_batch_valid_loss / len(data_valid)
                log_valid_loss.append(avg_valid_loss)
                
                print(f"Epoch {epoch:03d} | step {step:03d} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f} | Time: {time.time()-start_time:.2f}")
                if epoch % 5 == 0:
                    sample_plot_model_image(forward_diffusion_sample, sample_timestep, data_train, max_t, 5, 10, epoch, dataset, diffusion_name, show_plots)
                    sample_plot_model_image(forward_diffusion_sample, sample_timestep, data_valid, max_t, 5, 10, epoch, dataset, diffusion_name, show_plots, True)
    
    sample_plot_model_image(forward_diffusion_sample, sample_timestep, data_train, max_t, 5, 10, epoch, dataset, diffusion_name, show_plots)
    sample_plot_model_image(forward_diffusion_sample, sample_timestep, data_valid, max_t, 5, 10, epoch, dataset, diffusion_name, show_plots, True)
    plot_learning_curve(log_train_loss, log_valid_loss, dataset, diffusion_name, show_plots)

    # save the model (only model parameters)
    torch.save(model.state_dict(), saved_model_filename)
    
    return model


def calc_error_rmse(img1, img2):
    """Calculate root mean squared error between 2 batches of imgs

    :param img1: batch of img1 imgs
    :type img1: array
    :param img2: batch of img2 imgs
    :type img2: array
    :return: array of rmse
    :rtype: array
    """
    return torch.sqrt(torch.mean((img1 - img2)**2))

def calc_error_fid(img1:torch.Tensor, img2:torch.Tensor, implementation='pytorch_fid'):
    """https://pytorch.org/ignite/generated/ignite.metrics.FID.html
    https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/#evaluation-metrics
    https://github.com/pytorch/ignite/issues/2423
    https://github.com/mseitzer/pytorch-fid

    :param img1: _description_
    :type img1: _type_
    :param img2: _description_
    :type img2: _type_
    :return: _description_
    :rtype: _type_
    """

    # condition input imgs if they are not 3 channel that FID is expecting
    if img1.shape[1] == 1:
        b, c, h, w = img1.shape
        img1 = img1.expand(b, 3, h, w)
        img2 = img2.expand(b, 3, h, w)

    def interpolate(batch, fn):
        arr = []
        for img in batch:
            pil_img = torchvision.transforms.ToPILImage()(img)
            resized_img = pil_img.resize((299,299), PIL.Image.BILINEAR)
            resized_img = torchvision.transforms.ToTensor()(resized_img)
            # torchvision.utils.save_image(resized_img, fn)
            arr.append(resized_img)
        return torch.stack(arr)
    img1 = interpolate(img1, 'test.png')
    img2 = interpolate(img2, 'test2.png')

    def eval_step(engine, batch):
        return batch

    default_evaluator = ignite.ignite.engine.Engine(eval_step)

    if implementation == 'ignite':
        metric = ignite.ignite.metrics.FID()
        metric.attach(default_evaluator, "fid")
        state = default_evaluator.run([[img1, img2]])
        return state.metrics["fid"]
    else:
        # wrapper class as feature_extractor
        class WrapperInceptionV3(nn.Module):

            def __init__(self, fid_incv3):
                super().__init__()
                self.fid_incv3 = fid_incv3

            @torch.no_grad()
            def forward(self, x):
                y = self.fid_incv3(x)
                y = y[0]
                y = y[:, :, 0, 0]
                return y

        # use cpu rather than cuda to get comparable results
        device = "cpu"

        # pytorch_fid model
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).to(device)

        # wrapper model to pytorch_fid model
        wrapper_model = WrapperInceptionV3(model)
        wrapper_model.eval()

        # comparable metric
        metric = ignite.ignite.metrics.FID(num_features=dims, feature_extractor=wrapper_model, device=DEVICE)
        metric.attach(default_evaluator, "fid")
        state = default_evaluator.run([[img1, img2]])
        return state.metrics["fid"]


def calc_error_ssim(img1, img2):
    """Calcuate Structural Similarity Index Measure (SSIM) is the perceived quality of digital pictures

    https://pytorch.org/ignite/generated/ignite.metrics.SSIM.html#ssim
    
    https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity
    https://scikit-image.org/docs/stable/auto_examples/transform/plot_ssim.html

    :param img1: _description_
    :type img1: _type_
    :param img2: _description_
    :type img2: _type_
    """
    
    def eval_step(engine, batch):
        return batch

    default_evaluator = ignite.ignite.engine.Engine(eval_step)
    metric = ignite.ignite.metrics.SSIM(data_range=1.0, device=DEVICE)
    metric.attach(default_evaluator, 'ssim')
    state = default_evaluator.run([[img1, img2]])
    rv = state.metrics['ssim']

    return rv