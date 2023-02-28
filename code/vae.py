import torch
import torch.nn as nn

l1_loss = torch.nn.L1Loss()


class Block(nn.Module):
    """
    Class for the basic convolutional building block
    """

    def __init__(self, in_ch, out_ch):
        """
        Constructor.
        :param in_ch: number of input channels to the block
        :param out_ch: number of output channels of the block
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu =  # TODO  # leaky ReLU
        self.bn1 = # TODO   # batch normalisation
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = # TODO 

    def forward(self, x):
        """
        Returns the output of a forward pass of the block
        :param x: the input tensor
        :return: the output tensor of the block
        """
        # a block consists of two convolutional layers
        # with ReLU activations
        # use batch normalisation

        # TODO
        return x


class Encoder(nn.Module):
    """
    Class for the encoder part of the VAE.
    """

    def __init__(self, spatial_size=[64, 64], z_dim=256, chs=(1, 64, 128, 256)):
        """
        Constructor.
        :param chs: tuple giving the number of input channels of each block in the encoder
        """
        super().__init__()
        # convolutional blocks
        self.enc_blocks = nn.ModuleList(
            # TODO
        )
        # max pooling
        self.pool = # TODO
        # height and width of images at lowest resolution level
        _h, _w = # TODO

        # flattening
        self.out = nn.Sequential(nn.Flatten(1), nn.Linear(chs[-1] * _h * _w, 2 * z_dim))

    def forward(self, x):
        """
        Returns the list of the outputs of all the blocks in the encoder
        :param x: input image tensor
        """

        for block in self.enc_blocks:
            # TODO: conv block           
            # TODO: pooling 
        # TODO: output layer          
        return torch.chunk(x, 2, dim=1)  # 2 chunks - 1 each for mu and logvar


class Generator(nn.Module):
    """
    Class for the generator part of the GAN.
    """

    def __init__(self, z_dim=256, chs=(256, 128, 64, 32), h=8, w=8):
        """
        Constructor.
        :param chs: tuple giving the number of input channels of each block in the decoder
        """
        super().__init__()
        self.chs = chs
        self.h = h  # height of image at lowest resolution level
        self.w = w  # width of image at lowest resolution level
        self.z_dim = z_dim  # dimension of latent space
        self.proj_z = nn.Linear(
            self.z_dim, self.chs[0] * self.h * self.w
        )  # fully connected layer on latent space
        self.reshape = lambda x: torch.reshape(
            x, (-1, self.chs[0], self.h, self.w)
        )  # reshaping

        self.upconvs = nn.ModuleList(
            # TODO: transposed convolution            
        )

        self.dec_blocks = nn.ModuleList(
            # TODO: conv block           
        )
        self.proj_o = nn.Sequential(
            # TODO         
        )  # output layer with Tanh activation

    def forward(self, z):
        """
        Returns the output of the decoder part of the VAE
        :param x: input tensor to the generator
        """
        x = # TODO: fully connected layer
        x = # TODO: reshape to image dimensions
        for i in range(len(self.chs) - 1):
            # TODO: transposed convolution
            # TODO: convolutional block
        x = # TODO: output layer
        return x


class VAE(nn.Module):
    """
    Class for the VAE
    """

    def __init__(
        self,
        enc_chs=(1, 64, 128, 256),
        dec_chs=(256, 128, 64, 32),
    ):
        """
        Constructor.
        :param enc_chs: tuple giving the number of input channels of each block in the encoder
        :param dec_chs: tuple giving the number of input channels of each block in the encoder
        """
        super().__init__()
        self.encoder = Encoder()
        self.generator = Generator()

    def forward(self, x):
        """
        Returns the output of a forward pass of the vae
        That is, both the reconstruction and mean + logvar
        :param x: the input tensor to the encoder
        """
        mu, logvar = self.encoder(x)
        latent_z = sample_z(mu, logvar)

        return self.generator(latent_z), mu, logvar


def get_noise(n_samples, z_dim, device="cpu"):
    """
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    """
    return torch.randn(n_samples, z_dim, device=device)


def sample_z(mu, logvar):
    """
    Samples noise vector with reparameterization trick.
    """
    eps = torch.randn(mu.size(), device=mu.device).to(mu.dtype)
    return (logvar / 2).exp() * eps + mu


def kld_loss(mu, logvar):
    """
    Returns KLD loss
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def vae_loss(inputs, recons, mu, logvar):
    """
    Returns VAE loss, sum of reconstruction and KLD loss
    """
    return l1_loss(inputs, recons) + kld_loss(mu, logvar)
