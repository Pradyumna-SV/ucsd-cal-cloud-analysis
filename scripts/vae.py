import torch
from torch import optim
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

class ConvBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels, num_convs):
        super(ConvBlock, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
        layers.append(nn.LeakyReLU(inplace=True))
        in_channels = out_channels
        for _ in range(num_convs-1):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.LeakyReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class ConvBlock2dT(pl.LightningModule):
    def __init__(self, in_channels, out_channels, num_convs):
        super(ConvBlock2dT, self).__init__()
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding = 1))
        layers.append(nn.LeakyReLU(inplace=True))
        in_channels = out_channels
        for _ in range(num_convs-1):
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding = 1))
            layers.append(nn.LeakyReLU(inplace=True))
            in_channels = out_channels
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class VAELightningModule(pl.LightningModule):
    def __init__(self, latent_dim = 512*2*2, lr = 1e-5):
        super(VAELightningModule, self).__init__()
        self.save_hyperparameters()
        self.losses = 0
        self.val_losses = 0

        #Encoder part

        self.blocken1 = ConvBlock(3, 32, 3)
        self.blocken2 = ConvBlock(32, 64, 3)
        self.blocken3 = ConvBlock(64, 128, 3)
        self.blocken4 = ConvBlock(128, 256, 3)
        self.blocken5 = ConvBlock(256, 512, 3)

        self.fcen1 = nn.Linear(512*4*4, latent_dim)
        self.fcen2 = nn.Linear(512*4*4, latent_dim)
        # self.encoder = Encoder(latent_dim)

        self.fc = nn.Linear(latent_dim, 512*4*4)

        #Decoder part

        self.blockde5 = ConvBlock2dT(512, 256, 3)
        self.blockde4 = ConvBlock2dT(256, 128, 3)
        self.blockde3 = ConvBlock2dT(128, 64, 3)
        self.blockde2 = ConvBlock2dT(64, 32, 3)
        self.blockde1 = ConvBlock2dT(32, 3, 3)
        # self.decoder = Decoder(latent_dim)
        self.tanh = nn.Tanh()

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std + 1e-6 
        
    def encoder(self, x):
        ## Encoder
        x = self.blocken1(x)
        x = self.blocken2(x)
        x = self.blocken3(x)
        x = self.blocken4(x)
        x = self.blocken5(x)
        x = x.view(x.size(0), -1)
        mean = self.fcen1(x)
        log_var = self.fcen2(x)
        return mean, log_var

    def decoder(self,x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4)
        x = self.blockde5(x)
        x = self.blockde4(x)
        x = self.blockde3(x)
        x = self.blockde2(x)
        x = self.blockde1(x)
        x = self.tanh(x)
        return x
    
    def transform_data(self, data_in):
    # applies some random flips and rotations to the data
        rand_i = np.random.choice([0, 1, 2, 3, 4])
        if rand_i in [1, 3]:
            # rotate 90 degrees
            data_in = data_in.rot90(rand_i, [2, 3])
        elif rand_i == 2:
            # vert mirror
            data_in = data_in.flip(2)
        elif rand_i == 4:
            # horiz mirror
            data_in = data_in.flip(3)
        # else do nothing, use orig image
        return data_in

    def forward(self, x):
        ## Encoder
        # mean, log_var = self.encoder(self.transform_data(x))
        mean, log_var = self.encoder(x)
        ## Reparameterization
        x = self.reparameterize(mean, log_var)
        ## Decoder
        x = self.decoder(x)
        return x, mean, log_var

    def loss_function(self, recon_x, x, mean, log_var):
        BCE = nn.functional.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        epsilon = 1e-10
        log_var = torch.clamp(log_var, min=-10, max=10)
        
        # Kullback-Leibler divergence
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - (log_var + epsilon).exp()) / x.size(0)
        return BCE + 0.1*KLD
    
    def training_step(self, batch, batch_idx):
        recon_x, mean, log_var = self.forward(batch)
        loss = self.loss_function(recon_x, batch, mean, log_var)
        self.losses += loss.data.item()
        self.log("training_loss_cum", self.losses, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist = True)
        self.log("training_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist = True)
        self.log("Train_epoch_loss", self.losses, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist = True)
        return loss 
    
    def on_train_epoch_end(self):
        self.losses = 0

    def validation_step(self, batch, batch_idx):
        recon_x, mean, log_var = self.forward(batch)
        loss = self.loss_function(recon_x, batch, mean, log_var)
        self.val_losses += loss.data.item()
        self.log("validation_loss_step", self.val_losses, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("Validation_epoch_loss", self.val_losses, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist = True)

    def on_validation_epoch_end(self):
        self.val_losses = 0

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer