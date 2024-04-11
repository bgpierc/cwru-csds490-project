# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 12:04:49 2023

@author: jlbraid, nrjost, bgpierc
"""
import time as t
t0 = t.time() #timerimport numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
from termcolor import colored
#from pytorch_ssim import SSIM #don't use pip installed version, is not maintained
from torchvision import transforms


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, dilation=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7, stride=2, dilation=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=7, stride=2, dilation=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=7, stride=2, dilation=1, padding=3),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256 * 25 * 25, latent_dim)
        self.fc_logvar = nn.Linear(256 * 25 * 25, latent_dim)
        print(colored(self.fc_mu, 'green'))
        print(colored(self.fc_logvar, 'blue'))
    
    def forward(self, x):
        x = self.conv(x)
        print(colored(x.size(), 'red'))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * 25 * 25)# * 25 * 25)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=7, stride=2, dilation=1, padding=3, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=7, stride=2, dilation=1, padding=3, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=7, stride=2, dilation=1, padding=3, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=7, stride=2, dilation=1, padding=3, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 256, 25, 25)
        x_deconv = self.deconv(x)
        x_deconv = nn.functional.upsample(x_deconv, size=(400, 400), mode='bilinear', align_corners=False)
        return x_deconv

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar).to(device)
        eps = torch.randn_like(std).to(device)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar, bce_weight, kld_weight, ssim_weight):
    print(colored("Shape of x is", 'magenta'))
    print(colored(x.shape, 'magenta'))
    print(colored(("Shape of recon_x is"), 'cyan'))
    print(colored(recon_x.shape, 'cyan'))
    # print(colored("Dtype of x is", 'light_magenta'))
    # print(colored(x.dtype, 'light_magenta'))
    # print(colored("Dtype of recon_x is", 'light_cyan'))
    # print(colored(recon_x.dtype, 'light_cyan'))
    recon_loss = nn.functional.binary_cross_entropy(recon_x.view(-1, 400*400), x.view(-1, 400*400), reduction='sum') #adapt to size of input array
    # height, width = x.shape[-2], x.shape[-1]
    # recon_x_resized = torch.nn.functional.interpolate(recon_x, size=(height, width), mode="bilinear", align_corners=False)
    ssim_loss = SSIM(window_size=11)
    ssimloss = 1 - ssim_loss(recon_x, x)
    ssimloss = ssimloss.to(device)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld_loss = kld_loss.to(device)
    print("Current BCE loss =%f" % recon_loss)
    print("Current SSIM loss =%f" % ssimloss)
    print("Current KLD loss =%f" % kld_loss)
    total_loss = (
        bce_weight * recon_loss +
        kld_weight * kld_loss +
        ssim_weight * ssimloss
    )
    return total_loss
