# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 12:04:49 2023

@author: jlbraid, nrjost, bgpierc
"""
import time as t
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
        
        hidden = 16384
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
    
    def forward(self, x):
        x = self.conv(x)
        #print(colored(x.size(), 'red'))
        x = x.view(x.size(0), -1)
        #print(x.shape)
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
        x_deconv = nn.functional.interpolate(x_deconv, size=(128, 128), mode='bilinear', align_corners=False)
        return x_deconv

class VAE(nn.Module):
    def __init__(self, latent_dim, device):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.device = device
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar).to(self.device)
        eps = torch.randn_like(std).to(self.device)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z

def VAE_loss(recon_x, x, mu, logvar):
    batch, chan, h, w = x.shape
    recon_loss = nn.functional.binary_cross_entropy(recon_x.view(-1, h*w), x.view(-1, h*w), reduction='sum')
    #recon_loss = torch.nn.functional.mse_loss(x, recon_x)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss, recon_loss, kld_loss
    
