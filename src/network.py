import torch
from torch import nn
from torchvision import models

<<<<<<< HEAD

=======
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
# Baseline model
class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)  # Pretrained on resnet, 50 layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, 1)
        self.fc1 = nn.Linear(512 * 4, 256)
        self.fc2 = nn.Linear(512 * 4, 256)

        del self.resnet.fc
        del self.resnet.avgpool

        self.lin_1 = torch.nn.Sequential(
            torch.nn.Linear(256,100),
            torch.nn.BatchNorm1d(100),
            torch.nn.ReLU6(),
        )
        self.lin_2 = torch.nn.Sequential(
            torch.nn.Linear(100, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.avgpool(x)
        rep = torch.flatten(x,1)
        x = self.fc1(rep)

        x = self.lin_1(x)
        x = self.lin_2(x)
        return x


# Encoder for $Q_{Z|X}
class ResnetEncoder(nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.resnet = models.resnet50(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, 1)
        self.fc1 = nn.Linear(512 * 4, latent_dim)
        self.fc2 = nn.Linear(512 * 4, latent_dim)
<<<<<<< HEAD
        del self.resnet.fc
        del self.resnet.avgpool

    # Reparameterization trick using variance for when alpha > 1
    def reparameterize_highalpha(self,mu,var):
        std = var ** 0.5
        # random normal distribution
        eps = torch.randn_like(std, device=self.device)
        return mu + std * eps
=======

        del self.resnet.fc
        del self.resnet.avgpool

    # Reparameterization trick
    def reparametrize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std,device=self.device)
        return mu + std*eps
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.avgpool(x)
<<<<<<< HEAD

        rep = torch.flatten(x,1)
        mu = self.fc1(rep)

        # sigmoid added to keep variance between 0 and 1 to allow for alpha > 1
        var = self.fc2(rep)
        var = torch.sigmoid(var)
        z = self.reparameterize_highalpha(mu,var)
        return z, mu, var
=======
        rep = torch.flatten(x,1)
        mu = self.fc1(rep)
        logvar = self.fc2(rep)

        z = self.reparametrize(mu,logvar)
        return z, mu, logvar
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd


# Decoder for Q_{Y|Z}
class Decoder(nn.Module):
    def __init__(self,latent_dim,output_dim):
        super().__init__()

        self.lin1 = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, output_dim),
        )

    def forward(self,z):
<<<<<<< HEAD
        yhat = self.lin1(z)
        return yhat
=======
        z = self.lin1(z)
        return z
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd


# Decoder for Q_{Y|S,Z}
class FairDecoder(nn.Module):
    def __init__(self,latent_dim,output_dim):
        super().__init__()

        self.lin1 = torch.nn.Sequential(
            torch.nn.Linear(latent_dim+1, output_dim),
        )

    def forward(self, z, s):
        s = s.view(-1, 1)
<<<<<<< HEAD
        yhat_fair = self.lin1(torch.cat((z, s), 1))
        return yhat_fair
=======
        z = self.lin1(torch.cat((z, s), 1))
        return z
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd


# Main RFIB model
class RFIB(nn.Module):
    def __init__(self,latent_dim,output_dim=1):
        super().__init__()

        self.encoder = ResnetEncoder(latent_dim)
        self.decoder = Decoder(latent_dim,output_dim)
        self.fair_decoder = FairDecoder(latent_dim,output_dim)

<<<<<<< HEAD
    def forward(self, x, s):
        z, mu, logvar = self.encoder(x)
        yhat = self.decoder(z)
        yhat_fair = self.fair_decoder(z, s)
=======
    def forward(self, x, a):
        z, mu, logvar = self.encoder(x)
        yhat = self.decoder(z)
        yhat_fair = self.fair_decoder(z, a)
>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd

        return yhat, yhat_fair, mu, logvar

    def getz(self,x):
        return self.encoder(x)


<<<<<<< HEAD
=======


>>>>>>> af517cb84769923823d05b0472dfa76d89aafadd
