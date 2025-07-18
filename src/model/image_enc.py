import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, input_size = 256, rep_dim = 32, channels = 3):
        super().__init__()

        assert(rep_dim % 2 == 0 and rep_dim >= 4)
        self.rep_dim =  rep_dim

        self.scale_factor = 256 // input_size
        assert(self.scale_factor in [1,2,4])

        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.conv1 = nn.Conv2d(channels, 32, 5, bias=False, padding=2, stride=2)
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2, stride=2)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2, stride=2)
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)

        self.conv4 = nn.Conv2d(128, 128, 5, bias=False, padding=2, stride=2)
        nn.init.xavier_uniform_(self.conv4.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)

        self.fc1 = nn.Linear(128 * 1 * 1, self.rep_dim, bias=False)
        self.bn5 = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)

    def forward(self, x):
        enc = self.conv1(x)
        enc = self.pool(F.leaky_relu(self.bn1(enc)))
        # print(enc.shape) # torch.Size([6, 32, 64, 64])

        enc = self.conv2(enc)
        enc = self.pool(F.leaky_relu(self.bn2(enc)))
        # print(enc.shape) # torch.Size([6, 64, 16, 16])

        enc = self.conv3(enc)
        enc = self.pool(F.leaky_relu(self.bn3(enc)))
        # print(enc.shape) # torch.Size([6, 128, 4, 4])

        enc = self.conv4(enc)
        # enc = self.pool(F.leaky_relu(self.bn4(enc)))
        if self.scale_factor not in [2, 4]:
            enc = self.pool(F.leaky_relu(self.bn4(enc)))
        else:
            enc = F.leaky_relu(self.bn4(enc))
        # print(enc.shape) # torch.Size([6, 128, 1, 1])

        enc = enc.view(enc.size(0), -1)
        # print(enc.shape) # torch.Size([6, 128])

        enc = self.bn5(self.fc1(enc))
        # print(enc.shape) # torch.Size([6, 32])
        return enc


class Decoder(nn.Module):

    def __init__(self, input_size = 256, rep_dim = 32, channels = 3):

        super().__init__()

        assert(rep_dim % 2 == 0 and rep_dim >= 4)
        self.rep_dim =  rep_dim

        self.scale_factor = 256 // input_size
        assert(self.scale_factor in [1,2,4])

        if self.scale_factor in [2, 4]:
            self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (1 * 1)), 64, 5, bias=False, padding=2)
            nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
            self.bn6 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        else:
            self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (2 * 2)), 64, 5, bias=False, padding=2)

        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn6 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.deconv2 = nn.ConvTranspose2d(64, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn7 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn8 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.deconv4 = nn.ConvTranspose2d(32, channels, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x: torch.Tensor):
        if self.scale_factor in [2, 4]:
            dec = x.view(x.size(0), int(self.rep_dim / (1 * 1)), 1, 1)
        else:
            dec = x.view(x.size(0), int(self.rep_dim / (2 * 2)), 2, 2)

        # print(dec.shape)
        dec = F.leaky_relu(dec)

        dec = self.deconv1(dec)
        dec = F.interpolate(F.leaky_relu(self.bn6(dec)), scale_factor=2)
        # print(dec.shape) # torch.Size([6, 64, 4, 4])

        dec = self.deconv2(dec)
        dec = F.interpolate(F.leaky_relu(self.bn7(dec)), scale_factor=4)
        # print(dec.shape) # torch.Size([6, 32, 64, 64])

        dec = self.deconv3(dec)
        dec = F.interpolate(F.leaky_relu(self.bn8(dec)), scale_factor=4)
        # print(dec.shape) # torch.Size([64, 3, 256, 256])

        if self.scale_factor == 4:
            dec = F.interpolate(self.deconv4(dec), scale_factor=2)
            # print(dec.shape) # torch.Size([6, 3, 128, 128])
        else:
            dec = F.interpolate(self.deconv4(dec), scale_factor=4)
            # print(dec.shape) # torch.Size([6, 3, 128, 128])

        dec = torch.sigmoid(dec)

        return dec


class Autoencoder(nn.Module):

    def __init__(self, input_size = 256, rep_dim = 32, channels = 3):

        super().__init__()

        assert(rep_dim % 2 == 0 and rep_dim >= 4)
        self.rep_dim =  rep_dim

        self.scale_factor = 256 // input_size
        assert(self.scale_factor in [1,2,4])

        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.conv1 = nn.Conv2d(channels, 32, 5, bias=False, padding=2, stride=2)
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2, stride=2)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2, stride=2)
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)

        self.conv4 = nn.Conv2d(128, 128, 5, bias=False, padding=2, stride=2)
        nn.init.xavier_uniform_(self.conv4.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)

        self.fc1 = nn.Linear(128 * 1 * 1, self.rep_dim, bias=False)
        self.bn5 = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)

        # Decoder
        if self.scale_factor in [2, 4]:
            self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (1 * 1)), 64, 5, bias=False, padding=2)
            nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
            self.bn6 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        else:
            self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (2 * 2)), 64, 5, bias=False, padding=2)

        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn6 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.deconv2 = nn.ConvTranspose2d(64, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn7 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn8 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.deconv4 = nn.ConvTranspose2d(32, channels, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        # print(x.shape) # torch.Size([6, 3, 256, 256])

        enc = self.conv1(x)
        enc = self.pool(F.leaky_relu(self.bn1(enc)))
        # print(enc.shape) # torch.Size([6, 32, 64, 64])

        enc = self.conv2(enc)
        enc = self.pool(F.leaky_relu(self.bn2(enc)))
        # print(enc.shape) # torch.Size([6, 64, 16, 16])

        enc = self.conv3(enc)
        enc = self.pool(F.leaky_relu(self.bn3(enc)))
        # print(enc.shape) # torch.Size([6, 128, 4, 4])

        enc = self.conv4(enc)
        if self.scale_factor not in [2, 4]:
            enc = self.pool(F.leaky_relu(self.bn4(enc)))
        else:
            enc = F.leaky_relu(self.bn4(enc))
        # print(enc.shape) # torch.Size([6, 128, 1, 1])

        enc = enc.view(enc.size(0), -1)
        # print(enc.shape) # torch.Size([6, 128])

        enc = self.bn5(self.fc1(enc))
        # print(enc.shape) # torch.Size([6, 32])
        # return enc

        # dec = enc.view(enc.size(0), int(self.rep_dim / (2 * 2)), 2, 2)
        if self.scale_factor in [2, 4]:
            dec = enc.view(enc.size(0), int(self.rep_dim / (1 * 1)), 1, 1)
        else:
            dec = enc.view(enc.size(0), int(self.rep_dim / (2 * 2)), 2, 2)

        # print(dec.shape) # torch.Size([6, 8, 2, 2])
        dec = F.leaky_relu(dec)

        dec = self.deconv1(dec)
        dec = F.interpolate(F.leaky_relu(self.bn6(dec)), scale_factor=2)
        # print(dec.shape) # torch.Size([6, 64, 4, 4])

        dec = self.deconv2(dec)
        dec = F.interpolate(F.leaky_relu(self.bn7(dec)), scale_factor=4)
        # print(dec.shape) # torch.Size([6, 32, 64, 64])

        dec = self.deconv3(dec)
        dec = F.interpolate(F.leaky_relu(self.bn8(dec)), scale_factor=4)
        # print(dec.shape) # torch.Size([64, 3, 256, 256])

        if self.scale_factor == 4:
            dec = F.interpolate(self.deconv4(dec), scale_factor=2)
            # print(dec.shape) # torch.Size([6, 3, 128, 128])
        else:
            dec = F.interpolate(self.deconv4(dec), scale_factor=4)
            # print(dec.shape) # torch.Size([6, 3, 128, 128])

        dec = torch.sigmoid(dec)
        return dec

