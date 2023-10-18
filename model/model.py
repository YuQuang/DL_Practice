import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.conv_block = nn.Sequential(
            # 1 x 256 x 256 -> 64 x 128 x 128
            nn.Conv2d(1, 64, (6, 6), (2, 2), (2, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),

            # 64 x 128 x 128 -> 128 x 64 x 64
            nn.Conv2d(64, 128, (6, 6), (2, 2), (2, 2), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            # 128 x 64 x 64 -> 256 x 32 x 32
            nn.Conv2d(128, 256, (6, 6), (2, 2), (2, 2), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            # 256 x 32 x 32 -> 512 x 16 x 16
            nn.Conv2d(256, 512, (6, 6), (2, 2), (2, 2), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            # 512 x 16 x 16 -> 1024 x 8 x 8
            nn.Conv2d(512, 1024, (6, 6), (2, 2), (2, 2), bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, True),
            
            nn.Flatten(),
            nn.Linear(1024 * 8 * 8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        out = self.conv_block(x)
        return out

class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.fc = nn.Linear(100, 8 * 8 * 1024)
        
        self.conv_block = nn.Sequential(
            # 8 x 8 x 1024 -> 16 x 16 x 512
            nn.ConvTranspose2d(1024, 512, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 16 x 16 x 512 -> 32 x 32 x 256
            nn.ConvTranspose2d(512, 256, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 32 x 32 x 256 -> 64 x 64 x 128
            nn.ConvTranspose2d(256, 128, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 x 64 x 128 -> 128 x 128 x 64
            nn.ConvTranspose2d(128, 64, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        
            # 128 x 128 x 64 -> 256 x 256 x 1
            nn.ConvTranspose2d(64, 1, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2), bias=True),
            nn.Tanh()
        )

    def forward(self, noise):
        out = self.fc(noise)
        out = out.view(out.shape[0], 1024, 8, 8)
        out = self.conv_block(out)
        return out
