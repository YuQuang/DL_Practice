import numpy as np
import cv2 as cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from model.model import Discriminator, Generator
from handdataset import TrainDatasets

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # G = Generator().to(device)
    # D = Discriminator().to(device)
    G = torch.load("Generator_epoch_100.pth")
    D = torch.load("Discriminator_epoch_100.pth")

    # G.apply(weights_init)
    # D.apply(weights_init)

    # Settings
    epochs = 500
    batch_size = 128

    loss_function = nn.BCELoss()
    g_optimizer = optim.Adam(G.parameters(), lr=0.0035, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(D.parameters(), lr=0.00001, betas=(0.5, 0.999))

    # Dataloader
    train_dataset = TrainDatasets("Hands\Hands", r"Hand_[0-9]+[\.]jpg")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size//2, shuffle=False, num_workers=4)
    
    for epoch in range(1, epochs+1):
        for batch_idx, (real_data, real_label) in enumerate(train_dataloader):
            #
            # Train Discriminator with real data
            #
            real_data  = real_data.to(device)
            real_label = real_label.unsqueeze(1).float().to(device)

            d_optimizer.zero_grad()
            d_pred = D(real_data)
            d_real_loss = loss_function(d_pred, real_label)
            d_real_loss.backward()
            d_optimizer.step()

            #
            # Train Discriminator with fake data
            #
            noise      = torch.from_numpy(np.random.normal(0, 0.1, (batch_size//2, 100) )).float().to(device)
            fake_label = torch.zeros( ( batch_size//2, 1), device=device)
            fake_data  = G(noise)

            d_optimizer.zero_grad()
            d_pred = D(fake_data)
            d_fake_loss = loss_function(d_pred, fake_label)
            d_fake_loss.backward()
            d_optimizer.step()

            
            #
            # Train Generator
            #
            noise  = torch.from_numpy(np.random.normal(0, 0.1, (batch_size, 100) )).float().to(device)
            target = torch.ones( (batch_size, 1), device=device )

            g_optimizer.zero_grad()
            d_pred = D( G(noise) )
            g_loss = loss_function(d_pred, target)
            g_loss.backward()
            g_optimizer.step()

            cv2.imshow("gen_img", fake_data[0].cpu().detach().numpy().reshape(256, 256, 1))
            cv2.imshow("real_img", real_data[0].cpu().detach().numpy().reshape(256, 256, 1))
            cv2.waitKey(10)

            print(f"Epochs={epoch}/{epochs}, iter={batch_idx}/{train_dataloader.__len__()}")
            print(f"d_real_loss={d_real_loss.item()}, d_fake_loss={d_fake_loss.item()}, g_loss={g_loss.item()}")
        
        if epoch % 50 == 0:
            torch.save(G, f"Generator_epoch_{epoch}.pth")
            torch.save(D, f"Discriminator_epoch_{epoch}.pth")