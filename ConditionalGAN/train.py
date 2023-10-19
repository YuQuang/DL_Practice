import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import cv2 as cv2

from model.model import Discriminator, Generator


if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
    # G = Generator().to(device)
    # D = Discriminator().to(device)
    G = torch.load("G_epoch200.pth")
    D = torch.load("D_epoch200.pth")

    # Settings
    epochs        = 200
    batch_size    = 1024
    loss_function = nn.MSELoss().to(device)
    g_optimizer   = optim.Adam(G.parameters(), lr=0.0005, betas=(0.5, 0.999))
    d_optimizer   = optim.Adam(D.parameters(), lr=0.00003, betas=(0.5, 0.999))

    
    # Load data
    transform    = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set    = datasets.MNIST('mnist/', train=True, download=True, transform=transform)
    test_set     = datasets.MNIST('mnist/', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    for epoch in range(1, epochs+1):
        for iter_count, (real_data, real_label) in enumerate(train_loader):
            #
            # Train with fake image
            #
            noise = (torch.rand(batch_size, 128) - 0.5) / 0.5
            noise = noise.to(device)
            fake_label = torch.randint(0, 10, (batch_size,)).to(device)
            fake_data = G(noise, fake_label)

            d_pred = D(fake_data, fake_label)

            d_optimizer.zero_grad()
            fake_d_loss = loss_function(d_pred, torch.zeros(d_pred.size()[0], 1).to(device))
            fake_d_loss.backward()
            d_optimizer.step()

            #
            # Train with fake image
            #
            real_data  = real_data.to(device)
            real_label = real_label.to(device)
            d_pred  = D(real_data, real_label)
            
            d_optimizer.zero_grad()
            real_d_loss = loss_function(d_pred, torch.ones(d_pred.size()[0], 1).to(device))
            real_d_loss.backward()
            d_optimizer.step()

            #
            # Train Generator
            #
            noise = (torch.rand(batch_size, 128) - 0.5) / 0.5
            noise = noise.to(device)
            fake_label = torch.randint(0, 10, (batch_size,)).to(device)
            g_pred = D(G(noise, fake_label), fake_label)

            g_optimizer.zero_grad()
            real_g_loss = loss_function(g_pred, torch.ones(g_pred.size()[0], 1).to(device))
            real_g_loss.backward()
            g_optimizer.step()


            print(f"Epoch={epoch}/{epochs}, iter={iter_count}/{train_loader.__len__()}")
            print(f"fake_D_loss={fake_d_loss.item()}, real_D_loss={real_d_loss.item()}")
            print(f"real_g_loss={real_g_loss.item()}\n")

        
        noise = (torch.rand(10, 128) - 0.5) / 0.5
        noise = noise.to(device)
        fake_label = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).int().to(device)
        fake_data = G(noise, fake_label)
        for i in range(10):
            fake_img = fake_data[i].cpu().detach().numpy()
            fake_img = cv2.resize(fake_img, (128, 128))
            cv2.imshow(f"{fake_label[i].cpu().detach().numpy()}", fake_img)
            cv2.waitKey(10)
        
        if epoch % 100 == 0:
            torch.save(G, f"G_epoch{epoch}.pth")
            torch.save(D, f"D_epoch{epoch}.pth")