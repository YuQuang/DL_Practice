import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt
from model.model import Discriminator, Generator

def show_images(images):
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))

    for index, image in enumerate(images):
        plt.subplot(sqrtn, sqrtn, index+1)
        plt.imshow(image.reshape(28, 28))

def train(model, optimizer, loss_function, data, label):
    pred = model(data)
    optimizer.zero_grad()
    loss = loss_function(pred, label)
    loss.backward()
    optimizer.step()
    return loss.item()

if __name__ == "__main__":
    plt.rcParams['image.cmap'] = 'gray'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    G = Generator().to(device)
    D = Discriminator().to(device)

    # Settings
    epochs = 200
    batch_size = 64
    loss_function = nn.BCELoss()
    g_optimizer = optim.Adam(G.parameters(), lr=0.0006, betas=(0.1, 0.999))
    d_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.1, 0.999))

    # Transform
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # Load data
    train_set = datasets.MNIST('mnist/', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('mnist/', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Train
    for epoch in range(1, epochs+1):
        d_loss, g_loss = 0, 0
        for times, data in enumerate(train_loader):
            """
                This part is training the Discriminator
            """
            # Real data
            real_inputs = data[0].to(device).flatten(1)
            real_label = torch.ones(real_inputs.shape[0], 1).to(device)

            # Fake data
            noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
            noise = noise.to(device)
            fake_inputs = G(noise)
            fake_label = torch.zeros(fake_inputs.shape[0], 1).to(device)

            d_loss += train(
                D,
                d_optimizer,
                loss_function,
                torch.cat( (real_inputs, fake_inputs), 0),
                torch.cat( (real_label, fake_label), 0)
            )


            """
                This part is training the Generator
            """
            noise = (torch.rand(real_inputs.shape[0], 128)-0.5)/0.5
            noise = noise.to(device)
            targets = torch.ones([real_inputs.shape[0], 1]).to(device)

            g_loss += train(
                nn.Sequential(G, D),
                g_optimizer,
                loss_function,
                noise,
                targets
            )

            if times % 100 == 0 or times == len(train_loader):
                print(f'[{epoch}/{epochs}, {times}/{len(train_loader)}] D_loss: {d_loss/(times+1)} G_loss: {g_loss/(times+1)}')

        if epoch % 30 == 0:
            imgs_numpy = (fake_inputs.data.cpu().numpy()+1.0)/2.0
            show_images(imgs_numpy[:16])
            plt.show()
        
        if epoch % 50 == 0:
            torch.save(G, 'Generator_epoch_{}.pth'.format(epoch))
            print('Model saved.')

    print('Training Finished.')