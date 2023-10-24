import torch
import torch.nn as nn
import torch.optim as optim

from model import Meta
from torch.utils.data.dataloader import DataLoader
from dataset import miniImage

if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    #
    # Init model
    #
    meta = torch.load("Meta_29999.pt").to(device)
    epochs = 10
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(meta.parameters(), lr=0.0001, weight_decay=1e-5)

    #
    # Prepare dataset
    #
    test_dataset = miniImage("./miniImage/", n_way=5, k_shot=5, k_query=15)
    test_loader = DataLoader(test_dataset, batch_size=1)

    #
    # Start fine-tune
    #
    for epoch in range(epochs):
        for spt_data, spt_label, qry_data, qry_label in test_loader:
            spt_data = spt_data[0].to(device)
            spt_label = spt_label[0].to(device)
            qry_data = qry_data[0].to(device)
            qry_label = qry_label[0].to(device)

            y = meta(spt_data)

            optimizer.zero_grad()
            loss = loss_fn(y, spt_label)
            loss.backward()
            optimizer.step()

            break

        print(f"Epoch={epoch}/{epochs}, loss={loss.item()}")
    
    #