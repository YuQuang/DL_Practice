import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model import Meta
from torch.utils.data.dataloader import DataLoader
from miniimage import MiniImage
from omniglot import Omniglot

def count_acc(y, target):
    corect_count = 0
    for index, y_index in enumerate(F.softmax(y, 1).argmax(1)):
        if y_index.item() == target[index]: corect_count+=1

    return corect_count/target.__len__() * 100

if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    #
    # Init model
    #
    meta = torch.load("Omniglot_32Batch4Task_5way_5shot_1000.pth").to(device)
    epochs = 10
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(meta.parameters(), lr=0.0001, weight_decay=1e-8)

    #
    # Prepare dataset
    #
    # test_dataset = miniImage("./miniImage/", n_way=5, k_shot=5, k_query=100)
    # test_loader = DataLoader(test_dataset, batch_size=1)
    test_dataset = Omniglot("./omniglot/omniglot-py/", n_way=5, k_shot=5, k_query=20, background=False)
    test_loader = DataLoader(test_dataset, batch_size=1)

    #
    # Start fine-tune
    #
    for epoch in range(epochs):
        for spt_data, spt_label, _, _ in test_loader:
            spt_data = spt_data[0].to(device)
            spt_label = spt_label[0].to(device)

            y, _ = meta(spt_data)
            optimizer.zero_grad()
            loss = loss_fn(y, spt_label)
            loss.backward()
            optimizer.step()
            
            acc = count_acc(y, spt_label)
            break
        print(f"Epoch={epoch}/{epochs}, train_loss={loss.item()}, train_acc={acc}%")

        for _, _, qry_data, qry_label in test_loader:
            qry_data = qry_data[0].to(device)
            qry_label = qry_label[0].to(device)

            y, _ = meta(qry_data)
            loss = loss_fn(y, qry_label)
            acc = count_acc(y, qry_label)
            break
        print(f"test_loss={loss.item()}, test_acc={acc}%")

        
