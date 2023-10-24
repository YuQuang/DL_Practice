import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import torchvision.models as models
from dataset import miniImage

def count_acc(y, target):
    corect_count = 0
    for index, y_index in enumerate(F.softmax(y, 1).argmax(1)):
        if y_index.item() == target[index]: corect_count+=1

    return corect_count/target.__len__() * 100

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #
    # Init Model
    #
    alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).to(device)
    epochs = 20
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(alexnet.parameters(), lr=0.0002, weight_decay=1e-6)

    #
    # Dataset
    #
    test_dataset = miniImage("./miniImage", n_way=5, k_shot=5, k_query=40)
    test_loader  = DataLoader(test_dataset, batch_size=1)
    
    #
    # Fine-tune
    #
    for epoch in range(epochs):
        for spt_data, spt_label, _, _ in test_loader:
            spt_data, spt_label = spt_data.to(device), spt_label.to(device)

            optimizer.zero_grad()
            y = alexnet(spt_data[0])
            loss = criterion(y, spt_label[0])
            loss.backward()
            optimizer.step()

            acc = count_acc(y, spt_label[0])

            print(f"\nEpoch={epoch}/{epochs}, Non-Meta train_loss={loss.item()}, train_acc={acc}%")
            break
        
        for _, _, qry_data, qry_label in test_loader:
            qry_data, qry_label = qry_data.to(device), qry_label.to(device)

            y = alexnet(qry_data[0])
            loss = criterion(y, qry_label[0])
            acc = count_acc(y, qry_label[0])

            print(f"test_loss={loss.item()}, test_acc={acc}%")
            break