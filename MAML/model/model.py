import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import cv2 as cv2
import torchvision.models as models

from dataset import miniImage
from learner import MyAlexNet
from ops import linear, conv2d, dropout, maxpool

class Meta(nn.Module):
    def __init__(self, device):
        super(Meta, self).__init__()
        #
        # Task
        #
        self.network = MyAlexNet(num_classes=5).to(device)

        #
        # Meta Model
        #
        original_model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

        self.features = nn.Sequential(*list(original_model.features.children()))
        self.classifier = nn.Sequential(*list(original_model.classifier.children())[:-1])
        self.last_layer = nn.Linear(4096, 5)

        self.weight_to_be_used = []
        self.bias_to_be_used = []

        for child in self.features.children():
            if isinstance(child, nn.modules.conv.Conv2d) or isinstance(child, nn.modules.Linear):
                self.weight_to_be_used.append(child.weight)
                self.bias_to_be_used.append(child.bias)

        for child in self.classifier.children():
            if isinstance(child, nn.modules.conv.Conv2d) or isinstance(child, nn.modules.Linear):
                self.weight_to_be_used.append(child.weight)
                self.bias_to_be_used.append(child.bias)

        self.weight_to_be_used.append(self.last_layer.weight)
        self.bias_to_be_used.append(self.last_layer.bias)

    def assign_meta_to_network(self, device):
        with torch.no_grad():
            for index, weight in enumerate(self.weight_to_be_used):
                new_weight = torch.from_numpy(weight.detach().cpu().numpy()).to(device)
                self.network.weight_to_be_used[index] = torch.nn.Parameter(new_weight)

            for index, bias in enumerate(self.bias_to_be_used):
                new_bias = torch.from_numpy(bias.detach().cpu().numpy()).to(device)
                self.network.bias_to_be_used[index] = torch.nn.Parameter(new_bias)

    def forward(self, x, meta_loss=None, meta_step_size=None, stop_gradient=False):
        # Features Block
        # Block 1
        x = conv2d(inputs=x, weight=self.weight_to_be_used[0], bias=self.bias_to_be_used[0], meta_step_size=0.001,
                   stride=4, padding=2, dilation=1, groups=1, meta_loss=None,
                   stop_gradient=False)

        x = F.relu(input=x, inplace=True)

        x = maxpool(inputs=x, kernel_size=3, stride=2)

        # Block 2
        x = conv2d(inputs=x, weight=self.weight_to_be_used[1], bias=self.bias_to_be_used[1], meta_step_size=0.001,
                   padding=2, dilation=1, groups=1, meta_loss=None,
                   stop_gradient=False)

        x = F.relu(input=x, inplace=True)

        x = maxpool(inputs=x, kernel_size=3, stride=2)

        # Block 3
        x = conv2d(inputs=x, weight=self.weight_to_be_used[2], bias=self.bias_to_be_used[2], meta_step_size=0.001,
                   padding=1, dilation=1, groups=1, meta_loss=None,
                   stop_gradient=False)

        x = F.relu(input=x, inplace=True)

        # Block 4
        x = conv2d(inputs=x, weight=self.weight_to_be_used[3], bias=self.bias_to_be_used[3], meta_step_size=0.001,
                   padding=1, dilation=1, groups=1, meta_loss=None,
                   stop_gradient=False)

        x = F.relu(input=x, inplace=True)

        # Block 5
        x = conv2d(inputs=x, weight=self.weight_to_be_used[4], bias=self.bias_to_be_used[4], meta_step_size=0.001,
                   padding=1, dilation=1, groups=1, meta_loss=None,
                   stop_gradient=False)

        x = F.relu(input=x, inplace=True)

        x = maxpool(inputs=x, kernel_size=3, stride=2)

        x = x.view(x.size(0), 256 * 6 * 6)

        # Classifier Block

        x = dropout(inputs=x)

        # Block 1
        x = linear(inputs=x,
                   weight=self.weight_to_be_used[5], bias=self.bias_to_be_used[5],
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)

        x = F.relu(input=x, inplace=True)

        x = dropout(inputs=x)

        # Block 2
        x = linear(inputs=x,
                   weight=self.weight_to_be_used[6], bias=self.bias_to_be_used[6],
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)

        x = F.relu(input=x, inplace=True)

        x = linear(inputs=x,
                 weight=self.weight_to_be_used[7], bias=self.bias_to_be_used[7],
                 meta_loss=meta_loss,
                 meta_step_size=meta_step_size,
                 stop_gradient=stop_gradient)

        x = F.softmax(input=x, dim=-1)
        return x
    
    def support_task(self, data, target, step=1, device="cpu"):
        self.network.train()

        task_loss_fn = nn.CrossEntropyLoss()
        task_optimer = optim.SGD([
            { "params": self.network.bias_to_be_used },
            { "params": self.network.weight_to_be_used }
        ], lr=0.0005)
        for _ in range(step):
            data = data.to(device)
            target = target.to(device)
            y = self.network(data)

            task_optimer.zero_grad()
            loss = task_loss_fn(y, target)
            loss.backward()
            task_optimer.step()
            print(f"Support_set task loss {loss.item()}")

    def query_task(self, data, target, optimizer, device="cpu"):
        self.network.train()

        task_loss_fn = nn.CrossEntropyLoss()
        task_optimer = optim.SGD([
            { "params": self.network.bias_to_be_used },
            { "params": self.network.weight_to_be_used }
        ], lr=0.0005)

        data = data.to(device)
        target = target.to(device)
        y = self.network(data)
        task_optimer.zero_grad()
        optimizer.zero_grad()
        loss = task_loss_fn(y, target)
        loss.backward()

        grad_list = []
        for _, parameter in meta.network.named_parameters(recurse=False):
                grad_list.append(parameter.grad)

        for index, (_, parameter) in enumerate(meta.named_parameters(recurse=False)):
            parameter.grad = grad_list[index]
        optimizer.step()

        task_optimer.zero_grad()
        optimizer.zero_grad()

        print(f"Query_set task loss {loss.item()}")

if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    #
    # Init Model
    #
    meta = Meta(device).to(device)
    epochs = 1
    meta_loss_fn = nn.CrossEntropyLoss()
    meta_optimizer = optim.SGD([
        { "params": meta.weight_to_be_used }, { "params": meta.bias_to_be_used }
    ], lr=0.01)

    train_dataset = miniImage("./miniImage/")
    train_loader = DataLoader(train_dataset, batch_size=8)

    for epoch in range(epochs):
        #
        # Train Meta Model
        #
        for iter_count, (spt_data, spt_lable, qry_data, qry_label) in enumerate(train_loader):
            for batch, _ in enumerate(spt_data):
                meta.support_task(spt_data[batch], spt_lable[batch], step=2, device=device)
                meta.query_task(qry_data[batch], qry_label[batch], optimizer=meta_optimizer, device=device)
            meta.assign_meta_to_network(device)

            print(f"\nEpoch={epoch+1}/{epochs}, Iter={iter_count+1}/{train_loader.__len__()}")

            cv2.imshow("Spport_set", spt_data[0][0].numpy().reshape(224, 224, 3))
            cv2.imshow("Query_set", qry_data[0][0].numpy().reshape(224, 224, 3))
            cv2.waitKey(10)

            if (iter_count+1) % 100 == 0:
                torch.save(meta, f"Meta_{iter_count+1}.pt")