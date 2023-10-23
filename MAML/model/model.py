import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from learner import MyAlexNet
from ops import linear, conv2d, dropout, maxpool

class Meta(nn.Module):
    def __init__(self, k_way=5, n_shot=10):
        super(Meta, self).__init__()
        #
        # Task
        #
        self.network = MyAlexNet(num_classes=10)
        self.task_loss_fn = nn.CrossEntropyLoss()
        self.task_optimer = optim.Adam(
            filter(lambda p: p.requires_grad, self.network.parameters()),
            lr=1e-6, weight_decay=1e-8)

        #
        # Meta Model
        #
        original_model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

        self.features = nn.Sequential(*list(original_model.features.children()))
        self.classifier = nn.Sequential(*list(original_model.classifier.children())[:-1])
        self.last_layer = nn.Linear(4096, 10)

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

    def assign_meta_to_network(self):
        for index, weight in enumerate(self.weight_to_be_used):
            self.network.weight_to_be_used[index] = weight

        for index, bias in enumerate(self.bias_to_be_used):
            self.network.bias_to_be_used[index] = bias

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
    
    def train_task(self, support_set, query_set):
        self.network.train()            
        
        #
        # Support_set train task
        #
        for data, label in support_set:
            y = self.network(data)

            self.task_optimer.zero_grad()
            loss = self.task_loss_fn(y, label)
            loss.backward()
            self.task_optimer.step()
            
            print(f"Support_set task loss {loss.item()}")
            break

        #
        # Query_set train task
        #
        for data, label in query_set:
            y = self.network(data)

            self.task_optimer.zero_grad()
            loss = self.task_loss_fn(y, label)
            loss.backward()

            print(f"Query_set task loss {loss.item()}")
            break

if __name__ == "__main__":
    #
    # Init Model
    #
    meta = Meta()
    epochs = 10
    n_shot = 20
    batch_size = 32
    meta_loss_fn = nn.CrossEntropyLoss()
    meta_optimizer = optim.Adam([
        { "params": meta.weight_to_be_used }, { "params": meta.bias_to_be_used }
    ], lr=1e-4, weight_decay=1e-8)

    #
    # Load Dataset
    #
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3,1,1)),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize((224, 224)),
    ])
    mnist_train = datasets.MNIST('mnist/', train=True, download=True, transform=transform)
    train_support_set, train_query_set = torch.utils.data.random_split(mnist_train, [0.9, 0.1])
    train_support_loader = DataLoader(train_support_set, batch_size=n_shot, shuffle=True)
    train_query_loader = DataLoader(train_query_set, batch_size=n_shot, shuffle=True)

    mnist_test = datasets.MNIST('mnist/', train=False, download=True, transform=transform)
    test_query_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

    #
    # Train Meta Model
    #
    for epoch in range(epochs):
        meta.train_task(support_set=train_support_loader, query_set=train_query_loader)

        grad_list = []
        for _, parameter in meta.network.named_parameters(recurse=False):
            grad_list.append(parameter.grad)

        for index, (name, parameter) in enumerate(meta.named_parameters(recurse=False)):
            parameter.grad = grad_list[index]
        meta_optimizer.step()

        meta.assign_meta_to_network()
        print(f"Epoch={epoch}")

    #
    # Test Meta Model
    #
    for epoch in range(epochs):
        for data, label in test_query_loader:

            y = meta(data)

            meta_optimizer.zero_grad()
            loss = meta_loss_fn(y, label)
            loss.backward()
            meta_optimizer.step()

            print(f"Meta train loss={loss.item()}")