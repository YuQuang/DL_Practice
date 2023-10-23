import math
import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn
from ops import linear, conv2d, dropout, maxpool

class MyAlexNet(nn.Module):

    def __init__(self, num_classes=10):
        original_model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        freeze_layers = True
        super(MyAlexNet, self).__init__()

        # freeze the pre-trained features
        if freeze_layers:
            for i, param in original_model.features.named_parameters():
                param.requires_grad = False

        self.features = nn.Sequential(*list(original_model.features.children()))
        self.classifier = nn.Sequential(*list(original_model.classifier.children())[:-1])
        self.last_layer = nn.Linear(4096, num_classes)

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

        # when you add the convolution and batch norm, below will be useful
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

if __name__ == "__main__":
    model = MyAlexNet(5)
    pred_y = model(torch.randn((1, 3, 224, 224)))

