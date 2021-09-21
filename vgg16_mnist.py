import datetime
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.autograd import Variable
from torch.hub import load_state_dict_from_url
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Union, List, Dict, Any, cast

__all__ = [
    'VGG', 'vgg16', 'vgg16_bn'
]

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
}


class VGG(nn.Module):

    def __init__(
            self,
            features: nn.Module,
            num_classes: int = 10,
            init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


# select the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# define the model
model = vgg16(False, True).to(device)

trans = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Pad(2),
     transforms.Normalize((0.5), (0.5))])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=trans)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=trans)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

batch_size = 64
num_epoches = 50
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

train_loss = np.zeros(0)
train_err = np.zeros(0)
test_loss = np.zeros(0)
top1_acc = np.zeros(0)
top5_acc = np.zeros(0)
starttime = datetime.datetime.now()
print("start time:", starttime)
for epoch in range(num_epoches):
    print('**********epoch {}*********'.format(epoch + 1))
    running_loss = 0.0
    running_acc = 0.0
    total_train = len(train_dataset)
    for i, data in enumerate(train_loader, 0):

        inputs, label = data
        # cuda
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            label = label.cuda()
        inputs = Variable(inputs)
        label = Variable(label)
        outputs = model(inputs)
        loss = criterion(outputs, label)
        running_loss += loss.item() * label.size(0)
        _, pred = torch.max(outputs, 1)
        num_correct = (pred == label).sum()
        accuracy = (pred == label).float().mean()
        running_acc += num_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Finish Training {} epoch, Loss: {:.6f}, Error: {:.6f}'.format(
        epoch + 1, running_loss / total_train, (1 - (running_acc / total_train))))
    train_loss = np.append(train_loss, running_loss / total_train)
    train_err = np.append(train_err, (1 - (running_acc / total_train)))

    model.eval()
    eval_loss = 0
    eval_acc_top1 = 0
    eval_acc_top5 = 0
    total_test = len(test_dataset)
    for data in test_loader:
        inputs, label = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            label = label.cuda()
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, label)
            eval_loss += loss.item() * label.size(0)
            _, pred = torch.max(outputs, 1)
            num_correct = (pred == label).sum()
            eval_acc_top1 += num_correct.item()

            correct = 0
            maxk = max((1, 5))
            label_resize = label.view(-1, 1)
            _, pred = outputs.topk(maxk, 1, True, True)
            correct += torch.eq(pred, label_resize).sum()
            eval_acc_top5 += correct.item()
    print('Test Loss: {:.6f}, Top1_Acc: {:.6f} , Top5_Acc: {:.6f}'.format(
        eval_loss / total_test, eval_acc_top1 / total_test, eval_acc_top5 / total_test))
    test_loss = np.append(test_loss, eval_loss / total_test)
    top1_acc = np.append(top1_acc, eval_acc_top1 / total_test)
    top5_acc = np.append(top5_acc, eval_acc_top5 / total_test)

endtime = datetime.datetime.now()
print("End time:", endtime)
