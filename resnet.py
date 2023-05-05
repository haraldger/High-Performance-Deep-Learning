import numpy as np
import torch
from torch import nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x 

        out = self.conv1(x)
        out = self.bn1(out) 
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels//4)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels//4)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x 

        out = self.conv1(x)
        out = self.bn1(out) 
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out) 
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out) 

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):
    def __init__(self, block_type, layers, large_mode=False) -> None:
        super().__init__()
        self.block_type = block_type
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # Batch normalization after the initial convolutional layer
        self.bn1 = nn.BatchNorm2d(64)
        # Max pooling after the batch normalization layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Output channels
        if large_mode:
            output_channels = [256, 512, 1024, 2048]
        else:
            output_channels = [64, 128, 256, 512]

        # ResNet layers
        self.layer1 = self.make_layer(64, output_channels[0], layers[0])
        self.layer2 = self.make_layer(output_channels[0], output_channels[1], layers[1])
        self.layer3 = self.make_layer(output_channels[1], output_channels[2], layers[2])
        self.layer4 = self.make_layer(output_channels[2], output_channels[3], layers[3])

        # Prediction head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if large_mode:
            self.fc = nn.Linear(2048, 1000)
        else:
            self.fc = nn.Linear(512, 1000)
        self.softmax = nn.Softmax(dim=1)




        
    def forward(self, x):
        # Initial convolutional layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.maxpool(out)

        # ResNet layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Prediction head
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = self.softmax(out)

        # Return learned features
        return out

        
    
    def make_layer(self, in_channels, out_channels, num_blocks):
        blocks = []

        # Create the first block separately to perform downsampling
        if in_channels != out_channels:
            blocks.append(self.block_type(in_channels, out_channels, stride=2))
        else:
            blocks.append(self.block_type(in_channels, out_channels, stride=1))

        # Create the rest of the blocks
        for i in range(1, num_blocks):    
            blocks.append(self.block_type(out_channels, out_channels, stride=1))

        return nn.Sequential(*blocks)

    
# Plain ResNets

def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet101():
    return ResNet(BasicBlock, [3, 4, 23, 3])

def resnet152():
    return ResNet(BasicBlock, [3, 8, 36, 3])



# ResNets with bottleneck blocks

def resnet18_bottleneck():
    return ResNet(Bottleneck, [2, 2, 2, 2])

def resnet34_bottleneck():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def resnet50_bottleneck():
    return ResNet(Bottleneck, [3, 4, 6, 3], large_mode=True)

def resnet101_bottleneck():
    return ResNet(Bottleneck, [3, 4, 23, 3], large_mode=True)

def resnet152_bottleneck():
    return ResNet(Bottleneck, [3, 8, 36, 3], large_mode=True)


# Test the network

def test_plain_resnet_models_forward():
    net = resnet18()
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    print(f'ResNet18 network = {net}')

    net = resnet34()
    y = net(x)
    print(f'ResNet34 network = {net}')

    net = resnet50()
    y = net(x)
    print(f'ResNet50 network = {net}')

    net = resnet101()
    y = net(x)
    print(f'ResNet101 network = {net}')

    net = resnet152()
    y = net(x)
    print(f'ResNet152 network = {net}')

def test_bottleneck_resnet_models_forward():
    net = resnet18_bottleneck()
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    print(f'ResNet18 network = {net}')

    net = resnet34_bottleneck()
    y = net(x)
    print(f'ResNet34 network = {net}')

    net = resnet50_bottleneck()
    y = net(x)
    print(f'ResNet50 network = {net}')

    net = resnet101_bottleneck()
    y = net(x)
    print(f'ResNet101 network = {net}')

    net = resnet152_bottleneck()
    y = net(x)
    print(f'ResNet152 network = {net}')

def test_plain_resnet_models_shapes():
    net = resnet18()
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    print(f'ResNet18 shapes = {y.shape}')
    assert y.shape == (2, 1000)

    net = resnet34()
    y = net(x)
    print(f'ResNet34 shapes = {y.shape}')
    assert y.shape == (2, 1000)

    net = resnet50()
    y = net(x)
    print(f'ResNet50 shapes = {y.shape}')
    assert y.shape == (2, 1000)

    net = resnet101()
    y = net(x)
    print(f'ResNet101 shapes = {y.shape}')
    assert y.shape == (2, 1000)

    net = resnet152()
    y = net(x)
    print(f'ResNet152 shapes = {y.shape}')
    assert y.shape == (2, 1000)

def test_bottleneck_resnet_models_shapes():
    net = resnet18_bottleneck()
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    print(f'ResNet18 shapes = {y.shape}')
    assert y.shape == (2, 1000)

    net = resnet34_bottleneck()
    y = net(x)
    print(f'ResNet34 shapes = {y.shape}')
    assert y.shape == (2, 1000)

    net = resnet50_bottleneck()
    y = net(x)
    print(f'ResNet50 shapes = {y.shape}')
    assert y.shape == (2, 1000)

    net = resnet101_bottleneck()
    y = net(x)
    print(f'ResNet101 shapes = {y.shape}')
    assert y.shape == (2, 1000)

    net = resnet152_bottleneck()
    y = net(x)
    print(f'ResNet152 shapes = {y.shape}')
    assert y.shape == (2, 1000)


def main():
    test_plain_resnet_models_forward()
    test_bottleneck_resnet_models_forward()
    test_plain_resnet_models_shapes()
    test_bottleneck_resnet_models_shapes()
    

if __name__ == '__main__':
    main()
