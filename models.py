import torch
import torch.nn as nn
import torch.nn.functional as F

# Resnet 3 Layers: Feature maps (64, 128, 256) with Dropouts
class BasicBlockDropouts(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.0):
        super(BasicBlockDropouts, self).__init__()  # Corrected this line
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.dropout1 = nn.Dropout(dropout_rate) #Dropouts as per https://arxiv.org/pdf/1904.03392.pdf
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout2 = nn.Dropout(dropout_rate) #Dropouts as per https://arxiv.org/pdf/1904.03392.pdf

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)  # Apply dropout after activation (new proposed method) https://arxiv.org/pdf/1904.03392.pdf
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)  # Apply dropout after activation (new proposed method) https://arxiv.org/pdf/1904.03392.pdf
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet3Dropouts(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, dropout_rate=0.5):
        super(ResNet3Dropouts, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
    
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(256*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out) # Dropout before final Class. Layer
        out = self.linear(out)
        return out

def ResNet3_with_dropout():
    return ResNet3Dropouts(BasicBlockDropouts, [2, 2, 2], dropout_rate=0.5)  # For example, a dropout rate of 0.5


# def test():
#     net = ResNet3_with_dropout()
#     y = net(torch.randn(1, 3, 32, 32))
#     print(y.size())

# test()


#--------- use in final model alongside resnet 3
class BasicBlock(nn.Module): 
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        # self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))   

        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

#Resnet 2 Layers: Feature maps (64, 256)
class ResNet2(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet2, self).__init__()
        self.in_planes = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=2)
    
        # Adaptive Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Classifier
        self.linear = nn.Linear(256*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def Resnet2(dropout_rate=0.0):
    return ResNet2(BasicBlock, [2, 2])

# def test():
#     net = Resnet2()
#     y = net(torch.randn(1, 3, 32, 32))
#     print(y.size())

# test()




# Resnet 3 Layers: Feature maps (64, 128, 256)   ---------------Final Model
class ResNet3(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet3, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
    
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(256*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def Resnet3():
    return ResNet3(BasicBlock, [2, 2, 2])

# def test():
#     net = Resnet3()
#     y = net(torch.randn(1, 3, 32, 32))
#     print(y.size())

# test()



# Resnet 4 Layers: Feature maps (64, 128, 128, 256)


class ResNet4(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet4, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
    
        self.linear = nn.Linear(256*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4) 
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Adjusting the number of blocks to reduce the total parameters
def Resnet4():
    return ResNet4(BasicBlock, [2, 2, 2, 2]) #numbers indicate amount of blocks per layer. 1st 3, 2nd 2, 3rd 3, 4th 3,  

# def test():
#     net = Resnet4()
#     y = net(torch.randn(1, 3, 32, 32))
#     print(y.size())

# test()

# Resnet 5 Layers: Feature maps (64, 64, 128, 128, 256)


class ResNet5(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet5, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(block, 256, num_blocks[4], stride=2)
    
        self.linear = nn.Linear(256*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = F.avg_pool2d(out, 2) # reduces the deeper the network gets: due to way conv layers are setup, here pooling over 2by 2 region
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Creating the model with 5 layers
def Resnet5():
    return ResNet5(BasicBlock, [2, 2, 2, 2, 2]) # 5 layers with 2 blocks each

# def test():
#     net = Resnet5()
#     y = net(torch.randn(1, 3, 32, 32))
#     print(y.size())

# test()

