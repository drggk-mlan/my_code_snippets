"""
基础模型模块
"""
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """基础模型类"""
    
    def __init__(self):
        super(BaseModel, self).__init__()
    
    def forward(self, x):
        raise NotImplementedError("子类必须实现forward方法")
    
    def get_optimizer(self, lr: float = 0.001, weight_decay: float = 1e-4):
        """
        获取优化器
        
        Args:
            lr: 学习率
            weight_decay: 权重衰减
            
        Returns:
            优化器
        """
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
    
    def get_scheduler(self, optimizer, step_size: int = 30, gamma: float = 0.1):
        """
        获取学习率调度器
        
        Args:
            optimizer: 优化器
            step_size: 步长
            gamma: 衰减系数
            
        Returns:
            调度器
        """
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    def save(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        self.load_state_dict(torch.load(path))


class SimpleCNN(BaseModel):
    """简单的CNN模型"""
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        """
        初始化模型
        
        Args:
            num_classes: 类别数量
            input_channels: 输入通道数
        """
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class SimpleResNet(BaseModel):
    """简单的ResNet模型"""
    
    def __init__(self, num_classes: int = 10, num_blocks: list = [2, 2, 2, 2]):
        """
        初始化模型
        
        Args:
            num_classes: 类别数量
            num_blocks: 每层的残差块数量
        """
        super(SimpleResNet, self).__init__()
        
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, out_channels: int, num_blocks: int, stride: int):
        """构建残差层"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
