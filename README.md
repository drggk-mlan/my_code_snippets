# my_code_snippets
关于动手学深度学习一书的一些代码框架之类的小玩意

## PyTorch深度学习脚手架

一个简单但完整的PyTorch训练框架，适合快速开始深度学习项目。

### 特性

- 🚀 **开箱即用**: 包含完整的训练流程和常用模型
- 📊 **训练监控**: 集成TensorBoard支持
- 💾 **检查点管理**: 自动保存最佳模型和训练检查点
- 🔧 **配置灵活**: 支持YAML配置文件
- 📈 **指标跟踪**: 自动跟踪损失和准确率
- 🎯 **示例丰富**: 包含CIFAR-10和MNIST数据集示例

### 目录结构

```
pytorch_scaffold/
├── __init__.py          # 包初始化
├── config.py            # 配置管理
├── trainer.py           # 训练器
├── models/              # 模型定义
│   └── __init__.py      # 基础模型、SimpleCNN、SimpleResNet
├── data/                # 数据加载
│   └── __init__.py      # 数据集加载器
├── utils/               # 工具函数
│   └── __init__.py      # 辅助工具
└── configs/             # 配置文件
    └── default_config.yaml
```

### 安装

1. 克隆仓库:
```bash
git clone https://github.com/drggk-mlan/my_code_snippets.git
cd my_code_snippets
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

### 快速开始

#### 基础使用

运行示例训练脚本:
```bash
python train_example.py
```

#### 自定义训练

```python
import torch
import torch.nn as nn
from pytorch_scaffold.models import SimpleCNN
from pytorch_scaffold.data import get_cifar10_loaders
from pytorch_scaffold.trainer import Trainer
from pytorch_scaffold.utils import set_seed

# 设置随机种子
set_seed(42)

# 加载数据
train_loader, test_loader = get_cifar10_loaders(batch_size=128)

# 创建模型
model = SimpleCNN(num_classes=10)

# 创建优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 创建训练器
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    save_dir='./checkpoints'
)

# 开始训练
trainer.fit(num_epochs=10)
```

#### 使用配置文件

1. 修改配置文件 `pytorch_scaffold/configs/default_config.yaml`:
```yaml
model:
  name: 'simple_resnet'
  num_classes: 10

training:
  num_epochs: 100
  batch_size: 128
  learning_rate: 0.1
```

2. 运行训练:
```bash
python train_example.py
```

### 核心组件

#### 1. 模型 (models/)

提供了几个基础模型:

- **BaseModel**: 基础模型类，提供通用方法
- **SimpleCNN**: 简单的CNN模型
- **SimpleResNet**: 简化版的ResNet模型

自定义模型示例:
```python
from pytorch_scaffold.models import BaseModel
import torch.nn as nn

class MyModel(BaseModel):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

#### 2. 训练器 (trainer.py)

`Trainer`类封装了完整的训练流程:

- 自动训练循环
- 验证评估
- 检查点保存
- TensorBoard日志
- 学习率调度

#### 3. 数据加载 (data/)

提供常用数据集的加载器:

- `get_cifar10_loaders()`: CIFAR-10数据集
- `get_mnist_loaders()`: MNIST数据集
- `CustomDataset`: 自定义数据集模板

#### 4. 工具函数 (utils/)

实用工具函数:

- `set_seed()`: 设置随机种子
- `get_device()`: 自动选择设备
- `count_parameters()`: 统计模型参数
- `AverageMeter`: 指标计算
- `accuracy()`: 准确率计算
- `save_checkpoint()` / `load_checkpoint()`: 检查点管理

#### 5. 配置管理 (config.py)

支持灵活的配置管理:

```python
from pytorch_scaffold.config import Config

# 从YAML加载
config = Config.from_yaml('config.yaml')

# 获取配置
batch_size = config.get('training.batch_size', 32)

# 设置配置
config.set('training.learning_rate', 0.001)
```

### 高级功能

#### 使用TensorBoard监控训练

训练期间会自动记录指标到TensorBoard:

```bash
tensorboard --logdir=./checkpoints/runs
```

然后在浏览器中打开 http://localhost:6006

#### 从检查点恢复训练

```python
trainer = Trainer(...)
trainer.load_checkpoint('checkpoints/best_model.pth')
trainer.fit(num_epochs=10)  # 继续训练
```

#### 自定义学习率调度

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100
)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    ...
)
```

### 示例

查看 `train_example.py` 获取完整的训练示例。

### 依赖项

- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.24.0
- pyyaml >= 6.0
- tensorboard >= 2.13.0
- tqdm >= 4.65.0
- matplotlib >= 3.7.0

### 贡献

欢迎提交Issue和Pull Request！

### 许可证

MIT License
