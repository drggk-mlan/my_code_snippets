#!/usr/bin/env python3
"""
训练示例脚本
展示如何使用PyTorch脚手架进行模型训练
"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from pytorch_scaffold.config import Config
from pytorch_scaffold.models import SimpleCNN, SimpleResNet
from pytorch_scaffold.data import get_cifar10_loaders, get_mnist_loaders
from pytorch_scaffold.trainer import Trainer
from pytorch_scaffold.utils import set_seed, get_device, count_parameters


def main():
    """主函数"""
    # 加载配置
    config_path = project_root / 'pytorch_scaffold/configs/default_config.yaml'
    config = Config.from_yaml(str(config_path))
    
    # 设置随机种子
    set_seed(config.get('seed', 42))
    
    # 设置设备
    device = get_device(config.get('device'))
    print(f'使用设备: {device}')
    
    # 加载数据
    dataset_name = config.get('data.dataset', 'cifar10')
    batch_size = config.get('training.batch_size', 128)
    num_workers = config.get('data.num_workers', 4)
    pin_memory = config.get('data.pin_memory', True)
    data_dir = config.get('data.data_dir', './data')
    
    print(f'加载数据集: {dataset_name}')
    if dataset_name == 'cifar10':
        train_loader, test_loader = get_cifar10_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    elif dataset_name == 'mnist':
        train_loader, test_loader = get_mnist_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    else:
        raise ValueError(f'不支持的数据集: {dataset_name}')
    
    # 创建模型
    model_name = config.get('model.name', 'simple_cnn')
    num_classes = config.get('model.num_classes', 10)
    
    print(f'创建模型: {model_name}')
    if model_name == 'simple_cnn':
        input_channels = 3 if dataset_name == 'cifar10' else 1
        model = SimpleCNN(num_classes=num_classes, input_channels=input_channels)
    elif model_name == 'simple_resnet':
        num_blocks = config.get('model.num_blocks', [2, 2, 2, 2])
        model = SimpleResNet(num_classes=num_classes, num_blocks=num_blocks)
    else:
        raise ValueError(f'不支持的模型: {model_name}')
    
    print(f'模型参数数量: {count_parameters(model):,}')
    
    # 创建损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 创建优化器
    optimizer_type = config.get('optimizer.type', 'adam')
    learning_rate = config.get('training.learning_rate', 0.001)
    weight_decay = config.get('training.weight_decay', 5e-4)
    
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        momentum = config.get('training.momentum', 0.9)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f'不支持的优化器: {optimizer_type}')
    
    # 创建学习率调度器
    scheduler_type = config.get('scheduler.type', 'step')
    
    if scheduler_type == 'step':
        step_size = config.get('scheduler.step_size', 30)
        gamma = config.get('scheduler.gamma', 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif scheduler_type == 'multistep':
        milestones = config.get('scheduler.milestones', [50, 75])
        gamma = config.get('scheduler.gamma', 0.1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
    elif scheduler_type == 'cosine':
        T_max = config.get('training.num_epochs', 100)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max
        )
    else:
        scheduler = None
    
    # 创建训练器
    save_dir = config.get('logging.save_dir', './checkpoints')
    log_interval = config.get('logging.log_interval', 10)
    use_tensorboard = config.get('logging.use_tensorboard', True)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir,
        log_interval=log_interval,
        use_tensorboard=use_tensorboard
    )
    
    # 开始训练
    num_epochs = config.get('training.num_epochs', 10)
    print(f'\n开始训练，共 {num_epochs} 轮...\n')
    trainer.fit(num_epochs=num_epochs, save_best=True)
    
    print(f'\n训练完成！最佳验证准确率: {trainer.best_acc:.2f}%')


if __name__ == '__main__':
    main()
