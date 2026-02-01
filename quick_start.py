#!/usr/bin/env python3
"""
快速开始示例
最小化的训练示例，展示如何快速使用脚手架
"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from pytorch_scaffold.models import SimpleCNN
from pytorch_scaffold.data import get_mnist_loaders
from pytorch_scaffold.trainer import Trainer
from pytorch_scaffold.utils import set_seed


def main():
    """快速训练示例"""
    print("=" * 60)
    print("PyTorch深度学习脚手架 - 快速开始示例")
    print("=" * 60)
    
    # 1. 设置随机种子
    set_seed(42)
    print("\n✓ 已设置随机种子")
    
    # 2. 加载数据
    print("✓ 正在加载MNIST数据集...")
    train_loader, test_loader = get_mnist_loaders(
        data_dir='./data',
        batch_size=128,
        num_workers=2
    )
    print(f"  训练集: {len(train_loader.dataset)} 样本")
    print(f"  测试集: {len(test_loader.dataset)} 样本")
    
    # 3. 创建模型
    print("\n✓ 创建SimpleCNN模型")
    model = SimpleCNN(num_classes=10, input_channels=1)
    
    # 4. 设置训练参数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 5. 创建训练器
    print("✓ 初始化训练器")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        save_dir='./checkpoints',
        log_interval=50,
        use_tensorboard=True
    )
    
    # 6. 开始训练
    print("\n" + "=" * 60)
    print("开始训练 (5轮)")
    print("=" * 60 + "\n")
    
    trainer.fit(num_epochs=5, save_best=True)
    
    # 7. 显示结果
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"\n最佳验证准确率: {trainer.best_acc:.2f}%")
    print(f"模型保存在: ./checkpoints/best_model.pth")
    print(f"\n提示: 运行以下命令查看TensorBoard:")
    print("  tensorboard --logdir=./checkpoints/runs")
    print("=" * 60)


if __name__ == '__main__':
    main()
