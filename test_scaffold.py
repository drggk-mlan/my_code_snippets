#!/usr/bin/env python3
"""
测试脚本 - 验证脚手架的基本功能
"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """测试所有模块是否可以正常导入"""
    print("测试模块导入...")
    
    try:
        from pytorch_scaffold import __version__
        print(f"  ✓ pytorch_scaffold 版本: {__version__}")
        
        from pytorch_scaffold.config import Config, DEFAULT_CONFIG
        print("  ✓ config模块")
        
        from pytorch_scaffold.models import BaseModel, SimpleCNN, SimpleResNet
        print("  ✓ models模块")
        
        from pytorch_scaffold.data import get_cifar10_loaders, get_mnist_loaders, CustomDataset
        print("  ✓ data模块")
        
        from pytorch_scaffold.trainer import Trainer
        print("  ✓ trainer模块")
        
        from pytorch_scaffold.utils import (
            set_seed, get_device, count_parameters, 
            AverageMeter, accuracy, save_checkpoint, load_checkpoint
        )
        print("  ✓ utils模块")
        
        print("\n所有模块导入成功！✓")
        return True
    except Exception as e:
        print(f"\n导入失败: {e}")
        return False


def test_config():
    """测试配置管理"""
    print("\n测试配置管理...")
    
    try:
        from pytorch_scaffold.config import Config, DEFAULT_CONFIG
        
        # 测试默认配置
        config = Config(DEFAULT_CONFIG)
        batch_size = config.get('training.batch_size')
        print(f"  ✓ 读取配置: batch_size = {batch_size}")
        
        # 测试设置配置
        config.set('training.batch_size', 64)
        assert config.get('training.batch_size') == 64
        print("  ✓ 设置配置成功")
        
        # 测试YAML加载
        config_path = project_root / 'pytorch_scaffold/configs/default_config.yaml'
        if config_path.exists():
            config = Config.from_yaml(str(config_path))
            print(f"  ✓ 从YAML加载配置: {config_path.name}")
        
        print("\n配置管理测试通过！✓")
        return True
    except Exception as e:
        print(f"\n配置管理测试失败: {e}")
        return False


def test_models():
    """测试模型创建"""
    print("\n测试模型创建...")
    
    try:
        import torch
        from pytorch_scaffold.models import SimpleCNN, SimpleResNet
        from pytorch_scaffold.utils import count_parameters
        
        # 测试SimpleCNN
        model_cnn = SimpleCNN(num_classes=10, input_channels=3)
        params_cnn = count_parameters(model_cnn)
        print(f"  ✓ SimpleCNN创建成功, 参数量: {params_cnn:,}")
        
        # 测试前向传播
        x = torch.randn(2, 3, 32, 32)
        output = model_cnn(x)
        assert output.shape == (2, 10)
        print(f"  ✓ SimpleCNN前向传播成功: {output.shape}")
        
        # 测试SimpleResNet
        model_resnet = SimpleResNet(num_classes=10, num_blocks=[2, 2, 2, 2])
        params_resnet = count_parameters(model_resnet)
        print(f"  ✓ SimpleResNet创建成功, 参数量: {params_resnet:,}")
        
        output = model_resnet(x)
        assert output.shape == (2, 10)
        print(f"  ✓ SimpleResNet前向传播成功: {output.shape}")
        
        print("\n模型测试通过！✓")
        return True
    except Exception as e:
        print(f"\n模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utils():
    """测试工具函数"""
    print("\n测试工具函数...")
    
    try:
        import torch
        from pytorch_scaffold.utils import (
            set_seed, get_device, AverageMeter, accuracy
        )
        
        # 测试设置随机种子
        set_seed(42)
        print("  ✓ 设置随机种子")
        
        # 测试设备选择
        device = get_device()
        print(f"  ✓ 获取设备: {device}")
        
        # 测试AverageMeter
        meter = AverageMeter()
        meter.update(1.0, 1)
        meter.update(2.0, 1)
        assert meter.avg == 1.5
        print(f"  ✓ AverageMeter: avg = {meter.avg}")
        
        # 测试准确率计算
        output = torch.randn(4, 10)
        target = torch.randint(0, 10, (4,))
        acc = accuracy(output, target, topk=(1,))[0]
        print(f"  ✓ 准确率计算: {acc.item():.2f}%")
        
        print("\n工具函数测试通过！✓")
        return True
    except Exception as e:
        print(f"\n工具函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("PyTorch深度学习脚手架 - 功能测试")
    print("=" * 60)
    
    results = []
    
    # 运行测试
    results.append(("模块导入", test_imports()))
    results.append(("配置管理", test_config()))
    results.append(("模型创建", test_models()))
    results.append(("工具函数", test_utils()))
    
    # 显示测试结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("=" * 60)
    if all_passed:
        print("所有测试通过！✓")
        print("\n脚手架已准备就绪，可以开始使用:")
        print("  python quick_start.py")
        print("  python train_example.py")
    else:
        print("部分测试失败 ✗")
        sys.exit(1)
    print("=" * 60)


if __name__ == '__main__':
    main()
