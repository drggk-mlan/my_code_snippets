#!/usr/bin/env python3
"""
基础验证脚本 - 不依赖PyTorch，仅验证代码结构
"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def verify_structure():
    """验证目录结构"""
    print("验证目录结构...")
    
    required_files = [
        'pytorch_scaffold/__init__.py',
        'pytorch_scaffold/config.py',
        'pytorch_scaffold/trainer.py',
        'pytorch_scaffold/models/__init__.py',
        'pytorch_scaffold/data/__init__.py',
        'pytorch_scaffold/utils/__init__.py',
        'pytorch_scaffold/configs/default_config.yaml',
        'requirements.txt',
        'train_example.py',
        'quick_start.py',
        '.gitignore',
        'README.md',
    ]
    
    missing = []
    for file in required_files:
        file_path = project_root / file
        if file_path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (缺失)")
            missing.append(file)
    
    if missing:
        print(f"\n缺失 {len(missing)} 个文件")
        return False
    else:
        print(f"\n所有 {len(required_files)} 个文件都存在！✓")
        return True


def verify_syntax():
    """验证Python语法"""
    print("\n验证Python语法...")
    
    import py_compile
    
    python_files = [
        'pytorch_scaffold/__init__.py',
        'pytorch_scaffold/config.py',
        'pytorch_scaffold/trainer.py',
        'pytorch_scaffold/models/__init__.py',
        'pytorch_scaffold/data/__init__.py',
        'pytorch_scaffold/utils/__init__.py',
        'train_example.py',
        'quick_start.py',
        'test_scaffold.py',
    ]
    
    errors = []
    for file in python_files:
        file_path = project_root / file
        try:
            py_compile.compile(str(file_path), doraise=True)
            print(f"  ✓ {file}")
        except py_compile.PyCompileError as e:
            print(f"  ✗ {file}: {e}")
            errors.append(file)
    
    if errors:
        print(f"\n{len(errors)} 个文件有语法错误")
        return False
    else:
        print(f"\n所有 {len(python_files)} 个Python文件语法正确！✓")
        return True


def verify_config():
    """验证配置文件"""
    print("\n验证配置系统...")
    
    try:
        from pytorch_scaffold.config import Config, DEFAULT_CONFIG
        
        # 测试默认配置
        config = Config(DEFAULT_CONFIG)
        assert 'model' in config.config
        assert 'training' in config.config
        print("  ✓ 默认配置结构正确")
        
        # 测试配置读取
        batch_size = config.get('training.batch_size')
        assert batch_size is not None
        print(f"  ✓ 配置读取正常: batch_size = {batch_size}")
        
        # 测试配置设置
        config.set('training.test_value', 12345)
        assert config.get('training.test_value') == 12345
        print("  ✓ 配置设置正常")
        
        # 测试YAML加载
        config_path = project_root / 'pytorch_scaffold/configs/default_config.yaml'
        config = Config.from_yaml(str(config_path))
        assert config.config is not None
        print(f"  ✓ YAML配置加载正常")
        
        return True
    except Exception as e:
        print(f"  ✗ 配置验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_readme():
    """验证README文档"""
    print("\n验证README文档...")
    
    readme_path = project_root / 'README.md'
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_sections = [
            'PyTorch深度学习脚手架',
            '特性',
            '安装',
            '快速开始',
            '核心组件',
        ]
        
        missing_sections = []
        for section in required_sections:
            if section in content:
                print(f"  ✓ 包含章节: {section}")
            else:
                print(f"  ✗ 缺少章节: {section}")
                missing_sections.append(section)
        
        if missing_sections:
            return False
        
        print(f"\nREADME文档完整！✓")
        return True
    except Exception as e:
        print(f"  ✗ README验证失败: {e}")
        return False


def main():
    """运行所有验证"""
    print("=" * 60)
    print("PyTorch深度学习脚手架 - 结构验证")
    print("=" * 60 + "\n")
    
    results = []
    
    # 运行验证
    results.append(("目录结构", verify_structure()))
    results.append(("Python语法", verify_syntax()))
    results.append(("配置系统", verify_config()))
    results.append(("README文档", verify_readme()))
    
    # 显示验证结果
    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("=" * 60)
    if all_passed:
        print("✓ 所有验证通过！")
        print("\nPyTorch深度学习脚手架已成功创建！")
        print("\n包含以下核心功能:")
        print("  • 灵活的配置管理系统")
        print("  • 完整的训练器类")
        print("  • 示例模型 (SimpleCNN, SimpleResNet)")
        print("  • 数据加载工具")
        print("  • 实用工具函数")
        print("  • TensorBoard集成")
        print("  • 检查点管理")
        print("\n使用方法:")
        print("  1. 安装依赖: pip install -r requirements.txt")
        print("  2. 快速开始: python quick_start.py")
        print("  3. 完整示例: python train_example.py")
        print("  4. 查看文档: 阅读 README.md")
    else:
        print("✗ 部分验证失败")
        sys.exit(1)
    print("=" * 60)


if __name__ == '__main__':
    main()
