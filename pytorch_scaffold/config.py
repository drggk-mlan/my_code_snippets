"""
配置管理模块
"""
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """配置类，用于管理训练配置"""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        初始化配置
        
        Args:
            config_dict: 配置字典
        """
        self.config = config_dict or {}
        
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """
        从YAML文件加载配置
        
        Args:
            yaml_path: YAML文件路径
            
        Returns:
            Config对象
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，支持点号分隔的嵌套键
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """
        设置配置值
        
        Args:
            key: 配置键，支持点号分隔的嵌套键
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def __repr__(self):
        return f"Config({self.config})"


# 默认配置
DEFAULT_CONFIG = {
    'model': {
        'name': 'resnet18',
        'num_classes': 10,
    },
    'training': {
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'momentum': 0.9,
    },
    'data': {
        'num_workers': 4,
        'pin_memory': True,
    },
    'optimizer': {
        'type': 'adam',
    },
    'scheduler': {
        'type': 'step',
        'step_size': 30,
        'gamma': 0.1,
    },
    'logging': {
        'log_interval': 10,
        'save_dir': 'checkpoints',
    },
}
