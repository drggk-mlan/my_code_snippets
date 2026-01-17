"""
工具函数模块
"""
import random
import numpy as np
import torch
from pathlib import Path


def set_seed(seed: int = 42):
    """
    设置随机种子以确保可重复性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = None) -> torch.device:
    """
    获取计算设备
    
    Args:
        device: 设备名称 ('cpu', 'cuda', 'cuda:0'等)
        
    Returns:
        torch.device对象
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)


def count_parameters(model: torch.nn.Module) -> int:
    """
    计算模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state: dict, save_dir: str, filename: str = 'checkpoint.pth'):
    """
    保存模型检查点
    
    Args:
        state: 包含模型状态的字典
        save_dir: 保存目录
        filename: 文件名
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(state, save_path / filename)


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer = None) -> dict:
    """
    加载模型检查点
    
    Args:
        checkpoint_path: 检查点文件路径
        model: PyTorch模型
        optimizer: 优化器（可选）
        
    Returns:
        包含训练状态的字典
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


class AverageMeter:
    """计算并存储平均值和当前值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置所有统计信息"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        """
        更新统计信息
        
        Args:
            val: 新值
            n: 样本数量
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    计算top-k准确率
    
    Args:
        output: 模型输出 (batch_size, num_classes)
        target: 目标标签 (batch_size,)
        topk: top-k值的元组
        
    Returns:
        top-k准确率列表
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
