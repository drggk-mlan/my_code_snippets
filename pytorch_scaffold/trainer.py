"""
训练器模块
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from typing import Optional

from ..utils import AverageMeter, accuracy, save_checkpoint, get_device


class Trainer:
    """训练器类"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 criterion: nn.Module = None,
                 optimizer: torch.optim.Optimizer = None,
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 device: str = None,
                 save_dir: str = 'checkpoints',
                 log_interval: int = 10,
                 use_tensorboard: bool = True):
        """
        初始化训练器
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 设备
            save_dir: 检查点保存目录
            log_interval: 日志打印间隔
            use_tensorboard: 是否使用TensorBoard
        """
        self.device = get_device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or torch.optim.Adam(model.parameters())
        self.scheduler = scheduler
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        
        # TensorBoard
        self.writer = None
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=self.save_dir / 'runs')
        
        # 训练状态
        self.epoch = 0
        self.best_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 记录指标
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), data.size(0))
            top1.update(acc1.item(), data.size(0))
            
            # 更新进度条
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'Loss': f'{losses.avg:.4f}',
                    'Acc': f'{top1.avg:.2f}%'
                })
            
            # TensorBoard记录
            if self.writer:
                step = self.epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), step)
                self.writer.add_scalar('Train/Accuracy', acc1.item(), step)
        
        return losses.avg, top1.avg
    
    def validate(self):
        """验证模型"""
        if self.val_loader is None:
            return None, None
        
        self.model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validating'):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                acc1 = accuracy(output, target, topk=(1,))[0]
                losses.update(loss.item(), data.size(0))
                top1.update(acc1.item(), data.size(0))
        
        return losses.avg, top1.avg
    
    def fit(self, num_epochs: int, save_best: bool = True):
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
            save_best: 是否保存最佳模型
        """
        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证
            if self.val_loader is not None:
                val_loss, val_acc = self.validate()
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)
                
                print(f'Epoch {self.epoch}/{num_epochs}:')
                print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                
                # TensorBoard记录
                if self.writer:
                    self.writer.add_scalar('Val/Loss', val_loss, self.epoch)
                    self.writer.add_scalar('Val/Accuracy', val_acc, self.epoch)
                
                # 保存最佳模型
                if save_best and val_acc > self.best_acc:
                    self.best_acc = val_acc
                    self.save_checkpoint('best_model.pth')
                    print(f'  Saved best model with accuracy: {val_acc:.2f}%')
            else:
                print(f'Epoch {self.epoch}/{num_epochs}:')
                print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            
            # 学习率调整
            if self.scheduler:
                self.scheduler.step()
                if self.writer:
                    self.writer.add_scalar('Learning_Rate', 
                                         self.optimizer.param_groups[0]['lr'], 
                                         self.epoch)
            
            # 定期保存检查点
            if self.epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{self.epoch}.pth')
        
        # 保存最终模型
        self.save_checkpoint('final_model.pth')
        
        if self.writer:
            self.writer.close()
        
        print('Training completed!')
    
    def save_checkpoint(self, filename: str):
        """
        保存检查点
        
        Args:
            filename: 文件名
        """
        state = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
        }
        
        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        
        save_checkpoint(state, self.save_dir, filename)
    
    def load_checkpoint(self, checkpoint_path: str, map_location: str = None):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
            map_location: 设备映射（可选）
        """
        if map_location is None:
            map_location = str(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_acc = checkpoint.get('best_acc', 0.0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accs = checkpoint.get('val_accs', [])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f'Loaded checkpoint from epoch {self.epoch}')
