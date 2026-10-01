# -*- coding: utf-8 -*-
"""数据加载：下载 CIFAR-10、预处理、划分训练/验证集、构建 DataLoader。

所有可调项都在 config.py 里，本文件不硬编码任何参数。

构建逻辑写成了函数而不是在导入时直接执行，原因：
命令行参数要在"建数据"之前生效。如果导入 dataset 时就建好了 DataLoader，
那之后再改配置就没用了。
"""

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from config import load_config


def build_transforms(cfg):
    """构建预处理管道，返回 (train_tf, eval_tf)。

    训练集多几重随机增强；验证和测试集只做 ToTensor + Normalize，
    因为评估必须可复现，不能带随机性。
    """
    # 顺序不能乱：
    #   RandomCrop / RandomHorizontalFlip 操作的是 PIL 图片 → 必须在 ToTensor 之前
    #   RandomErasing 操作的是张量，且 value=0 在归一化空间里等于"数据集均值色"
    #     → 必须放在 Normalize 之后（放前面的话 0 就是纯黑，会引入不自然的黑块）
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=cfg.crop_padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cfg.mean, cfg.std),
        transforms.RandomErasing(p=cfg.erase_p, scale=(0.02, 0.2),
                                 ratio=(0.3, 3.3), value=0),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.mean, cfg.std),
    ])
    return train_tf, eval_tf


def build_datasets(cfg=None):
    """下载/加载 CIFAR-10 并划分训练集与验证集，返回 (train_set, val_set, test_set)。

    这里建了两个 train=True 的数据集对象（train_full / val_full），
    它们读的是同一份磁盘数据，只有 transform 不同（一个带增强，一个不带）。
    这样才能让验证集避开随机增强 —— 如果只建一个对象再 random_split，
    两个子集会共用同一个 transform，验证集也会被随机裁剪翻转，指标就会抖动。
    """
    cfg = cfg or load_config()
    train_tf, eval_tf = build_transforms(cfg)

    train_full = torchvision.datasets.CIFAR10(
        root=str(cfg.data_root), train=True, download=True, transform=train_tf)
    val_full = torchvision.datasets.CIFAR10(
        root=str(cfg.data_root), train=True, download=True, transform=eval_tf)
    test_set = torchvision.datasets.CIFAR10(
        root=str(cfg.data_root), train=False, download=True, transform=eval_tf)

    # 用固定种子的独立 generator 打乱索引，保证每次运行划分一致。
    # 注意这个种子只管"怎么切分"，管不到模型初始化和 shuffle，
    # 后者由 utils.set_seed() 负责，两者都要有。
    g = torch.Generator().manual_seed(cfg.split_seed)
    indices = torch.randperm(len(train_full), generator=g)

    n_val = cfg.val_size
    train_set = Subset(train_full, indices[:-n_val])   # 其余全部用于训练
    val_set = Subset(val_full, indices[-n_val:])       # 末尾 n_val 张做验证
    return train_set, val_set, test_set


def make_loader(dataset, cfg, shuffle):
    """按配置给数据集套一个 DataLoader。"""
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        # num_workers=0 时不能开 persistent_workers，会直接报错
        persistent_workers=cfg.num_workers > 0,
    )


def build_loaders(cfg=None):
    """构建三个 DataLoader，返回 (train_loader, val_loader, test_loader)。

    shuffle 的取值：训练集 True（每轮重新打乱顺序），验证/测试集 False
    （评估只看整体指标，固定顺序才能让两次结果可比）。
    """
    cfg = cfg or load_config()
    train_set, val_set, test_set = build_datasets(cfg)
    return (make_loader(train_set, cfg, shuffle=True),
            make_loader(val_set, cfg, shuffle=False),
            make_loader(test_set, cfg, shuffle=False))


def build_test_only(cfg=None):
    """只构建测试集和它的 DataLoader，返回 (test_set, test_loader)。

    test.py 专用：不用加载 45000 张训练图，省时间和内存。
    """
    cfg = cfg or load_config()
    _, eval_tf = build_transforms(cfg)
    test_set = torchvision.datasets.CIFAR10(
        root=str(cfg.data_root), train=False, download=True, transform=eval_tf)
    return test_set, make_loader(test_set, cfg, shuffle=False)


if __name__ == "__main__":
    #自检：python dataset.py
    cfg = load_config()
    train_loader, val_loader, test_loader = build_loaders(cfg)
    print(f"设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"训练/验证/测试: {len(train_loader.dataset)} / "
          f"{len(val_loader.dataset)} / {len(test_loader.dataset)}")
    imgs, labels = next(iter(train_loader))
    print(f"一个 batch: {tuple(imgs.shape)} {imgs.dtype} | 标签 {tuple(labels.shape)}")
    print(f"均值: {imgs.mean():.3f}（负数说明 RandomCrop 的黑边填充生效了）")
    a, _ = next(iter(train_loader))
    b, _ = next(iter(train_loader))
    print(f"train shuffle 生效: {not torch.equal(a, b)}")
