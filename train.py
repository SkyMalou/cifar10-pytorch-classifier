# -*- coding: utf-8 -*-
"""训练脚本。

用法：
    python train.py                                # 用 config.py 里的默认配置
    python train.py --exp my_exp --epochs 50       # 命令行覆盖单个参数
    python train.py --lr 3e-4 --no-amp             # 关掉混合精度
    python train.py --help                         # 查看全部可调参数

优先级：命令行参数 > config.py 里的默认值
"""

import argparse
import dataclasses

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from config import load_config
from dataset import build_loaders
from model import classify
from utils import set_seed

#matplotlib 中文显示（否则标题和坐标轴会变成方块）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

#定义训练设备
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def parse_args():
    """命令行参数。不传的项为 None，表示"沿用 config.py 的默认值"。"""
    p = argparse.ArgumentParser(description="训练 CIFAR-10 分类模型")
    p.add_argument("--exp", help="实验名，决定日志和存档的文件名")
    p.add_argument("--epochs", type=int, help="训练轮数")
    p.add_argument("--lr", type=float, help="学习率")
    p.add_argument("--weight-decay", type=float, help="权重衰减")
    p.add_argument("--batch-size", type=int, help="批大小")
    p.add_argument("--num-workers", type=int, help="DataLoader 子进程数")
    p.add_argument("--dropout", type=float, help="分类头 Dropout 概率")
    p.add_argument("--label-smoothing", type=float, help="标签平滑系数")
    p.add_argument("--erase-p", type=float, help="RandomErasing 概率，0 表示关闭")
    p.add_argument("--seed", type=int, help="随机种子")
    p.add_argument("--data-root", help="CIFAR-10 数据目录")
    p.add_argument("--no-amp", action="store_true", help="关闭混合精度")
    return p.parse_args()


def evaluate(model, loader, criterion, device, use_amp=False):
    """在验证集/测试集上算平均损失和准确率。返回 (loss, acc)。"""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for data in loader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)
            # 混合精度：用 bf16 做前向。bf16 的数值范围和 fp32 一样，
            # 所以不需要 GradScaler，代码比 fp16 方案简单且不会梯度下溢
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(imgs)
                loss = criterion(outputs, targets)
            # loss.item() 是"这一批的平均损失"，要乘样本数再累加，
            # 否则最后一批只有几十张，权重会被高估
            running_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += imgs.size(0)
    return running_loss / total, correct / total


def train_one_epoch(model, loader, criterion, optimizer, device, use_amp=False):
    """训练一轮，返回 (平均损失, 准确率)。"""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for data in loader:
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            outputs = model(imgs)
            loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total


def plot_curves(history, savepath):
    """画训练/验证的损失和准确率曲线，存到 outputs/。"""
    ep = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(ep, history["train_loss"], label="Train Loss")
    ax1.plot(ep, history["val_loss"], label="Val Loss")
    ax1.set(xlabel="Epoch", ylabel="Loss", title="Loss")
    ax1.legend()

    ax2.plot(ep, history["train_acc"], label="Train Accuracy")
    ax2.plot(ep, history["val_acc"], label="Val Accuracy")
    ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    ax2.legend()

    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    print(f"训练曲线已保存: {savepath}")


def main():
    args = parse_args()
    # 值为 None 的项会被 load_config 忽略，所以可以直接把 argparse 结果传进去
    cfg = load_config(
        exp_name=args.exp,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        erase_p=args.erase_p,
        seed=args.seed,
        data_root=args.data_root,
        use_amp=False if args.no_amp else None,
    )

    set_seed(cfg.seed)                              # 固定随机种子，保证结果可复现
    use_amp = cfg.use_amp and device.type == "cuda"  # 混合精度只在 GPU 上有意义

    print("=" * 62)
    print(f"实验名      : {cfg.exp_name}")
    print(f"设备        : {device} | 混合精度: {'开启 (bf16)' if use_amp else '关闭'}")
    print(f"epochs      : {cfg.epochs} | lr: {cfg.lr} | weight_decay: {cfg.weight_decay}")
    print(f"batch_size  : {cfg.batch_size} | num_workers: {cfg.num_workers}")
    print(f"dropout     : {cfg.dropout} | label_smoothing: {cfg.label_smoothing} "
          f"| erase_p: {cfg.erase_p}")
    print(f"数据目录    : {cfg.data_root}")
    print(f"存档        : {cfg.ckpt_path}")
    print(f"日志        : {cfg.log_path}")
    print("=" * 62)

    train_loader, val_loader, _ = build_loaders(cfg)   # 第三个是测试集，训练阶段不用

    model = classify(num_classes=cfg.num_classes, dropout=cfg.dropout).to(device)
    #label_smoothing：不再要求模型对正确类输出 100% 置信度，改为 0.9，
    #其余各类平分 0.1。能抑制模型过度自信，提升泛化
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    writer = SummaryWriter(str(cfg.log_path))
    best_acc, best_epoch = 0.0, 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_amp)
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, use_amp)
        scheduler.step()

        #模型选择：只在验证准确率变好时才覆盖存档（用严格大于，
        #打平时保留更早的那一轮，避免无意义的反复写入）
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": best_acc,
                # 把配置一起存下来，以后能回溯"这个模型是用什么参数训的"
                "config": dataclasses.asdict(cfg),
            }, cfg.ckpt_path)
            mark = "  <-- best"
        else:
            mark = ""

        print(f"Epoch [{epoch:>3}/{cfg.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e}{mark}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    print(f"\n最佳验证准确率 {best_acc:.4f}（第 {best_epoch} 轮）")

    #训练结束时 model 里装的是"最后一轮"的权重，不一定是最好那轮，
    #所以重新加载一次，保证后续 test.py 拿到的是最佳模型
    ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"已加载第 {ckpt['epoch']} 轮的模型，val_acc={ckpt['best_acc']:.4f}")

    writer.close()
    plot_curves(history, cfg.out_dir / f"curves_{cfg.exp_name}.png")


if __name__ == "__main__":
    main()
