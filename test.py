# -*- coding: utf-8 -*-
"""测试集评估：总体准确率、每类指标、混淆矩阵、正确/错误样本可视化。

用法：
    python test.py                              # 评估 config.py 里 exp_name 对应的模型
    python test.py --ckpt resnet18_cifar.pth    # 评估指定的模型存档
    python test.py --help                       # 查看全部可调参数

输出的图片文件名会带上实验名（例如 confusion_resnet18_cifar.png），
这样多个实验的结果不会互相覆盖。
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from config import CLASSES, load_config
from dataset import build_test_only
from model import classify

#matplotlib 中文显示（否则标题和坐标轴会变成方块）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    p = argparse.ArgumentParser(description="在 CIFAR-10 测试集上评估模型")
    p.add_argument("--ckpt", help="模型文件名，默认用 config.py 里的 exp_name")
    p.add_argument("--data-root", help="CIFAR-10 数据目录")
    p.add_argument("--batch-size", type=int, help="批大小")
    p.add_argument("--num-workers", type=int, help="DataLoader 子进程数")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(data_root=args.data_root, batch_size=args.batch_size,
                      num_workers=args.num_workers)
    ckpt_name = args.ckpt or f"{cfg.exp_name}.pth"
    ckpt_path = cfg.ckpt_dir / ckpt_name

    #存档不存在时给个清晰的提示，而不是抛一堆 FileNotFoundError 堆栈
    if not ckpt_path.exists():
        print(f"找不到模型存档: {ckpt_path}")
        avail = sorted(p.name for p in cfg.ckpt_dir.glob("*.pth"))
        if avail:
            print("\n可用的存档:")
            for name in avail:
                print(f"  {name}")
            print(f"\n用 --ckpt 指定，例如:\n  python test.py --ckpt {avail[0]}")
        else:
            print("checkpoints/ 目录下没有任何 .pth 文件，请先运行 train.py")
        return

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = classify(num_classes=cfg.num_classes, dropout=cfg.dropout).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"模型: {ckpt_name}  第 {ckpt['epoch']} 轮, 验证准确率 {ckpt['best_acc']:.4f}")
    saved = ckpt.get("config")
    if saved:   # 训练时把配置一起存进了存档，这里可以核对一下
        print(f"  （训练配置: epochs={saved.get('epochs')} lr={saved.get('lr')} "
              f"dropout={saved.get('dropout')} erase_p={saved.get('erase_p')}）")

    #只加载测试集，不用管那 45000 张训练图，省时间也省内存
    test_set, test_loader = build_test_only(cfg)

    #正式测试：先把全部预测收集起来，再统一算指标。
    #不能边跑边算准确率，因为每类指标和混淆矩阵需要知道"哪个样本被错分成了哪一类"
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            all_preds.append(outputs.argmax(1).cpu())   # 张量要搬到 CPU 才能转 numpy
            all_labels.append(targets)

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    print("预测结果:", len(preds), "真实标签:", len(labels))

    accuracy = (preds == labels).mean()
    print(f"总体准确率: {accuracy:.4f}")

    #混淆矩阵：行=真实类，列=预测类（参数顺序写反会得到转置的矩阵）
    cm = confusion_matrix(labels, preds)
    print(classification_report(labels, preds, target_names=CLASSES, digits=4))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, cbar=True)
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.title(f"CIFAR-10 混淆矩阵  ({ckpt_path.stem}, acc={accuracy:.4f})")
    plt.tight_layout()
    cm_path = cfg.out_dir / f"confusion_{ckpt_path.stem}.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    print(f"\n混淆矩阵已保存: {cm_path}")

    #反归一化：把归一化后的张量还原成能显示的图片
    _mean = torch.tensor(cfg.mean).view(3, 1, 1)
    _std = torch.tensor(cfg.std).view(3, 1, 1)

    def to_display(idx):
        img, _ = test_set[idx]      # test_set 无增强，每次取到的是同一张
        return (img * _std + _mean).clamp(0, 1).permute(1, 2, 0).numpy()

    def plot_samples(idxs, title, savepath, cols=4):
        """把若干张图拼成网格，标题标出真实类别(T)和预测类别(P)，对绿错红。"""
        rows = (len(idxs) + cols - 1) // cols        # 向上取整算行数
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.8))
        for ax, i in zip(np.ravel(axes), idxs):
            ax.imshow(to_display(i))
            ok = preds[i] == labels[i]
            ax.set_title(f"T: {CLASSES[labels[i]]}\nP: {CLASSES[preds[i]]}",
                         color="green" if ok else "red", fontsize=9)
            ax.axis("off")
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"已保存: {savepath}")

    #随机抽取（固定种子，保证可复现），避免只看到某个 batch 里的样本
    rng = np.random.default_rng(42)
    wrong_idx = np.flatnonzero(preds != labels)      # 错分样本的下标
    right_idx = np.flatnonzero(preds == labels)      # 正确样本的下标
    n_show = 8

    plot_samples(rng.choice(right_idx, n_show, replace=False),
                 "正确预测样本 (T=真实类别, P=预测类别)",
                 cfg.out_dir / f"correct_{ckpt_path.stem}.png")
    plot_samples(rng.choice(wrong_idx, n_show, replace=False),
                 "错误预测样本 (T=真实类别, P=预测类别)",
                 cfg.out_dir / f"wrong_{ckpt_path.stem}.png")

    #分析错误样本：统计最容易被混淆的类别对，为后续优化提供方向
    print(f"\n错误样本共 {len(wrong_idx)} 张，最常见的混淆（真实 → 预测）:")
    pairs = {}
    for t, p in zip(labels[wrong_idx], preds[wrong_idx]):
        pairs[(t, p)] = pairs.get((t, p), 0) + 1
    for (t, p), c in sorted(pairs.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {CLASSES[t]:<11} → {CLASSES[p]:<11} {c:>4} 张")

    plt.show()


#必须加这个保护：Windows 下 DataLoader 的 num_workers>0 会用 spawn 启动子进程，
#子进程会重新 import 本文件（身份是 __mp_main__）。没有这层保护的话，
#子进程会把整个脚本再跑一遍，导致无限递归报错。
if __name__ == "__main__":
    main()
