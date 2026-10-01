# -*- coding: utf-8 -*-
"""单张/批量图片预测。

用法：
    python predict.py                              # 扫描 predict_picture/ 目录
    python predict.py 猫.jpg 狗.png                 # 指定若干张图片
    python predict.py C:\\Users\\me\\Pictures       # 扫描整个目录（递归）
    python predict.py --ckpt resnet18_cifar.pth    # 指定模型
    python predict.py --help                       # 查看全部可调参数
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from config import BASE_DIR, CLASSES, load_config
from model import classify

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


def parse_args():
    p = argparse.ArgumentParser(description="用训练好的模型预测图片类别")
    p.add_argument("images", nargs="*",
                   help="图片路径或目录；不传则用 predict_picture/ 目录")
    p.add_argument("--ckpt", help="模型文件名，默认用 config.py 里的 exp_name")
    p.add_argument("--topk", type=int, default=5, help="显示概率最高的前几个类别（默认 5）")
    return p.parse_args()


def collect_paths(args):
    """把命令行参数展开成图片路径列表（目录会被递归扫描）。"""
    paths = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            paths += sorted(q for q in p.rglob("*") if q.suffix.lower() in IMG_EXTS)
        else:
            paths.append(p)
    return paths


def build_preprocess(cfg):
    """预测用的预处理，必须和测试集完全一致（只多一步 Resize）。

    外部图片尺寸任意，而模型只认 32×32，所以要缩放。

    这里用"拉伸"而不是"保持长宽比 + 中心裁剪"：实测宽扁图片(200×80)拉伸能判对，
    而裁剪会丢掉左右内容导致判错；窄高图片(80×200)则相反。两者互有胜负，
    拉伸更简单且不丢内容，所以保留拉伸。
    """
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.mean, cfg.std),
    ])


def predict_one(model, img_path, preprocess, device, topk):
    """预测单张图片，打印 top-k 概率，返回预测类别名（失败返回 None）。"""
    p = Path(img_path)
    if not p.exists():
        print(f"\n跳过（文件不存在）: {img_path}")
        return None
    try:
        #convert("RGB") 不能省：灰度图是 1 通道、带透明通道的 PNG 是 4 通道，
        #直接送进 Conv2d(3, ...) 会报维度错误
        img = Image.open(p).convert("RGB")
    except Exception as e:
        print(f"\n跳过（无法读取图片）: {img_path}  {type(e).__name__}: {e}")
        return None

    #unsqueeze(0) 补一个 batch 维：模型要 [N,C,H,W]，单张图是 [C,H,W]
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        #模型输出的是 logits（原始分数），要过 softmax 才是概率
        probs = model(x).softmax(1)[0]

    top = probs.topk(topk)
    print(f"\n图片: {img_path}  原始尺寸: {img.size}")
    for rank, (prob, idx) in enumerate(zip(top.values, top.indices), 1):
        bar = "#" * int(prob.item() * 40)
        print(f"  {rank}. {CLASSES[idx]:<12} {prob.item():>7.4f}  {bar}")
    return CLASSES[top.indices[0]]


def main():
    args = parse_args()
    cfg = load_config()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ckpt_name = args.ckpt or f"{cfg.exp_name}.pth"
    ckpt_path = cfg.ckpt_dir / ckpt_name
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = classify(num_classes=cfg.num_classes, dropout=cfg.dropout).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"模型: {ckpt_name}  第 {ckpt['epoch']} 轮, 验证准确率 {ckpt['best_acc']:.4f}")

    paths = collect_paths(args.images or [BASE_DIR / "predict_picture"])
    if not paths:
        print("没有找到任何图片")
        return
    print(f"共 {len(paths)} 张图片")

    preprocess = build_preprocess(cfg)
    done = 0
    for p in paths:
        if predict_one(model, p, preprocess, device, args.topk) is not None:
            done += 1
    print(f"\n完成 {done}/{len(paths)} 张")


if __name__ == "__main__":
    main()
