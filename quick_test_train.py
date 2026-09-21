# -*- coding: utf-8 -*-
#简单自检：确认数据、模型、训练、评估、保存都能跑通。
#只跑几个 batch，几秒钟出结果，不做完整训练。


from pathlib import Path

import torch

from dataset import train_dataloader, val_dataloader, test_dataloader
from model import classify

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

# ---------------------------------------------------------------- 1. 数据
print("\n[1] 数据")
print("  训练/验证/测试:", len(train_dataloader.dataset), len(val_dataloader.dataset), len(test_dataloader.dataset))
x, y = next(iter(train_dataloader))
print("  一个 batch:", tuple(x.shape), x.dtype, "| 标签", tuple(y.shape))
print("  均值: %.3f  (负数说明 RandomCrop 的黑边填充生效了)" % x.mean())

# ---------------------------------------------------------------- 2. 模型
print("\n[2] 模型")
model = classify().to(device)
out = model(x.to(device))
print("  输出:", tuple(out.shape), "(期望 (128, 10))")
print("  参数量:", f"{sum(p.numel() for p in model.parameters()):,}")

# ---------------------------------------------------------------- 3. 训练
print("\n[3] 训练 3 个 batch")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
model.train()
for i, (imgs, targets) in enumerate(train_dataloader):
    if i == 3:
        break
    imgs, targets = imgs.to(device), targets.to(device)
    loss = criterion(model(imgs), targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"  batch {i + 1}: loss={loss.item():.4f}   (初始应约 2.30)")

# ---------------------------------------------------------------- 4. 评估
print("\n[4] 验证集评估 5 个 batch")
model.eval()
loss_sum, correct, total = 0.0, 0, 0
with torch.no_grad():
    for i, (imgs, targets) in enumerate(val_dataloader):
        if i == 5:
            break
        imgs, targets = imgs.to(device), targets.to(device)
        out = model(imgs)
        loss_sum += criterion(out, targets).item() * len(imgs)
        correct += (out.argmax(1) == targets).sum().item()
        total += len(imgs)
print(f"  loss={loss_sum / total:.4f}  acc={correct / total:.4f}   (随机猜约 0.10)")

# ---------------------------------------------------------------- 5. 保存/加载
print("\n[5] 保存 / 加载")
tmp = Path("quick_test.pth")
torch.save({"model_state": model.state_dict()}, tmp)
fresh = classify().to(device)
fresh.load_state_dict(torch.load(tmp, map_location=device)["model_state"])
same = all(torch.equal(a, b) for a, b in zip(model.parameters(), fresh.parameters()))
print("  参数一致:", same)
tmp.unlink()#删除临时文件

print("\n自检结束。以上都正常说明数据和模型都没问题，可以跑 train.py 正式训练了。")
