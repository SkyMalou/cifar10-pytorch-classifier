import torch
import numpy as np
from pathlib import Path
from dataset import test_set, test_dataloader, mean, std
from model import classify
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


#matplotlib 中文显示（否则标题和坐标轴会变成方块）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

#输出目录
out_dir = Path(__file__).parent / "outputs"
out_dir.mkdir(exist_ok=True)

#脚本所在目录，用来定位模型（避免依赖运行时的工作目录）
BASE_DIR = Path(__file__).parent

#加载出最佳模型
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
ckpt =torch.load(BASE_DIR/"checkpoints"/"best_model.pth",map_location=device,weights_only=False)
model =classify().to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"第 {ckpt['epoch']} 轮, 验证准确率 {ckpt['best_acc']:.4f}")

#正式测试
all_preds,all_label = [],[]#存放预测和标签
with torch.no_grad():
    for imgs ,targets in test_dataloader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        all_preds.append(outputs.argmax(1).cpu())
        all_label.append(targets)

preds = torch.cat(all_preds).numpy()
labels =torch.cat(all_label).numpy()

#检测preds 与 labels 长度是否一致
print("预测结果:", len(preds), "真实标签:", len(labels))

#计算准确率
accuracy = (preds==labels).mean()#总体准确率
print(f"总体准确率: {accuracy:.4f}")
#计算每一类的准确率
cm = confusion_matrix(labels, preds)                # 10×10，行=真实类，列=预测类
classes=test_set.classes
print(classification_report(labels, preds, target_names=classes, digits=4))

#混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes, cbar=True)
plt.xlabel("预测类别")
plt.ylabel("真实类别")
plt.title("CIFAR-10 混淆矩阵")
plt.tight_layout()
plt.savefig(out_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
print(f"\n混淆矩阵已保存: {out_dir / 'confusion_matrix.png'}")


#正确样本和错误样本
#反归一化，不然无法输出图片
_mean=torch.tensor(mean).view(3,1,1)
_std=torch.tensor(std).view(3,1,1)

def to_display(idx):
    img, _ = test_set[idx]          # test_set 无增强，每次取到的是同一张
    return (img * _std + _mean).clamp(0, 1).permute(1, 2, 0).numpy()

#绘图函数
def plot_samples(idxs,title,savepath,cols=4):#一行四张图片
    rows=(len(idxs)+cols -1)//cols   #一共要多少行
    fig,axes =plt.subplots(rows,cols,figsize=(cols*2.2,rows*2.8))
    for ax,i in zip(np.ravel(axes),idxs):
        ax.imshow(to_display(i))
        ok=preds[i] == labels[i]
        ax.set_title(f"T: {classes[labels[i]]}\nP: {classes[preds[i]]}",
                     color="green" if ok else "red", fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(savepath,dpi=150,bbox_inches="tight")
    print(f"已保存: {savepath}")
#采样
#随机抽取（固定种子，保证可复现），避免只看到某个 batch 里的样本
rng = np.random.default_rng(42)
wrong_idx = np.flatnonzero(preds != labels)      # 错分样本的下标
right_idx = np.flatnonzero(preds == labels)      # 正确样本的下标
n_show = 8   #随机选8张  肉眼观察错哪里了

plot_samples(rng.choice(right_idx, n_show, replace=False),
             "正确预测样本 (T=真实类别, P=预测类别)",
             out_dir / "correct_samples.png")
plot_samples(rng.choice(wrong_idx, n_show, replace=False),
             "错误预测样本 (T=真实类别, P=预测类别)",
             out_dir / "wrong_samples.png")

#分析错误的样本
print(f"\n错误样本共 {len(wrong_idx)} 张，最常见的混淆（真实 → 预测）:")
pairs = {}
for t, p in zip(labels[wrong_idx], preds[wrong_idx]):
    pairs[(t, p)] = pairs.get((t, p), 0) + 1
for (t, p), c in sorted(pairs.items(), key=lambda kv: -kv[1])[:5]:   #只要错误率最高的5个种类
    print(f"  {classes[t]:<11} → {classes[p]:<11} {c:>4} 张")
#这样我们可以找到模型的问题出现在哪里，为优化模型提供方向
plt.show()