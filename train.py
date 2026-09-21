
from dataset import *
from model import classify
from utils import set_seed
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

#matplotlib 中文显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

#定义训练设备
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#matplotlib 中文显示（否则标题和坐标轴会变成方块）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def evaluate(model,loader,criterion,device):
    model.eval()
    running_loss,correct,total=0.0,0,0
    with torch.no_grad():
        for data in loader:
            imgs, targets = data
            imgs,targets=imgs.to(device),targets.to(device)
            outputs=model(imgs)
            loss=criterion(outputs,targets)#计算出一个batch的loss
            running_loss += loss.item()*imgs.size(0)#全部的损失值
            correct+=(outputs.argmax(1) == targets).sum().item()#统计预测正确的数量
            total+=imgs.size(0)
    return running_loss/total,correct/total


def train_one_epoch(model,loader,criterion,optimizer,device):
    model.train()
    running_loss,correct,total=0.0,0,0
    for data in loader:
        imgs,targets=data
        imgs, targets = imgs.to(device), targets.to(device)
        outputs=model(imgs)
        loss=criterion(outputs,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*imgs.size(0)
        correct+=(outputs.argmax(1) == targets).sum().item()
        total+=imgs.size(0)
    return running_loss/total,correct/total


def main():
    set_seed(42)          #固定随机种子，保证结果可复现

    #基本参数
    batch_size    = 128
    epochs        = 50
    lr            = 1e-3
    weight_decay  = 5e-4

    #创建网络模型
    model = classify().to(device)
    #定义损失函数
    criterion = torch.nn.CrossEntropyLoss()
    #定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    #定义学习器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)

    writer = SummaryWriter("logs")

    #选择和保存最佳模型
    ckpt_dir = Path(__file__).parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)      #确保目录不存在时不会报错
    best_path = ckpt_dir / "best_model.pth"
    best_acc, best_epoch = 0.0, 0
    #创建字典,用来保存每一轮的epoch的训练/验证损失,准确率
    history={"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[]}

    #开始训练每一轮
    for epoch in range(1,epochs+1):
        train_loss,train_acc =train_one_epoch(model,train_dataloader,criterion,optimizer,device)
        val_loss,val_acc = evaluate(model,val_dataloader,criterion,device)
        scheduler.step()
        #模型选择
        if val_acc>best_acc:
            best_acc,best_epoch=val_acc,epoch
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc":best_acc
            },best_path)
            mark="<-- best"
        else:
            mark=""

        print(f"Epoch [{epoch:>3}/{epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e}{mark}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)
        #将本轮算出的loss和acc保存到history字典中
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    print(f"最佳验证准确率 {best_acc:.4f}（第 {best_epoch} 轮）")

    #重新加载验证集表现最好的模型，供 test.py 使用
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"已加载第 {ckpt['epoch']} 轮的模型，val_acc={ckpt['best_acc']:.4f}")

    writer.close()
    #绘制loss acc 图像
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4))
    ep = range(1,len(history["train_loss"])+1)#生成横坐标   训练轮数
    #绘制第一个图 Loss
    ax1.plot(ep,history["train_loss"],label="Train Loss")
    ax1.plot(ep,history["val_loss"],label="Val Loss")
    ax1.set(xlabel="Epoch", ylabel="Loss", title="Loss")
    ax1.legend()
    #绘制第二图 Accuracy
    ax2.plot(ep,history["train_acc"],label="Train Accuracy")
    ax2.plot(ep,history["val_acc"],label="Val Accuracy")
    ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    ax2.legend()
    #保存绘制好的图片到outputs中
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    print(f"训练曲线已保存: {out_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()