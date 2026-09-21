import torchvision
import torch
from torch.utils.data import DataLoader,Subset
from torchvision import transforms
import matplotlib.pyplot as plt

mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)
#定义数据预处理
train_transform=transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std)
])
test_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std)
])
#下载构建数据
# 下载 + 构建，注意两个对象的 transform 不同
train_full = torchvision.datasets.CIFAR10(
    root="C:\\PythonProject\\分类数据集\\data",
    train=True, download=True, transform=train_transform)
val_full = torchvision.datasets.CIFAR10(
    root="C:\\PythonProject\\分类数据集\\data",
    train=True, download=True, transform=test_transform)
test_set = torchvision.datasets.CIFAR10(
    root="C:\\PythonProject\\分类数据集\\data",
    train=False, download=True, transform=test_transform)

# 固定种子，保证每次运行划分一致（可复现）
g = torch.Generator().manual_seed(42)
indices = torch.randperm(50000, generator=g)
#切割数据分成增强集   不增强的
train_set = Subset(train_full, indices[:45000])    #用于训练
val_set   = Subset(val_full,   indices[45000:])    #寻找最佳模型
#利用dataloader来处理数据
train_dataloader = DataLoader(train_set,batch_size=128, shuffle=True,num_workers=0, pin_memory=True)
val_dataloader   = DataLoader(val_set,batch_size=128, shuffle=False,num_workers=0, pin_memory=True)
test_dataloader  = DataLoader(test_set,batch_size=128, shuffle=False,num_workers=0, pin_memory=True)

#测试
if __name__ == "__main__":
    print(len(train_set), len(val_set), len(test_set))   # 45000 5000 10000

    images, labels = next(iter(train_dataloader))
    print(images.shape)   # torch.Size([128, 3, 32, 32])
    print(images.dtype)   # torch.float32
    print(labels.shape)   # torch.Size([128])

    # 归一化后整体均值应接近 0，说明 Normalize 生效了
    print(images.mean().item())   # 大致在 -1 ~ 1 之间，接近 0

    # 连续取两次，形状一致但内容不同，说明 shuffle 生效了
    a, _ = next(iter(train_dataloader))
    b, _ = next(iter(train_dataloader))
    print(torch.equal(a, b))      # False

if __name__ == "__main__":
    def denormalize(img, mean, std):
        img = img.clone()
        for c in range(3):
            img[c] = img[c] * std[c] + mean[c]
        return img
    img = denormalize(images[0], mean, std).permute(1, 2, 0)
    plt.imshow(img.clamp(0, 1))
    plt.show()