from torch import nn
from torchvision.models import resnet18


class SmallCNN(nn.Module):
    """基线模型：3 层卷积 + 2 层全连接，145,706 参数。
    保留在这里用于和 ResNet18 做对比实验。"""
    def __init__(self, num_classes=10, dropout=0.3):
        super(SmallCNN, self).__init__()
        self.model = nn.Sequential(
            #Conv 后面紧跟 BatchNorm 时，卷积的 bias 会被 BN 的减均值抵消，属于多余参数，故 bias=False
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out = self.model(x)
        return out


class classify(nn.Module):
    """适配 CIFAR-10 的 ResNet18（输入 32×32）。

    ResNet18 自带 BatchNorm（每个卷积后都有一个），所以泛化三件套里
    BatchNorm 由网络结构提供，Dropout 加在分类头，标签平滑在 train.py 的损失函数里。
    """
    def __init__(self, num_classes=10, dropout=0.3):
        super(classify, self).__init__()
        #weights=None 表示不加载 ImageNet 预训练权重，从零开始训练
        self.net = resnet18(weights=None)

        #原版 ResNet 为 224×224 设计，直接拿来跑 32×32 会下采样过头，必须改三处：
        #1) 首层 7×7 stride2 会把 32×32 直接砍成 16×16，太狠
        #   → 换成 3×3 stride1 padding1，尺寸保持不变
        self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        #2) 紧跟的 maxpool 会再砍一半（16→8）→ 用 Identity 去掉
        self.net.maxpool = nn.Identity()
        #3) ResNet 原版分类头没有 Dropout → 在最后的全连接前加一个
        in_features = self.net.fc.in_features        #512
        self.net.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.net(x)
