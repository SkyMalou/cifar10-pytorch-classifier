# -*- coding: utf-8 -*-
"""项目配置：所有路径和数据/训练参数集中在这里。

好处：
1. 改超参数只改这一个文件，不用在几个脚本里来回翻
2. 每个实验改 exp_name 即可，日志和存档自动按名字分开
3. train.py / test.py / predict.py 都支持命令行覆盖，优先级是：
      命令行参数  >  config.py 里的默认值

用法：
    from config import load_config
    cfg = load_config()                 # 全用默认值
    cfg = load_config(epochs=50)        # 只覆盖 epochs
"""

from dataclasses import dataclass
from pathlib import Path

#项目根目录（分类数据集/），用 __file__ 定位，不受运行时工作目录影响
BASE_DIR = Path(__file__).parent

#CIFAR-10 的 10 个类别，顺序和标签 0~9 严格一一对应
CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


@dataclass
class Config:
    # ==================== 路径 ====================
    data_root: Path = BASE_DIR / "data"          # CIFAR-10 数据存放目录
    ckpt_dir:  Path = BASE_DIR / "checkpoints"   # 模型权重
    log_dir:   Path = BASE_DIR / "logs"          # TensorBoard 日志
    out_dir:   Path = BASE_DIR / "outputs"       # 训练曲线、混淆矩阵等图片

    # ==================== 数据 ====================
    #CIFAR-10 训练集的逐通道均值和标准差（社区通用值，别随手改）
    mean: tuple = (0.4914, 0.4822, 0.4465)
    std:  tuple = (0.2470, 0.2435, 0.2616)

    batch_size:  int = 128
    num_workers: int = 2      # DataLoader 子进程数；Windows 下不建议超过 4
    val_size:    int = 5000   # 从 50000 张训练图里划出多少张做验证集
    split_seed:  int = 42     # 训练/验证划分的随机种子（保证每次切分一致）

    # ==================== 数据增强 ====================
    crop_padding: int   = 4      # RandomCrop 的填充像素数
    erase_p:      float = 0.25   # RandomErasing 触发概率，0 表示关闭

    # ==================== 模型 ====================
    num_classes: int   = 10
    dropout:     float = 0.3     # 分类头的 Dropout 概率

    # ==================== 训练 ====================
    exp_name:        str   = "resnet18_re"  # 实验名：决定日志和存档的文件名
    epochs:          int   = 100
    lr:              float = 1e-3
    weight_decay:    float = 5e-4
    label_smoothing: float = 0.1
    seed:            int   = 42
    use_amp:         bool  = True   # 混合精度（只在 GPU 上生效）

    # ==================== 派生路径 ====================
    @property
    def ckpt_path(self) -> Path:
        """本次实验的模型存档路径"""
        return self.ckpt_dir / f"{self.exp_name}.pth"

    @property
    def log_path(self) -> Path:
        """本次实验的 TensorBoard 日志目录"""
        return self.log_dir / self.exp_name

    def make_dirs(self):
        """创建所有需要的目录（已存在则跳过）"""
        for d in (self.data_root, self.ckpt_dir, self.log_dir, self.out_dir):
            Path(d).mkdir(parents=True, exist_ok=True)
        return self


def load_config(**overrides) -> Config:
    """创建配置对象，可传入覆盖项。

    值为 None 的覆盖项会被忽略，这样调用方可以直接把 argparse 的结果
    原样传进来（没传的命令行参数就是 None，表示"不覆盖"）。
    """
    cfg = Config()
    for key, value in overrides.items():
        if value is None:
            continue
        if not hasattr(cfg, key):
            raise ValueError(f"未知的配置项: {key}")
        # 命令行传进来的路径是字符串，统一转成 Path
        if key in ("data_root", "ckpt_dir", "log_dir", "out_dir"):
            value = Path(value)
        setattr(cfg, key, value)
    return cfg.make_dirs()
