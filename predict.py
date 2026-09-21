#整体流程
"""图片路径 → PIL 读取 → convert("RGB") → Resize(32,32) → ToTensor → Normalize
        → unsqueeze(0) 补 batch 维 → 送模型 → logits
        → softmax → 取 top-5 → 打印类别名和概率
"""
import sys
import torch
from pathlib import Path
from PIL import Image
from model import classify
from torchvision import transforms
from dataset import mean, std, test_set
device = torch.device("cuda")if torch.cuda.is_available() else torch.device("cpu")

#脚本所在目录，用来定位模型和默认图片（避免依赖运行时的工作目录）
BASE_DIR = Path(__file__).parent

#加载最佳模型
def load_model():
    ckpt=torch.load(BASE_DIR/"checkpoints"/"best_model.pth",map_location=device,weights_only=False)
    model=classify().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"已加载第 {ckpt['epoch']} 轮的模型，val_acc={ckpt['best_acc']:.4f}")
    return model

#predict图片的预处理
preprocess=transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

#预测的过程
def predict(model,img_path,topk=5):
    p=Path(img_path)
    if not p.exists():
        print(f"\n跳过（文件不存在）: {img_path}")
        return None
    try:
        img =Image.open(p).convert("RGB")#读取图片并转换为RGB
    except Exception as e:
        print(f"\n跳过（无法读取图片）: {img_path}  {type(e).__name__}: {e}")
        return None
    img_tensor=preprocess(img).unsqueeze(0).to(device)#预处理并分batch

    with torch.no_grad():
        probs = model(img_tensor).softmax(1)[0]

    top=probs.topk(topk)
    print(f"\n图片: {img_path}  原始尺寸: {img.size}")
    for rank, (p, idx) in enumerate(zip(top.values, top.indices), 1):
        bar = "#" * int(p.item() * 40)
        print(f"  {rank}. {test_set.classes[idx]:<12} {p.item():>7.4f}  {bar}")
    return test_set.classes[top.indices[0]]

#把命令行参数展开成图片路径列表（目录会被扫描）
def collect_paths(args):
    exts={".jpg",".jpeg",".png",".bmp",".webp",".gif"}
    paths=[]
    for a in args:
        p=Path(a)
        if p.is_dir():
            paths+=sorted(q for q in p.rglob("*") if q.suffix.lower() in exts)
        else:
            paths.append(p)
    return paths

if __name__ == "__main__":
    raw = sys.argv[1:]
    print(f"命令行参数: {raw if raw else '(无，使用默认 predict_picture)'}")

    model = load_model()

    args = raw or [BASE_DIR/"predict_picture"]
    paths = collect_paths(args)
    if not paths:
        print(f"没有找到任何图片（解析出的路径: {[str(a) for a in args]}）")
        sys.exit(1)

    print(f"共 {len(paths)} 张图片")
    done = sum(predict(model, p) is not None for p in paths)
    print(f"\n完成 {done}/{len(paths)} 张")