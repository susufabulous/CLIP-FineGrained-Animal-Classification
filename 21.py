"""CLIP ViT-B/32 单模型分类训练及推理脚本（修正版）。

依赖：PyTorch、torchvision、open_clip_torch、modelscope、Pillow、numpy、tqdm。
数据：<data-dir>/train/<类别名>/*.{jpg,png,...}；<data-dir>/test/*.{jpg,png,...}。
默认输出：<data-dir>/03/best_vit_b32.pth、pred_results.csv、pred_results.zip。

说明：ArcFace、CutMix、长尾采样、疑似噪声降权、EMA、TTA 都属于待验证的策略，
不会保证比朴素微调更准确。验证集用于模型及 TTA 选择，不能作为独立测试集。
预训练权重只允许从 ModelScope 魔搭获取或从指定本地文件读取，不访问 Hugging Face。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import sys
import traceback
import zipfile
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from PIL import Image, ImageFile, ImageOps
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
DEFAULT_STD = (0.26862954, 0.26130258, 0.2757711)
MODELSCOPE_MODEL_ID = "AI-ModelScope/CLIP-ViT-B-32-laion2B-s34B-b79K"
MODELSCOPE_WEIGHT_FILE = "open_clip_pytorch_model.bin"


class PadToSquare:
    """以白色补齐图像，避免直接拉伸引入几何畸变。"""

    def __call__(self, image: Image.Image) -> Image.Image:
        """返回保留长宽比的方形 RGB 图像。"""
        width, height = image.size
        side = max(width, height)
        left, top = (side - width) // 2, (side - height) // 2
        return ImageOps.expand(
            image,
            (left, top, side - width - left, side - height - top),
            fill=(255, 255, 255),
        )


class RandomImageQuality:
    """模拟降采样和 JPEG 压缩，仅作用于训练图像。"""

    def __init__(self, probability: float = 0.25) -> None:
        """设置退化增强概率。"""
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        """随机返回原图或质量退化图。"""
        if random.random() >= self.probability:
            return image
        width, height = image.size
        side = random.randint(112, 200)
        small = ImageOps.contain(image, (side, side), Image.Resampling.BILINEAR)
        image = small.resize((width, height), Image.Resampling.BILINEAR)
        if random.random() < 0.5:
            with io.BytesIO() as stream:
                image.save(stream, format="JPEG", quality=random.randint(55, 90))
                stream.seek(0)
                with Image.open(stream) as decoded:
                    image = decoded.convert("RGB")
        return image


class TrainGeometry:
    """训练时交替使用保形缩放与随机裁剪。"""

    def __init__(self) -> None:
        """准备空间变换。"""
        self.pad = PadToSquare()
        self.crop = transforms.RandomResizedCrop(
            224,
            scale=(0.50, 1.0),
            ratio=(0.80, 1.20),
            interpolation=transforms.InterpolationMode.BICUBIC,
        )

    def __call__(self, image: Image.Image) -> Image.Image:
        """返回 224×224 的 PIL 图像。"""
        if random.random() < 0.5:
            return self.crop(image)
        return self.pad(image).resize((224, 224), Image.Resampling.BICUBIC)


class EvalViews:
    """生成 1 张完整图 + 5 张区域裁剪图，固定索引 0 为完整图。"""

    def __init__(self, evaluate_transform: transforms.Compose) -> None:
        """设置各视图统一的预处理流程。"""
        self.transform = evaluate_transform

    def __call__(self, image: Image.Image) -> Tensor:
        """返回形状 [6, 3, 224, 224] 的视图集合。"""
        width, height = image.size
        crop_width = max(1, int(width * 0.85))
        crop_height = max(1, int(height * 0.85))
        x_mid = (width - crop_width) // 2
        y_mid = (height - crop_height) // 2
        crops = [
            image,
            image.crop((0, 0, crop_width, crop_height)),
            image.crop((width - crop_width, 0, width, crop_height)),
            image.crop((0, height - crop_height, crop_width, height)),
            image.crop((width - crop_width, height - crop_height, width, height)),
            image.crop((x_mid, y_mid, x_mid + crop_width, y_mid + crop_height)),
        ]
        return torch.stack([self.transform(crop) for crop in crops])


def make_transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD):
    """构建训练、验证变换；擦除必须发生于 ToTensor 和 Normalize 后。"""
    train = transforms.Compose(
        [
            TrainGeometry(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(
                degrees=8,
                translate=(0.04, 0.04),
                scale=(0.95, 1.05),
                fill=(255, 255, 255),
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.12),
            RandomImageQuality(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            # 在标准化空间擦除，0 相当于训练集均值颜色。
            transforms.RandomErasing(p=0.30, scale=(0.02, 0.25), ratio=(0.3, 3.3), value=0),
        ]
    )
    evaluate = transforms.Compose(
        [
            PadToSquare(),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train, evaluate


class ArcFaceNoiseLoss(nn.Module):
    """真实角度间隔 ArcFace + 平滑交叉熵 + 广义交叉熵。"""

    def __init__(self, weight: float = 0.0, margin: float = 0.15, scale: float = 20.0) -> None:
        """配置广义 CE 混合比例、角度间隔和余弦温度。"""
        super().__init__()
        self.weight = weight
        self.margin = margin
        self.scale = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, cosine: Tensor, target: Tensor) -> Tensor:
        """仅训练期给目标类别添加角度间隔；推理不添加。"""
        cosine = cosine.float().clamp(-1 + 1e-6, 1 - 1e-6)
        sine = torch.sqrt(torch.clamp(1.0 - cosine.square(), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        # 常见 ArcFace 单调性修正，避免 theta+m 越过 pi 后的反向梯度。
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        logits = cosine.clone()
        row = torch.arange(cosine.size(0), device=cosine.device)
        logits[row, target] = phi[row, target]
        logits = self.scale * logits
        cross_entropy = F.cross_entropy(logits, target, label_smoothing=0.05, reduction="none")
        log_pt = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        q = 0.3
        generalized_ce = -torch.expm1(q * log_pt) / q
        return ((1 - self.weight) * cross_entropy + self.weight * generalized_ce).mean()


class ImageDataset(Dataset):
    """返回图像张量及已映射成连续下标的标签。"""

    def __init__(self, paths: list[str], labels: list[int], transform) -> None:
        """保留图像路径、标签与变换。"""
        self.paths, self.labels, self.transform = paths, labels, transform
        if len(paths) != len(labels):
            raise ValueError("paths 与 labels 长度不一致")

    def __len__(self) -> int:
        """返回数据集大小。"""
        return len(self.paths)

    def __getitem__(self, index: int):
        """读取图片；EXIF 损坏时忽略方向校正，保留可读取的像素。"""
        path = self.paths[index]
        try:
            with Image.open(path) as image:
                try:
                    rgb = ImageOps.exif_transpose(image).convert("RGB")
                except (SyntaxError, ValueError, OSError) as exc:
                    import warnings

                    warnings.warn(
                        f"EXIF 解析失败，改用原始方向：{path}；{exc}",
                        RuntimeWarning,
                    )
                    rgb = image.convert("RGB")

                rgb.load()

            return self.transform(rgb), self.labels[index]
        except Exception as exc:
            raise RuntimeError(f"读取图片失败：{path}") from exc

def parse_data(root: Path):
    """按类目录读取训练样本，并按文件名确定测试输出顺序。"""
    train_dir, test_dir = root / "train", root / "test"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"训练目录不存在：{train_dir}")
    names = [path.name for path in train_dir.iterdir() if path.is_dir()]
    if all(name.isdigit() for name in names):
        names.sort(key=lambda name: (int(name), name))
    else:
        names.sort()
    paths, labels = [], []
    for class_id, name in enumerate(names):
        class_paths = sorted(
            (p for p in (train_dir / name).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
            key=lambda p: p.as_posix(),
        )
        if not class_paths:
            raise ValueError(f"类别 {name} 不包含有效图片")
        paths.extend(map(str, class_paths))
        labels.extend([class_id] * len(class_paths))
    if len(names) < 2:
        raise ValueError("至少需要两个非空类别目录")
    tests = sorted(
        (str(p) for p in test_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if test_dir.is_dir() else [],
        key=lambda p: Path(p).name,
    )
    if len({Path(p).name for p in tests}) != len(tests):
        raise ValueError("测试目录中有重名图片，提交格式无法唯一标识")
    return paths, labels, names, tests


def stratified_indices(paths: list[str], labels: list[int], names: list[str], ratio: float, seed: int):
    """单样本类只分配给训练；其余类别分别至少保留一张训练与验证图片。"""
    rng = random.Random(seed)
    train, val = [], []
    for class_id in range(len(names)):
        indices = [i for i, target in enumerate(labels) if target == class_id]
        indices.sort(key=lambda i: paths[i])
        rng.shuffle(indices)
        nval = min(len(indices) - 1, max(1, int(len(indices) * ratio))) if len(indices) >= 2 else 0
        val.extend(indices[:nval])
        train.extend(indices[nval:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def make_split(paths, labels, names, root: Path, args, output: Path):
    """加载已冻结划分或新建分层划分，核对不重叠且完整覆盖。"""
    relatives = [Path(p).relative_to(root).as_posix() for p in paths]
    if args.split_file:
        saved = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
        if saved.get("class_names") != names:
            raise ValueError("旧划分类别映射与当前数据不一致")
        lookup = {relative: i for i, relative in enumerate(relatives)}
        try:
            train = [lookup[p] for p in saved["train"]]
            val = [lookup[p] for p in saved["val"]]
        except (KeyError, TypeError) as exc:
            raise ValueError("划分文件中的样本不存在或字段异常") from exc
        if len(train) + len(val) != len(paths) or len(set(train + val)) != len(paths):
            raise ValueError("划分文件存在重复、缺失或数据集发生变化")
    else:
        train, val = stratified_indices(paths, labels, names, args.val_ratio, args.seed)
    if not train or not val:
        raise ValueError("训练集和验证集必须非空，且至少有一个类别具有 >=2 张图片")
    for title, subset in (("训练", train), ("验证", val)):
        missing = set(range(len(names))) - {labels[i] for i in subset}
        invalid = [i for i in missing if title == "训练" or labels.count(i) >= 2]
        if invalid:
            raise ValueError(f"{title}集遗漏了可划分的类别：{[names[i] for i in invalid]}")
    with (output / "class_split_counts.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["category", "total", "train", "val"])
        for class_id, name in enumerate(names):
            ntrain = sum(labels[i] == class_id for i in train)
            nval = sum(labels[i] == class_id for i in val)
            writer.writerow([name, ntrain + nval, ntrain, nval])
    (output / "split.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "val_ratio": args.val_ratio,
                "class_names": names,
                "method": "loaded" if args.split_file else "stratified",
                "train": [relatives[i] for i in train],
                "val": [relatives[i] for i in val],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return train, val


def select_device(requested: str) -> torch.device:
    """优先 CUDA、再 MPS，最后 CPU；显式指定不可用设备时报错。"""
    print(f"Python={sys.executable} | PyTorch={torch.__version__} | CUDA={torch.cuda.is_available()}")
    has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if requested == "auto":
        choice = "cuda" if torch.cuda.is_available() else "mps" if has_mps else "cpu"
    else:
        choice = requested
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；检查 NVIDIA 驱动、CUDA 版 PyTorch 或改用 --device cpu")
    if choice == "mps" and not has_mps:
        raise RuntimeError("MPS 不可用；需 Apple Silicon/macOS 对应 PyTorch 或改用 --device cpu")
    if choice == "cuda":
        print(f"GPU={torch.cuda.get_device_name(0)}")
    print(f"训练设备：{choice}")
    return torch.device(choice)


def autocast_context(enabled: bool):
    """仅 CUDA 模式启用官方 AMP，其余设备为无操作上下文。"""
    return torch.autocast(device_type="cuda", dtype=torch.float16) if enabled else nullcontext()


def make_scaler(enabled: bool):
    """创建兼容不同 PyTorch 版本的 CUDA AMP 梯度缩放器。"""
    amp_module = getattr(torch, "amp", None)

    # 新版 PyTorch
    if amp_module is not None and hasattr(amp_module, "GradScaler"):
        return amp_module.GradScaler("cuda", enabled=enabled)

    # 旧版 PyTorch
    if hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=enabled)

    raise RuntimeError(
        "当前 PyTorch 缺少 GradScaler，无法初始化混合精度训练。"
    )

class Classifier(nn.Module):
    """OpenCLIP ViT-B/32 图像编码器 + 归一化可训练分类头。"""

    def __init__(self, visual: nn.Module, classes: int, dropout: float, scale: float = 20.0):
        """只解冻最后两个 Transformer Block、ln_post 和 proj。"""
        super().__init__()
        self.visual = visual.float()
        self.head_only = False
        for name, parameter in self.visual.named_parameters():
            parameter.requires_grad_(
                name.startswith(("transformer.resblocks.10.", "transformer.resblocks.11.", "ln_post."))
                or name == "proj"
            )
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.empty(classes, visual.output_dim))
        nn.init.normal_(self.weight, std=0.02)
        self.logit_scale = scale

    def forward(self, images: Tensor) -> Tensor:
        """输出余弦相似度，不含 ArcFace 训练期间隔。"""
        with torch.no_grad() if self.head_only else nullcontext():
            features = self.visual(images).float()
        features = F.normalize(self.dropout(features), dim=-1)
        weights = F.normalize(self.weight, dim=-1)
        return F.linear(features, weights)


class EMA:
    """维护可训练参数的指数滑动平均，并支持临时验证。"""

    def __init__(self, model: nn.Module, decay: float) -> None:
        """复制模型参数作为 EMA 初始值。"""
        self.decay, self.steps = decay, 0
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """只在优化器完成真实更新后调用。"""
        self.steps += 1
        decay = min(self.decay, (1 + self.steps) / (10 + self.steps))
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(parameter.detach(), 1 - decay)

    @contextmanager
    def apply(self, model: nn.Module) -> Iterator[None]:
        """临时应用 EMA 权重，退出时恢复原始训练参数。"""
        backup: dict[str, Tensor] = {}
        try:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in self.shadow:
                        backup[name] = parameter.detach().clone()
                        parameter.copy_(self.shadow[name])
            yield
        finally:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in backup:
                        parameter.copy_(backup[name])


def optimizer_for(model: nn.Module, args):
    """分别设定骨干、分类头学习率，bias 与一维参数不做衰减。"""
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        backbone = name.startswith("visual.")
        lr = args.lr_backbone if backbone else args.lr_head
        if name.startswith("visual.transformer.resblocks.10."):
            lr *= 0.7
        wd = 0.0 if parameter.ndim < 2 or name.endswith(".bias") else 0.02
        groups.setdefault((lr, wd, backbone), []).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": parameters, "lr": lr, "base_lr": lr, "weight_decay": wd, "backbone": backbone}
            for (lr, wd, backbone), parameters in groups.items()
        ],
        eps=1e-6,
    )


def lr_factor(step: int, total: int, warmup: int) -> float:
    """逐步 warmup 后执行余弦衰减，最低保持 5% 学习率。"""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup - 1)
    return 0.05 + 0.475 * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(model: Classifier, loader: DataLoader, device: torch.device, amp: bool):
    """在无训练间隔的分类 logit 上计算验证 CE 与 top-1。"""
    model.eval()
    correct, count, loss_sum = 0, 0, 0.0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_context(amp):
            logits = model.logit_scale * model(images)
        loss_sum += F.cross_entropy(logits.float(), targets, reduction="sum").item()
        correct += (logits.argmax(1) == targets).sum().item()
        count += targets.numel()
    if count == 0:
        raise ValueError("无法评估空数据集")
    return 100.0 * correct / count, loss_sum / count


@torch.no_grad()
def view_scores(model: Classifier, images: Tensor, amp: bool, mode: str):
    """计算单视图、翻转或 1+5 裁剪翻转的平均概率。

    images 的形状为 [N, 6, 3, 224, 224]；索引 0 是全图。
    不对 plain 执行翻转或多裁剪，从而使验证模式和实际推理一致。
    """
    if mode not in {"plain", "flip", "five_crop_flip"}:
        raise ValueError(f"未知 TTA 模式：{mode}")
    with autocast_context(amp):
        base = (model.logit_scale * model(images[:, 0])).float().softmax(-1)
        if mode == "plain":
            return base
        flipped = (model.logit_scale * model(images[:, 0].flip(-1))).float().softmax(-1)
        base = (base + flipped) * 0.5
        if mode == "flip":
            return base
        total = base * 0.4
        # 余下五个局部裁剪各占 12%，中心裁剪位于索引 5。
        for view_id in range(1, 6):
            view = images[:, view_id]
            p = (model.logit_scale * model(view)).float().softmax(-1)
            p_flip = (model.logit_scale * model(view.flip(-1))).float().softmax(-1)
            total += (p + p_flip) * 0.06
    return total


@torch.no_grad()
def select_views(model: Classifier, loader: DataLoader, device: torch.device, amp: bool):
    """在独立于训练的验证划分上选定固定推理视图策略。"""
    model.eval()
    correct = {"plain": 0, "flip": 0, "five_crop_flip": 0}
    count = 0
    for images, labels in tqdm(loader, desc="TTA validation", mininterval=5):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        # 各模式分别计算，不让 plain 与 flip 共享错误的平均结果。
        for mode in correct:
            probabilities = view_scores(model, images, amp, mode)
            correct[mode] += (probabilities.argmax(1) == labels).sum().item()
        count += labels.numel()
    if count == 0:
        raise ValueError("TTA 验证集为空")
    accuracies = {mode: 100.0 * number / count for mode, number in correct.items()}
    selected = "plain"
    for mode in ("flip", "five_crop_flip"):
        if accuracies[mode] >= accuracies[selected] + 0.15:
            selected = mode
    return selected, accuracies


@torch.no_grad()
def update_noise_weights(model: Classifier, dataset: Dataset, indices: list[int], labels: list[int],
                         device: torch.device, amp: bool, workers: int):
    """按各类别训练样本的验证式 CE 检测异常损失，只对异常样本降权。

    使用无随机增强的数据集，且不查看验证集或测试集标签。
    异常高损失并不等价于错误标签，因此此策略只属于启发式。
    """
    model.eval()
    loader = DataLoader(
        Subset(dataset, indices), batch_size=64, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    losses: list[float] = []
    for images, targets in tqdm(loader, desc="Noise assessment", mininterval=5):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_context(amp):
            logits = model.logit_scale * model(images)
        ce = F.cross_entropy(logits.float(), targets, reduction="none")
        losses.extend(ce.cpu().tolist())
    loss_array = np.asarray(losses, dtype=np.float64)
    labels_array = np.asarray([labels[i] for i in indices])
    weights = np.ones(len(indices), dtype=np.float64)
    for class_id in np.unique(labels_array):
        class_positions = np.flatnonzero(labels_array == class_id)
        if len(class_positions) < 5:
            continue
        class_losses = loss_array[class_positions]
        threshold = class_losses.mean() + 2.0 * (class_losses.std() + 1e-6)
        abnormal_positions = class_positions[class_losses > threshold]
        weights[abnormal_positions] = 0.2  # 关键修复：不能将整类权重设为 0.2。
    return weights


def save_best(model: Classifier, output: Path, names: list[str], args, epoch: int,
              source: str, acc: float, loss: float) -> None:
    """原子保存最优模型和类别映射，防止进程中断留下半截权重文件。"""
    checkpoint = {
        "model": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "class_names": names,
        "config": vars(args),
        "backbone": "ViT-B/32",
        "epoch": epoch,
        "source": source,
        "val_acc": float(acc),
        "val_loss": float(loss),
    }
    temp = output / "best_vit_b32.pth.tmp"
    torch.save(checkpoint, temp)
    os.replace(temp, output / "best_vit_b32.pth")


def resolve_paths(args):
    """数据默认路径相对于脚本而非当前终端目录。"""
    data = Path(args.data_dir).expanduser()
    if not data.is_absolute():
        data = Path(__file__).resolve().parent / data
    root = data.resolve()
    out = Path(args.output_dir).expanduser()
    if not out.is_absolute():
        out = root / out
    return root, out.resolve()


def _sha256_file(path: Path) -> str:
    """流式计算权重 SHA-256，不把数百 MB 的文件完整读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_modelscope_checkpoint(args, output: Path) -> tuple[Path, dict]:
    """仅调用魔搭 SDK 下载 OpenCLIP 格式文件，支持本地权重与离线缓存。

    注意：不能把原来的 OpenCLIP pretrained 标签直接传入 create_model，
    那样在部分版本中会转向 Hugging Face；此处始终返回本地真实路径。
    """
    if args.clip_checkpoint:
        checkpoint = Path(args.clip_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"本地 OpenCLIP 权重不存在：{checkpoint}")
        metadata = {"source": "user_local_checkpoint", "repo_id": None, "revision": None}
    else:
        try:
            from modelscope.hub.file_download import model_file_download
        except ImportError as exc:
            raise RuntimeError(
                "缺少 ModelScope SDK。请运行：pip install -U modelscope；"
                "或者使用 --clip-checkpoint 指定已下载的 OpenCLIP 权重。"
            ) from exc

        root, _ = resolve_paths(args)
        cache = Path(args.modelscope_cache_dir).expanduser()
        if not cache.is_absolute():
            cache = root / cache
        cache = cache.resolve()
        cache.mkdir(parents=True, exist_ok=True)
        print(f"魔搭模型：{args.modelscope_model}")
        print(f"魔搭下载文件：{MODELSCOPE_WEIGHT_FILE}，缓存目录：{cache}")
        try:
            options = {
                "model_id": args.modelscope_model,
                "file_path": MODELSCOPE_WEIGHT_FILE,
                "cache_dir": str(cache),
                "local_files_only": args.offline,
            }
            if args.modelscope_revision:
                options["revision"] = args.modelscope_revision
            checkpoint = Path(model_file_download(**options)).resolve()
        except Exception as exc:
            raise RuntimeError(
                "无法从 ModelScope 魔搭获取权重文件。仓库："
                f"https://www.modelscope.cn/models/{args.modelscope_model}；"
                f"目标文件：{MODELSCOPE_WEIGHT_FILE}。"
                "请核对魔搭仓库的文件列表与网络状态；"
                "已经手动下载的情况下可用 --clip-checkpoint 指定本地文件；"
                "脚本不会自动转向 Hugging Face。"
            ) from exc
        metadata = {
            "source": "modelscope",
            "repo_id": args.modelscope_model,
            "revision": args.modelscope_revision or "default",
            "model_page": f"https://www.modelscope.cn/models/{args.modelscope_model}",
        }

    if not checkpoint.is_file():
        raise FileNotFoundError(f"权重路径无效：{checkpoint}")
    # OpenCLIP B/32 完整权重约 605MB。该下限可拦截 100 字节的 Git LFS 指针、HTML 错误页。
    if checkpoint.stat().st_size < 100 * 1024 * 1024:
        raise ValueError(
            f"权重文件过小（{checkpoint.stat().st_size} bytes）：{checkpoint}。"
            "这通常是未下载的 LFS 指针、错误页面或错误权重文件。"
        )
    metadata.update({
        "checkpoint_file": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "sha256": _sha256_file(checkpoint),
        "architecture": "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
    })
    return checkpoint, metadata


def load_clip(args, output: Path):
    """通过魔搭拿到本地权重后，使用 OpenCLIP 官方接口严格加载 ViT-B/32。"""
    # 防御性设置：后续库即使意外触发 Hub 调用，也不允许 Hugging Face 网络请求。
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    checkpoint, metadata = resolve_modelscope_checkpoint(args, output)
    import open_clip

    # model_name 是本地已注册的架构名；pretrained 只能是已有的本地文件路径。
    # 故意不使用 'hf-hub:*'、内置远程 pretrained 标签或外部配置文件。
    try:
        base, _, official_eval = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=str(checkpoint), device="cpu"
        )
    except Exception as exc:
        raise RuntimeError(
            f"OpenCLIP 无法加载权重：{checkpoint}。请确认仓库提供的是完整的 "
            "open_clip_pytorch_model.bin，而非 Transformers 格式的 pytorch_model.bin。"
        ) from exc
    visual = base.visual
    if (
        getattr(visual.conv1, "kernel_size", None) != (32, 32)
        or len(visual.transformer.resblocks) != 12
        or visual.conv1.out_channels != 768
        or visual.output_dim != 512
    ):
        raise ValueError("加载的视觉编码器不是预期的 ViT-B/32（224px / 512维）")
    normalizations = [op for op in official_eval.transforms if isinstance(op, transforms.Normalize)]
    if len(normalizations) != 1:
        raise RuntimeError("未能识别 OpenCLIP 官方 Normalize 预处理，请核查 open_clip_torch 版本")
    mean = tuple(float(v) for v in normalizations[0].mean)
    std = tuple(float(v) for v in normalizations[0].std)
    metadata["image_mean"] = mean
    metadata["image_std"] = std
    (output / "model_source.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"OpenCLIP: ViT-B-32 | 本地权重={checkpoint} | mean={mean} | std={std}")
    print(f"权重 SHA-256：{metadata['sha256']}")
    del base
    return visual, mean, std


def get_args(argv=None):
    """读取训练配置；参数名称兼容旧脚本中的 mixup-* 标志。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="../datasets")
    parser.add_argument("--output-dir", default="03")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--modelscope-model", default=MODELSCOPE_MODEL_ID,
                        help="魔搭模型仓库 ID，不得填写 hf-hub 名称")
    parser.add_argument("--modelscope-revision", default=None,
                        help="可选：锁定魔搭仓库版本/提交")
    parser.add_argument("--modelscope-cache-dir", default="_modelscope_cache",
                        help="相对于数据目录的魔搭权重缓存目录，可复用不同训练输出")
    parser.add_argument("--offline", action="store_true",
                        help="仅从魔搭本地缓存读取，禁止网络下载")
    parser.add_argument("--clip-checkpoint", default=None,
                        help="可选：手动下载的本地 OpenCLIP .bin 权重（优先级最高）")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--head-only-epochs", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tta-batch-size", type=int, default=8, help="TTA 数据有 6 视图，建议用较小批次")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr-backbone", type=float, default=5e-6)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mixup-alpha", type=float, default=0.2, help="历史兼容参数：实际为 CutMix Beta alpha")
    parser.add_argument("--mixup-prob", type=float, default=0.25, help="历史兼容参数：实际为 CutMix 概率")
    parser.add_argument("--noise-weight", type=float, default=0.2)
    parser.add_argument("--clean-epochs", type=int, default=5)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args(argv)


def validate_args(args) -> None:
    """训练前拒绝不合法的超参数，避免中途报错。"""
    if args.epochs < 1 or args.batch_size < 1 or args.tta_batch_size < 1 or args.workers < 0:
        raise ValueError("epochs、batch-size、tta-batch-size 必须为正，workers >= 0")
    if not (0 <= args.head_only_epochs < args.epochs and 0 <= args.clean_epochs < args.epochs):
        raise ValueError("head-only-epochs、clean-epochs 必须处于 [0, epochs)")
    if not (0 <= args.mixup_prob <= 1 and args.mixup_alpha > 0):
        raise ValueError("CutMix 概率 / alpha 无效")
    if not (0 <= args.noise_weight <= 0.3 and 0 <= args.ema_decay < 1):
        raise ValueError("noise-weight 应在 [0, 0.3]，ema-decay 应在 [0, 1)")
    if not (0 < args.val_ratio < 1 and args.min_delta > 0 and args.patience >= 1):
        raise ValueError("val-ratio/min-delta/patience 参数无效")
    if not (0 <= args.dropout < 1 and args.lr_head > 0 and args.lr_backbone > 0):
        raise ValueError("dropout/learning-rate 参数无效")
    if not args.modelscope_model or args.modelscope_model.startswith(("hf-hub:", "hf://")):
        raise ValueError("--modelscope-model 必须是 ModelScope 仓库 ID，不接受 Hugging Face 协议")


def apply_cutmix(images: Tensor, targets: Tensor, alpha: float):
    """执行标准矩形区域 CutMix，并由实际裁剪面积修正 lambda。"""
    if len(targets) < 2:
        return images, targets, targets, 1.0
    lam = float(np.random.beta(alpha, alpha))
    order = torch.randperm(len(targets), device=targets.device)
    height, width = images.shape[-2:]
    cut_ratio = math.sqrt(1.0 - lam)
    crop_width = int(width * cut_ratio)
    crop_height = int(height * cut_ratio)
    center_x = np.random.randint(width)
    center_y = np.random.randint(height)
    x1 = max(0, center_x - crop_width // 2)
    x2 = min(width, center_x + crop_width // 2)
    y1 = max(0, center_y - crop_height // 2)
    y2 = min(height, center_y + crop_height // 2)
    if x2 > x1 and y2 > y1:
        # advanced indexing 的源数据是副本，不会受到左侧原位覆盖影响。
        images[:, :, y1:y2, x1:x2] = images[order, :, y1:y2, x1:x2]
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (height * width)
    return images, targets, targets[order], lam


def run(args) -> None:
    """完成一次训练、验证、模型选择及测试集预测。"""
    validate_args(args)
    device = select_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    root, output = resolve_paths(args)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best_vit_b32.pth"
    if checkpoint_path.exists():
        raise FileExistsError(f"检测到已有模型：{checkpoint_path}；请先备份或更换 --output-dir")
    paths, labels, names, tests = parse_data(root)
    train_ids, val_ids = make_split(paths, labels, names, root, args, output)

    # 在加载权重时拿到该权重实际使用的 OpenCLIP 图像预处理参数。
    visual, mean, std = load_clip(args, output)
    model = Classifier(visual, len(names), args.dropout).to(device)
    del visual
    train_tf, eval_tf = make_transforms(mean, std)
    train_dataset = ImageDataset(paths, labels, train_tf)
    clean_dataset = ImageDataset(paths, labels, eval_tf)
    amp = device.type == "cuda" and not args.no_amp
    scaler = make_scaler(amp)
    optimizer = optimizer_for(model, args)
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None
    criterion = ArcFaceNoiseLoss(weight=0.0, margin=0.15, scale=model.logit_scale)

    def create_loader(dataset: Dataset, *, sampler=None, batch_size: int | None = None):
        """为 train/val/test 创建一致的 DataLoader。"""
        return DataLoader(
            dataset,
            batch_size=batch_size or args.batch_size,
            sampler=sampler,
            shuffle=False,
            drop_last=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )

    val_loader = create_loader(Subset(clean_dataset, val_ids))
    clean_ids = random.Random(args.seed).sample(train_ids, min(len(train_ids), 2048))
    clean_loader = create_loader(Subset(clean_dataset, clean_ids))
    train_labels = np.asarray([labels[i] for i in train_ids], dtype=np.int64)
    class_counts = np.bincount(train_labels, minlength=len(names))
    class_weights = 1.0 / np.sqrt(np.maximum(class_counts, 1))
    sample_weights = class_weights[train_labels].astype(np.float64).copy()
    (output / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"ViT-B/32 | classes={len(names)} | train={len(train_ids)} | val={len(val_ids)} | test={len(tests)}")

    best_acc, best_loss = -1.0, float("inf")
    significant_best, stale = -1.0, 0
    history: list[dict] = []
    steps_per_epoch = math.ceil(len(train_ids) / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    warmup = min(3 * steps_per_epoch, max(1, total_steps // 5))

    for epoch in range(args.epochs):
        model.head_only = epoch < args.head_only_epochs
        model.train()
        if model.head_only:
            model.visual.eval()
        criterion.weight = args.noise_weight if epoch >= args.head_only_epochs + 2 else 0.0
        cutmix_probability = args.mixup_prob if epoch < args.epochs - args.clean_epochs else 0.0
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double), num_samples=len(train_ids), replacement=True
        )
        train_loader = create_loader(Subset(train_dataset, train_ids), sampler=sampler)
        nseen, sum_loss, mix_correct = 0, 0.0, 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", mininterval=5)

        for batch_id, (images, targets) in enumerate(progress):
            global_step = epoch * steps_per_epoch + batch_id
            factor = lr_factor(global_step, total_steps, warmup)
            for group in optimizer.param_groups:
                if group["backbone"]:
                    ft_step = max(0, (epoch - args.head_only_epochs) * steps_per_epoch + batch_id)
                    ft_total = max(1, (args.epochs - args.head_only_epochs) * steps_per_epoch)
                    ft_warmup = min(steps_per_epoch, max(0, ft_total - 1))
                    group["lr"] = 0.0 if model.head_only else group["base_lr"] * lr_factor(
                        ft_step, ft_total, ft_warmup
                    )
                else:
                    group["lr"] = group["base_lr"] * factor
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            lam, targets2 = 1.0, targets
            if len(targets) > 1 and np.random.random() < cutmix_probability:
                images, targets, targets2, lam = apply_cutmix(images, targets, args.mixup_alpha)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(amp):
                cosine = model(images)
                loss = lam * criterion(cosine, targets) + (1 - lam) * criterion(cosine, targets2)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("出现非有限损失 NaN/Inf，请检查图像或尝试 --no-amp")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None and scaler.get_scale() >= old_scale:
                ema.update(model)
            with torch.no_grad():
                predictions = cosine.argmax(1)
                mix_correct += lam * (predictions == targets).sum().item()
                mix_correct += (1 - lam) * (predictions == targets2).sum().item()
            sum_loss += loss.item() * len(targets)
            nseen += len(targets)
            progress.set_postfix(loss=f"{sum_loss / nseen:.4f}", mix_acc=f"{100 * mix_correct / nseen:.2f}%")

        # 训练样本的疑似噪声权重，用无随机增强的图像估计；跳过前两轮。
        if epoch >= 2 and args.noise_weight > 0:
            noise_weights = update_noise_weights(
                model, clean_dataset, train_ids, labels, device, amp, args.workers
            )
            sample_weights = class_weights[train_labels] * noise_weights

        raw_acc, raw_loss = evaluate(model, val_loader, device, amp)
        improved = False

        def consider(acc: float, loss_value: float, source: str) -> None:
            """使用准确率优先、CE 次优规则选择 checkpoint。"""
            nonlocal best_acc, best_loss, improved
            if acc > best_acc or (acc == best_acc and loss_value < best_loss - 1e-5):
                best_acc, best_loss, improved = acc, loss_value, True
                save_best(model, output, names, args, epoch + 1, source, acc, loss_value)

        consider(raw_acc, raw_loss, "raw")
        ema_acc = ema_loss = None
        if ema is not None:
            with ema.apply(model):
                ema_acc, ema_loss = evaluate(model, val_loader, device, amp)
                consider(ema_acc, ema_loss, "ema")
        clean_acc, _ = evaluate(model, clean_loader, device, amp)
        history.append(
            dict(
                epoch=epoch + 1,
                stage="head_only" if model.head_only else "finetune",
                train_mix_acc=100 * mix_correct / nseen,
                train_loss=sum_loss / nseen,
                clean_train_sample_acc=clean_acc,
                noise_weight=criterion.weight,
                cutmix_probability=cutmix_probability,
                raw_val_acc=raw_acc,
                raw_val_loss=raw_loss,
                ema_val_acc=ema_acc,
                ema_val_loss=ema_loss,
                best_val_acc=best_acc,
            )
        )
        with (output / "training_log.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        ema_text = f"{ema_acc:.2f}%" if ema_acc is not None else "OFF"
        print(
            f"Epoch {epoch + 1}: clean-train-sample={clean_acc:.2f}% "
            f"val={raw_acc:.2f}% EMA={ema_text} best={best_acc:.2f}%"
        )
        if best_acc >= significant_best + args.min_delta:
            significant_best, stale = best_acc, 0
        else:
            stale += 1
        if epoch + 1 >= max(10, args.head_only_epochs + 3) and stale >= args.patience:
            print("Early stopping")
            break

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("class_names") != names:
        raise ValueError("最佳权重的类别顺序与当前训练集不符")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.head_only = False
    model.eval()
    view_transform = EvalViews(eval_tf)
    val_view_loader = create_loader(
        Subset(ImageDataset(paths, labels, view_transform), val_ids), batch_size=args.tta_batch_size
    )
    selected_mode, view_accs = select_views(model, val_view_loader, device, amp)
    summary = {
        "best_epoch": checkpoint["epoch"],
        "best_source": checkpoint["source"],
        "single_view_val_acc": checkpoint["val_acc"],
        "view_accuracies": view_accs,
        "selected_view": selected_mode,
        "note": "验证集已用于 checkpoint/TTA 选择，不能当作独立测试评估。",
    }
    (output / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Selected TTA={selected_mode} | val={view_accs}")
    if not tests:
        print("测试目录无图片；只保留训练与验证结果。")
        return
    predictions_path = output / "pred_results.csv"
    test_dataset = ImageDataset(tests, list(range(len(tests))), view_transform)
    with predictions_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        for images, indices in tqdm(
            create_loader(test_dataset, batch_size=args.tta_batch_size),
            desc="Test prediction",
            mininterval=5,
        ):
            images = images.to(device, non_blocking=True)
            scores = view_scores(model, images, amp, selected_mode)
            for index, pred_id in zip(indices.tolist(), scores.argmax(1).cpu().tolist()):
                category = names[pred_id]
                category = f"{int(category):04d}" if category.isdigit() else category.zfill(4)
                writer.writerow([Path(tests[index]).name, category])
    with zipfile.ZipFile(output / "pred_results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(predictions_path, arcname="pred_results.csv")
    print(f"已生成提交文件：{output / 'pred_results.zip'}")


class Tee:
    """同时把训练日志发送到终端和文件。"""

    def __init__(self, terminal, logfile) -> None:
        """接收原标准流与日志流。"""
        self.terminal, self.logfile = terminal, logfile

    def write(self, message: str) -> int:
        """写入并立即刷新日志。"""
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()
        return len(message)

    def flush(self) -> None:
        """刷新两个输出流。"""
        self.terminal.flush()
        self.logfile.flush()

    def isatty(self) -> bool:
        """关闭 Rich/TQDM 对日志文件的彩色终端假设。"""
        return False

    @property
    def encoding(self):
        """报告原标准流字符编码。"""
        return getattr(self.terminal, "encoding", "utf-8")


def main(argv=None) -> None:
    """入口：校验参数、确定输出位置、记录运行日志。"""
    args = get_args(argv)
    validate_args(args)
    root, output = resolve_paths(args)
    if not (root / "train").is_dir():
        raise FileNotFoundError(f"找不到训练数据目录：{root / 'train'}")
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / f"training_log_{datetime.now():%Y%m%d_%H%M%S}.txt"
    with log_path.open("w", encoding="utf-8", buffering=1) as logfile:
        with redirect_stdout(Tee(sys.stdout, logfile)), redirect_stderr(Tee(sys.stderr, logfile)):
            print(f"数据目录：{root}\n输出目录：{output}\n日志路径：{log_path}")
            try:
                run(args)
            except Exception:
                traceback.print_exc()
                print("训练未完成，请检查上述异常，已有输出不会自动覆盖。")
                raise
            else:
                print("训练结束。")


if __name__ == "__main__":
    main()
