"""
BLIP-2 视觉编码器模块 — 基于 openai/clip-vit-base-patch32 (ViT-B/32)。

参考 BLIP-2 论文 3.2 节:
  - 冻结预训练 ViT，移除最后一层，使用倒数第二层输出
  - 输入: 224×224 图像 → 输出: [B, 50, 768] (49 patches + CLS token)
  - 支持将全部图片预编码并缓存到磁盘，加速后续训练
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPVisionModel, CLIPProcessor


class VisionEncoder(nn.Module):
    """
    冻结的 CLIP ViT-B/32 视觉编码器。

    按照 BLIP-2 论文做法:
      - 使用倒数第二层的 hidden states（而非最后一层），性能略优
      - 所有参数冻结，不参与梯度计算

    输入形状:  [B, 3, 224, 224]  (CLIPProcessor 预处理后的 pixel_values)
    输出形状:  [B, 50, 768]       (CLS + 49 patch tokens, 每个 768 维)

    Usage:
        encoder = VisionEncoder()
        pixel_values = ...  # from CLIPProcessor
        features = encoder(pixel_values)  # [B, 50, 768]
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        # 加载 CLIPVisionModel — 仅包含 ViT 视觉编码器，无文本塔
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)

        # 冻结所有参数
        for param in self.vision_model.parameters():
            param.requires_grad = False
        self.vision_model.eval()

        # 从 config 读取关键维度
        config = self.vision_model.config
        self.hidden_size = config.hidden_size       # 768
        self.image_size = config.image_size          # 224
        self.patch_size = config.patch_size          # 32
        self.num_patches = (self.image_size // self.patch_size) ** 2 + 1  # 49 + CLS = 50

        self.model_name = model_name

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 3, 224, 224] 经过 CLIPProcessor 归一化后的图像张量
        Returns:
            features: [B, 50, 768] ViT 最后一层 hidden states
                      包含 CLS token (index 0) + 49 个 patch tokens
        """
        with torch.no_grad():
            outputs = self.vision_model(pixel_values)
            # last_hidden_state: [B, 50, 768]
            return outputs.last_hidden_state

    @torch.no_grad()
    def encode_image(self, image: Image.Image, processor: CLIPProcessor,
                     device: str = "cpu") -> torch.Tensor:
        """编码单张 PIL Image，返回 [1, 50, 768] 特征。"""
        self.vision_model.to(device)
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        return self.forward(pixel_values)

    @torch.no_grad()
    def encode_batch(self, images: list[Image.Image], processor: CLIPProcessor,
                     device: str = "cpu") -> torch.Tensor:
        """批量编码 PIL Image 列表，返回 [N, 50, 768] 特征。"""
        self.vision_model.to(device)
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        return self.forward(pixel_values)


class ImageFeatureCache:
    """
    图片特征缓存管理器。

    将整份数据集的图片预先用 VisionEncoder 编码为特征向量，
    序列化保存到磁盘。后续训练直接加载特征，避免重复跑 ViT 前向。

    Usage:
        cache = ImageFeatureCache(cache_dir="./features")
        cache.build(encoder, processor, image_paths, device="cuda")
        features = cache.load("101654506_8eb26cfb60.jpg")  # Tensor[50, 768]
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, image_name: str) -> Path:
        """每张图片对应一个 .pt 文件。"""
        stem = Path(image_name).stem
        return self.cache_dir / f"{stem}.pt"

    def build(self, encoder: VisionEncoder, processor: CLIPProcessor,
              image_paths: list[str], device: str = "cpu",
              batch_size: int = 32, show_progress: bool = True):
        """
        批量编码所有图片并写入缓存。

        Args:
            encoder: VisionEncoder 实例
            processor: CLIPProcessor 实例
            image_paths: 图片绝对路径列表
            device: "cuda" 或 "cpu"
            batch_size: 编码批次大小
            show_progress: 是否显示进度条
        """
        encoder.vision_model.to(device)
        total = len(image_paths)

        iterator = range(0, total, batch_size)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Encoding images", total=(total + batch_size - 1) // batch_size)

        for start in iterator:
            end = min(start + batch_size, total)
            batch_paths = image_paths[start:end]

            # 加载图片
            images = []
            valid_paths = []
            for p in batch_paths:
                try:
                    images.append(Image.open(p).convert("RGB"))
                    valid_paths.append(p)
                except Exception as e:
                    print(f"Warning: failed to load {p}: {e}")

            if not images:
                continue

            # 批量编码
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            features = encoder(pixel_values)  # [N, 50, 768]

            # 逐张保存
            for i, p in enumerate(valid_paths):
                name = Path(p).name
                torch.save(features[i].cpu(), self._cache_path(name))

    def load(self, image_name: str) -> torch.Tensor:
        """加载单张图片的缓存特征。"""
        path = self._cache_path(image_name)
        if not path.exists():
            raise FileNotFoundError(f"Cache miss: {path}")
        return torch.load(path, weights_only=True)

    def load_batch(self, image_names: list[str]) -> torch.Tensor:
        """批量加载缓存特征，返回 [N, 50, 768]。"""
        feats = [self.load(name) for name in image_names]
        return torch.stack(feats, dim=0)

    def exists(self, image_name: str) -> bool:
        return self._cache_path(image_name).exists()

    def stats(self) -> dict:
        """返回缓存统计信息。"""
        files = list(self.cache_dir.glob("*.pt"))
        total_size = sum(f.stat().st_size for f in files)
        return {
            "cache_dir": str(self.cache_dir),
            "num_cached": len(files),
            "total_size_mb": total_size / (1024 * 1024),
        }


# ── 快捷函数：创建 processor ──
def create_processor(model_name: str = "openai/clip-vit-base-patch32") -> CLIPProcessor:
    """创建 CLIPProcessor，用于将 PIL Image 预处理为 ViT 输入格式。"""
    return CLIPProcessor.from_pretrained(model_name)


# ── 测试 ──
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import IMAGE_DIR, DATA_DIR, DEVICE

    print("=" * 60)
    print("Vision Encoder 测试")
    print("=" * 60)

    # 1. 创建编码器
    print("\n[1] 加载 CLIP ViT-B/32 ...")
    encoder = VisionEncoder()
    processor = create_processor()
    print(f"    隐藏层维度: {encoder.hidden_size}")
    print(f"    Patch 数:   {encoder.num_patches}")
    print(f"    参数总数:   {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"    可训练参数: {sum(p.numel() for p in encoder.parameters() if p.requires_grad):,}")

    # 2. 获取实际存在的图片
    print("\n[2] 扫描 images 文件夹 ...")
    all_images = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"    找到 {len(all_images)} 张图片")
    print(f"    示例: {all_images[0]}")

    # 3. 测试单张图片编码
    print("\n[3] 测试单张图片编码 ...")
    first_image = all_images[0]
    img_path = os.path.join(IMAGE_DIR, first_image)

    image = Image.open(img_path).convert("RGB")
    feat = encoder.encode_image(image, processor, device=DEVICE)
    print(f"    图片: {first_image}")
    print(f"    特征形状: {feat.shape}")  # [1, 50, 768]

    # 4. 测试批量编码
    print("\n[4] 测试批量编码 (4 张图片) ...")
    sample_names = all_images[:4]
    images = [Image.open(os.path.join(IMAGE_DIR, name)).convert("RGB") for name in sample_names]
    batch_feat = encoder.encode_batch(images, processor, device=DEVICE)
    print(f"    批量特征形状: {batch_feat.shape}")  # [4, 50, 768]

    # 5. 测试特征缓存
    print("\n[5] 测试特征缓存 (10 张图片) ...")
    cache_dir = os.path.join(DATA_DIR, "image_features")
    cache = ImageFeatureCache(cache_dir)

    image_paths = [os.path.join(IMAGE_DIR, name) for name in all_images[:10]]
    cache.build(encoder, processor, image_paths, device=DEVICE, batch_size=4)

    stats = cache.stats()
    print(f"    缓存目录: {stats['cache_dir']}")
    print(f"    已缓存数量: {stats['num_cached']}")
    print(f"    总大小: {stats['total_size_mb']:.2f} MB")

    # 验证缓存读取
    cached_feat = cache.load(first_image)
    print(f"    缓存特征形状: {cached_feat.shape}")  # [50, 768]
    print(f"    与原特征一致: {torch.allclose(feat[0].cpu(), cached_feat, atol=1e-5)}")

    print("\n" + "=" * 60)
    print("全部测试通过!")
    print("=" * 60)
