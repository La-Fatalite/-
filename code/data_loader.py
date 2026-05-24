"""
Flickr8k 数据加载模块 — 加载前 200 张图片与对应 caption。

数据格式:
  - captions.txt: CSV 格式 (image,caption)，每张图片 5 条 caption
  - images/: 200 张 JPEG 图片
"""

import os
import csv
import random
from collections import defaultdict

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, AutoTokenizer

from config import (
    CAPTIONS_FILE, IMAGE_DIR, NUM_IMAGES, TRAIN_RATIO, BATCH_SIZE,
    VISION_MODEL_NAME, LANGUAGE_MODEL_NAME, MAX_CAPTION_LENGTH,
    NUM_WORKERS, PIN_MEMORY,
)


def load_captions(captions_file: str = CAPTIONS_FILE,
                  num_images: int = NUM_IMAGES) -> dict[str, list[str]]:
    """
    读取 captions.txt，只保留 images 文件夹中实际存在的图片。

    Returns:
        dict: image_name -> [caption1, caption2, ..., caption5]
    """
    image_captions: dict[str, list[str]] = defaultdict(list)
    with open(captions_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row["image"].strip()
            cap = row["caption"].strip()
            if img and cap:
                image_captions[img].append(cap)

    existing = set(os.listdir(IMAGE_DIR))

    result: dict[str, list[str]] = {}
    for img, caps in image_captions.items():
        if img in existing:
            result[img] = caps
        if len(result) >= num_images:
            break

    return result


def split_train_val(image_captions: dict[str, list[str]],
                    train_ratio: float = TRAIN_RATIO) -> tuple[list[str], list[str]]:
    """随机划分训练集和验证集。"""
    images = list(image_captions.keys())
    random.shuffle(images)
    split = int(len(images) * train_ratio)
    return images[:split], images[split:]


class Flickr8kDataset(Dataset):
    """
    每条样本为一个 (image_path, caption) 对，返回预处理后的 tensor。
    每张图片有 5 条 caption，各自作为独立样本。
    """

    def __init__(self, image_names: list[str],
                 image_captions: dict[str, list[str]],
                 clip_processor: CLIPProcessor,
                 tokenizer: AutoTokenizer):
        self.samples: list[tuple[str, str]] = []
        for img_name in image_names:
            for caption in image_captions[img_name]:
                self.samples.append((img_name, caption))

        self.clip_processor = clip_processor
        self.tokenizer = tokenizer
        self.max_length = MAX_CAPTION_LENGTH

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_name, caption = self.samples[idx]
        img_path = os.path.join(IMAGE_DIR, img_name)

        # 图片预处理: PIL → CLIP 归一化 tensor [3, 224, 224]
        image = Image.open(img_path).convert("RGB")
        image_inputs = self.clip_processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].squeeze(0)

        # Caption tokenization: 字符串 → token ids + attention mask
        tokenized = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        return pixel_values, input_ids, attention_mask


def create_dataloaders() -> tuple[DataLoader, DataLoader, AutoTokenizer]:
    """创建训练和验证 DataLoader，同时返回 tokenizer 供 generate 使用。"""
    image_captions = load_captions()
    train_imgs, val_imgs = split_train_val(image_captions)

    print(f"总图片数: {len(image_captions)}")
    print(f"训练集:   {len(train_imgs)} 张图片, "
          f"{sum(len(image_captions[i]) for i in train_imgs)} 条样本")
    print(f"验证集:   {len(val_imgs)} 张图片, "
          f"{sum(len(image_captions[i]) for i in val_imgs)} 条样本")

    clip_processor = CLIPProcessor.from_pretrained(VISION_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = Flickr8kDataset(train_imgs, image_captions, clip_processor, tokenizer)
    val_dataset = Flickr8kDataset(val_imgs, image_captions, clip_processor, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    return train_loader, val_loader, tokenizer


if __name__ == "__main__":
    random.seed(42)

    print("=" * 50)
    print("加载 Flickr8k 前 200 张图片与 caption")
    print("=" * 50)

    captions = load_captions()
    print(f"\n已加载 {len(captions)} 张图片的 caption")

    print("\n前 3 张图片示例:")
    for i, (img, caps) in enumerate(list(captions.items())[:3]):
        print(f"  [{i+1}] {img}")
        for c in caps:
            print(f"      - {c[:60]}...")

    train_imgs, val_imgs = split_train_val(captions)
    print(f"\n训练集: {len(train_imgs)} 张, 验证集: {len(val_imgs)} 张")

    train_loader, val_loader, tokenizer = create_dataloaders()
    pixel_values, input_ids, attention_mask = next(iter(train_loader))
    print(f"\nBatch 形状:")
    print(f"  pixel_values:   {pixel_values.shape}")
    print(f"  input_ids:      {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    print(f"\n解码示例: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")

    print("\n数据加载模块测试通过!")
