import os
import torch
import torch.cuda.amp as amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import (
    DEVICE, USE_AMP, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    LOG_INTERVAL, VAL_EPOCH_INTERVAL, CHECKPOINT_DIR, BATCH_SIZE, MAX_CAPTION_LENGTH,
    PIN_MEMORY,
)
from data_loader import create_dataloaders
from model import MiniBLIP2


def print_gpu_info():
    if not torch.cuda.is_available():
        print("GPU: 不可用，使用 CPU 训练")
        return
    print(f"GPU:  {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA: {torch.version.cuda}")


def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print_gpu_info()
    print()

    print("Loading data...")
    train_loader, val_loader, tokenizer = create_dataloaders()

    print("Building model...")
    model = MiniBLIP2().to(DEVICE)

    # Only train Q-Former + Projection
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Mixed precision scaler
    scaler = amp.GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Device:  {DEVICE}")
    print(f"AMP:     {'on' if scaler.is_enabled() else 'off'}")
    print(f"Epochs:  {EPOCHS}, LR: {LEARNING_RATE}, Batch size: {BATCH_SIZE}")

    best_loss = float("inf")
    history = []

    for epoch in range(1, EPOCHS + 1):
        # ── Training ──
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for batch_idx, (pixel_values, input_ids, attention_mask) in enumerate(pbar):
            pixel_values = pixel_values.to(DEVICE, non_blocking=PIN_MEMORY)
            input_ids = input_ids.to(DEVICE, non_blocking=PIN_MEMORY)
            attention_mask = attention_mask.to(DEVICE, non_blocking=PIN_MEMORY)

            optimizer.zero_grad()

            with amp.autocast(enabled=scaler.is_enabled()):
                loss = model(pixel_values, input_ids, attention_mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if (batch_idx + 1) % LOG_INTERVAL == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        history.append((epoch, avg_loss))
        print(f"Epoch {epoch:3d} | Train Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()

        # ── Save best checkpoint ──
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "loss": avg_loss,
            }, os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            print(f"  → Saved best checkpoint (loss={avg_loss:.4f})")

        # ── Validation (sample generation) ──
        if epoch % VAL_EPOCH_INTERVAL == 0:
            model.eval()
            val_pixels, val_ids, val_mask = next(iter(val_loader))
            val_pixels = val_pixels[:3].to(DEVICE)
            val_ids = val_ids[:3]
            val_mask = val_mask[:3]

            print("  Validation samples:")
            for i in range(len(val_pixels)):
                true_caption = tokenizer.decode(
                    val_ids[i][val_mask[i] == 1], skip_special_tokens=True
                )
                gen_caption = model.generate(val_pixels[i:i+1], tokenizer)
                print(f"    [{i}] True: {true_caption}")
                print(f"        Gen:  {gen_caption}")

        # ── Save checkpoint every 10 epochs ──
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "loss": avg_loss,
            }, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pt"))

    # ── Save loss history ──
    print("\nTraining complete!")
    print("Loss history:")
    for ep, loss in history:
        print(f"  Epoch {ep:3d}: {loss:.4f}")

    return model, history


if __name__ == "__main__":
    train()
