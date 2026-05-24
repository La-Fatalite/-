"""
Mini-BLIP2 模型 — 整合 Vision Encoder + Q-Former + Projection + OPT Decoder。

Pipeline:
    Image → Frozen CLIP ViT → Trainable Mini Q-Former → Projection → Frozen OPT-125M → Caption

训练时只更新 Mini Q-Former 和 Projection Layer。
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    VISION_MODEL_NAME, LANGUAGE_MODEL_NAME, DEVICE,
    QF_NUM_QUERIES, QF_HIDDEN_DIM, QF_NUM_LAYERS, QF_NUM_HEADS, QF_FF_DIM, QF_DROPOUT,
    VISION_HIDDEN_SIZE, MAX_CAPTION_LENGTH,
)
from qformer import MiniQFormer
from vision_encoder import VisionEncoder


class MiniBLIP2(nn.Module):
    """
    Mini-BLIP2 for Image Captioning.

    Pipeline:
        Image → Frozen CLIP ViT → Mini Q-Former → Projection → Frozen OPT-125M → Caption
    """

    def __init__(self):
        super().__init__()

        # ── Frozen Vision Encoder ──
        self.vision_encoder = VisionEncoder(model_name=VISION_MODEL_NAME)

        # ── Trainable Mini Q-Former ──
        self.qformer = MiniQFormer(
            vision_hidden_size=VISION_HIDDEN_SIZE,
            hidden_dim=QF_HIDDEN_DIM,
            num_queries=QF_NUM_QUERIES,
            num_layers=QF_NUM_LAYERS,
            num_heads=QF_NUM_HEADS,
            ff_dim=QF_FF_DIM,
            dropout=QF_DROPOUT,
        )

        # ── Projection Layer改变维度──
        self.projection = nn.Linear(QF_HIDDEN_DIM, VISION_HIDDEN_SIZE)

        # ── Frozen Language Decoder语言解码器 ──
        self.language_model = AutoModelForCausalLM.from_pretrained(LANGUAGE_MODEL_NAME)
        for param in self.language_model.parameters():
            param.requires_grad = False

        self.llm_dtype = next(self.language_model.parameters()).dtype

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract image features and compress into query representations."""
        image_embeds = self.vision_encoder(pixel_values)       # [B, 50, 768]
        query_outputs = self.qformer(image_embeds)              # [B, num_queries, hidden_dim]
        visual_prefix = self.projection(query_outputs)           # [B, num_queries, 768]
        return visual_prefix

    def forward(self, pixel_values: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values:   [B, 3, 224, 224]
            input_ids:      [B, max_len]  caption token ids
            attention_mask: [B, max_len]  1 for real tokens, 0 for padding
        Returns:
            loss: scalar cross-entropy loss
        """
        B, max_len = input_ids.shape
        num_queries = QF_NUM_QUERIES

        visual_prefix = self.encode_image(pixel_values)  # [B, num_queries, 768]

        # Get text embeddings from frozen OPT (may be float16)
        embed_layer = self.language_model.model.decoder.embed_tokens
        with torch.no_grad():
            text_embeds = embed_layer(input_ids)  # [B, max_len, 768]

        # Align dtypes: projection output (float32) → match OPT (float16)
        visual_prefix = visual_prefix.to(dtype=text_embeds.dtype)

        # Combine: visual prefix + text tokens (drop last text token for input)
        combined_embeds = torch.cat([
            visual_prefix,
            text_embeds[:, :-1, :],
        ], dim=1)  # [B, num_queries + max_len - 1, 768]

        # Build combined attention mask
        visual_mask = torch.ones(B, num_queries, device=input_ids.device,
                                 dtype=attention_mask.dtype)
        combined_mask = torch.cat([
            visual_mask,
            attention_mask[:, :-1],
        ], dim=1)  # [B, num_queries + max_len - 1]

        # Forward through frozen OPT decoder
        opt_outputs = self.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
        )
        logits = opt_outputs.logits  # [B, num_queries+max_len-1, vocab_size]

        # Extract logits at text positions: position (num_queries-1) predicts input_ids[0]
        text_logits = logits[:, num_queries - 1:, :]  # [B, max_len, vocab_size]

        # Compute cross-entropy, ignoring padding
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        loss = nn.functional.cross_entropy(
            text_logits.reshape(-1, text_logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )
        return loss

    @torch.no_grad()
    def generate(self, pixel_values: torch.Tensor, tokenizer,
                 max_length: int = MAX_CAPTION_LENGTH) -> str:
        """
        Generate a caption for an image (greedy decoding).

        Args:
            pixel_values: [1, 3, 224, 224] single image
            tokenizer: OPT tokenizer
            max_length: max tokens to generate
        Returns:
            generated caption string
        """
        self.eval()
        visual_prefix = self.encode_image(pixel_values)  # [1, num_queries, 768]

        # Start with BOS token
        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            bos_id = tokenizer.eos_token_id
        generated_ids = [bos_id]

        embed_layer = self.language_model.model.decoder.embed_tokens

        for _ in range(max_length):
            with torch.no_grad():
                text_embeds = embed_layer(
                    torch.tensor([generated_ids], device=pixel_values.device)
                )
            # Align dtype
            visual = visual_prefix.to(dtype=text_embeds.dtype)
            combined = torch.cat([visual, text_embeds], dim=1)
            mask = torch.ones(1, combined.size(1), device=pixel_values.device)

            outputs = self.language_model(inputs_embeds=combined, attention_mask=mask)
            next_logits = outputs.logits[0, -1, :]
            next_token = next_logits.argmax(dim=-1).item()

            if next_token == tokenizer.eos_token_id:
                break
            generated_ids.append(next_token)

        caption = tokenizer.decode(generated_ids[1:], skip_special_tokens=True)
        return caption.strip()

    @torch.no_grad()
    def generate_beam(self, pixel_values: torch.Tensor, tokenizer,
                      max_length: int = MAX_CAPTION_LENGTH,
                      num_beams: int = 3) -> str:
        """
        Generate a caption using beam search.

        Args:
            pixel_values: [1, 3, 224, 224]
            tokenizer: OPT tokenizer
            max_length: max tokens to generate
            num_beams: beam size
        Returns:
            generated caption string
        """
        self.eval()
        visual_prefix = self.encode_image(pixel_values)  # [1, num_queries, 768]

        bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        eos_id = tokenizer.eos_token_id
        embed_layer = self.language_model.model.decoder.embed_tokens

        # Each beam: (token_ids, log_prob, finished)
        beams = [([bos_id], 0.0, False)]

        for _ in range(max_length):
            candidates = []

            for token_ids, log_prob, finished in beams:
                if finished:
                    candidates.append((token_ids, log_prob, True))
                    continue

                text_embeds = embed_layer(
                    torch.tensor([token_ids], device=pixel_values.device)
                )
                visual = visual_prefix.to(dtype=text_embeds.dtype)
                combined = torch.cat([visual, text_embeds], dim=1)
                mask = torch.ones(1, combined.size(1), device=pixel_values.device)

                outputs = self.language_model(inputs_embeds=combined, attention_mask=mask)
                next_logits = outputs.logits[0, -1, :]
                log_probs = torch.log_softmax(next_logits, dim=-1)

                topk_scores, topk_tokens = log_probs.topk(num_beams)

                for token, score in zip(topk_tokens, topk_scores):
                    tok = token.item()
                    new_ids = token_ids + [tok]
                    new_score = log_prob + score.item()
                    done = tok == eos_id
                    candidates.append((new_ids, new_score, done))

            # Keep top num_beams, normalize by length to avoid bias towards short seqs
            candidates.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
            beams = candidates[:num_beams]

            if all(b[2] for b in beams):
                break

        best = beams[0]
        caption = tokenizer.decode(best[0][1:], skip_special_tokens=True)
        return caption.strip()


if __name__ == "__main__":
    print("加载 Mini-BLIP2 模型 ...")
    model = MiniBLIP2().to(DEVICE)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数:     {total:,}")
    print(f"可训练参数: {trainable:,}")
    print(f"冻结参数:   {total - trainable:,}")

    # Test forward pass
    dummy_pixels = torch.randn(2, 3, 224, 224).to(DEVICE)
    dummy_ids = torch.randint(0, 1000, (2, 32)).to(DEVICE)
    dummy_mask = torch.ones(2, 32).to(DEVICE)

    loss = model(dummy_pixels, dummy_ids, dummy_mask)
    print(f"Loss: {loss.item():.4f}")

    # Test generate
    tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    caption = model.generate(dummy_pixels[:1], tokenizer, max_length=16)
    print(f"生成 caption: {caption}")
