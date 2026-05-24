import torch
import torch.nn as nn


class MiniQFormerLayer(nn.Module):
    """One layer of Q-Former: Self-Attn → Cross-Attn → FFN, pre-norm style."""

    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, queries, image_embeds):
        # Self-attention: queries attend to themselves
        q_normed = self.norm1(queries)
        self_attn_out, _ = self.self_attn(q_normed, q_normed, q_normed)
        queries = queries + self_attn_out

        # Cross-attention: queries attend to image features (frozen encoder output)
        q_normed = self.norm2(queries)
        cross_attn_out, _ = self.cross_attn(q_normed, image_embeds, image_embeds)
        queries = queries + cross_attn_out

        # Feed-forward
        q_normed = self.norm3(queries)
        ffn_out = self.ffn(q_normed)
        queries = queries + ffn_out

        return queries


class MiniQFormer(nn.Module):
    """
    Mini Q-Former: a lightweight transformer that compresses image features
    into a fixed number of query representations via cross-attention.

    Args:
        vision_hidden_size: dimension of input image features (e.g. 768 for CLIP ViT-B/32)
        hidden_dim: internal hidden dimension
        num_queries: number of learnable query tokens
        num_layers: number of Q-Former layers
        num_heads: attention heads
        ff_dim: feed-forward hidden dimension
        dropout: dropout rate
    """

    def __init__(self, vision_hidden_size, hidden_dim=256, num_queries=8,
                 num_layers=2, num_heads=8, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Project image features to hidden_dim if they differ
        self.vision_proj = nn.Linear(vision_hidden_size, hidden_dim) if vision_hidden_size != hidden_dim else nn.Identity()

        # Learnable query embeddings
        self.query_embeds = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)

        # Q-Former layers
        self.layers = nn.ModuleList([
            MiniQFormerLayer(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, image_embeds):
        """
        Args:
            image_embeds: [B, num_patches, vision_hidden_size] from frozen vision encoder
        Returns:
            query_outputs: [B, num_queries, hidden_dim]
        """
        B = image_embeds.shape[0]

        # Project image features
        image_embeds = self.vision_proj(image_embeds)  # [B, num_patches, hidden_dim]

        # Expand learnable queries to batch
        queries = self.query_embeds.expand(B, -1, -1)  # [B, num_queries, hidden_dim]

        # Pass through Q-Former layers
        for layer in self.layers:
            queries = layer(queries, image_embeds)

        return queries


if __name__ == "__main__":
    qformer = MiniQFormer(
        vision_hidden_size=768,
        hidden_dim=256,
        num_queries=8,
        num_layers=2,
        num_heads=8,
        ff_dim=1024,
    )
    dummy_image_embeds = torch.randn(2, 50, 768)  # batch=2, 50 patches, 768 dim
    output = qformer(dummy_image_embeds)
    print(f"Input shape:  {dummy_image_embeds.shape}")
    print(f"Output shape: {output.shape}")  # expected: [2, 8, 256]
    print(f"Trainable params: {sum(p.numel() for p in qformer.parameters() if p.requires_grad):,}")
