"""
PyTorch Neural Architectures for Sentiment Text Classification.
Implements:
1. Log-Linear Softmax Regressor (Multinomial Logistic Regression baseline)
2. Single-Hidden Shallow MLP (Tutorial baseline)
3. Deep Residual MLP (Ablation: Failed depth without sequence modeling)
4. In-Domain Self-Supervised Subword Embeddings (Shifted PPMI + TruncatedSVD, Levy & Goldberg 2014)
5. BiLSTM with Multi-Scale Conv Stem, Spatial Dropout & Multi-Pooling (Learned bidirectional sequential composition)
6. Twitter Transformer Encoder from Scratch (Multi-Head Self-Attention + Positional Embeddings + Multi-Pooling)
"""

from typing import Optional
from pathlib import Path
import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

RESULTS_DIR = Path("Assignment1/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_EMBED_PATH = RESULTS_DIR / "sppmi_svd_embeds_8192.npy"


def get_or_train_in_domain_embeddings(
    tokenizer: Optional[Tokenizer] = None,
    vocab_size: int = 8192,
    embed_dim: int = 128,
    cache_path: Path = DEFAULT_EMBED_PATH
) -> np.ndarray:
    """
    Self-Supervised In-Domain Subword Embedding Pre-training (ZERO External Data!).
    Implements Shifted Positive Pointwise Mutual Information (SPPMI) + Truncated SVD (Levy & Goldberg, 2014).
    Mathematically equivalent to Word2Vec Skip-Gram with Negative Sampling (SGNS), deterministic, and fast.
    """
    if cache_path.exists():
        embeds = np.load(cache_path)
        if embeds.shape == (vocab_size, embed_dim):
            return embeds

    if tokenizer is None:
        tok_path = RESULTS_DIR / "tokenizers/byte_bpe_8192.json"
        if not tok_path.exists():
            tok_path = RESULTS_DIR / "tokenizers/bpe_vocab_8192.json"
        tokenizer = Tokenizer.from_file(str(tok_path))

    from Assignment1.src.tokenizer import get_multi_domain_twitter_corpus
    corpus = get_multi_domain_twitter_corpus()

    encodings = tokenizer.encode_batch(corpus)
    window = 5
    rows, cols, weights = [], [], []
    V = vocab_size

    for enc in encodings:
        ids = enc.ids
        L = len(ids)
        for i in range(L):
            w1 = ids[i]
            if w1 >= V:
                continue
            for j in range(max(0, i - window), min(L, i + window + 1)):
                if i != j:
                    w2 = ids[j]
                    if w2 < V:
                        rows.append(w1)
                        cols.append(w2)
                        weights.append(1.0 / abs(i - j))

    C = sp.csr_matrix((weights, (rows, cols)), shape=(V, V), dtype=np.float32)
    row_sum = np.array(C.sum(axis=1)).flatten()
    col_sum = np.array(C.sum(axis=0)).flatten()
    total_sum = row_sum.sum()

    k_neg = 5.0
    C_coo = C.tocoo()
    val = C_coo.data
    r = C_coo.row
    c = C_coo.col

    pmi = np.log((val * total_sum) / (row_sum[r] * col_sum[c] + 1e-10)) - np.log(k_neg)
    sppmi_mask = pmi > 0
    sppmi = sp.csr_matrix((pmi[sppmi_mask], (r[sppmi_mask], c[sppmi_mask])), shape=(V, V), dtype=np.float32)

    # Truncated SVD
    u, s, _ = sp.linalg.svds(sppmi, k=embed_dim)
    embeds = (u * np.sqrt(s)).astype(np.float32)
    embeds[0] = 0.0  # padding token
    std = np.std(embeds[1:])
    if std > 0:
        embeds[1:] = (embeds[1:] / std) * 0.1

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeds)
    print(f"Computed & cached in-domain PPMI-SVD embeddings: shape {embeds.shape} to {cache_path}")
    return embeds


def init_weights_modern(m):
    """Modern weight initialization."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=0.0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.padding_idx is not None:
            with torch.no_grad():
                m.weight[m.padding_idx].fill_(0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# ==============================================================================
# BASELINE 1: LOG-LINEAR MODEL
# ==============================================================================
class LogLinearModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 3):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ==============================================================================
# BASELINE 2: SHALLOW MLP (1 HIDDEN LAYER)
# ==============================================================================
class ShallowMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.apply(init_weights_modern)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.activation(h)
        h = self.dropout(h)
        return self.fc2(h)


# ==============================================================================
# ABLATION 3: DEEP RESIDUAL MLP (BAG-OF-WORDS)
# ==============================================================================
class DeepResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 3, num_res_blocks: int = 2, dropout: float = 0.25):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout)
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout)
            ) for _ in range(num_res_blocks)
        ])
        self.head_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, num_classes)
        self.apply(init_weights_modern)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)
        h = self.head_norm(h)
        return self.output_head(h)


# ==============================================================================
# ATTENTION POOLING MODULE
# ==============================================================================
class AttentionPooling(nn.Module):
    """
    Self-Attention Pooling head.
    Computes a learned scalar score for each token step, enabling the model
    to dynamically weight sentiment-carrying tokens (e.g. 'not bad', 'love', 'awful').
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.att_linear = nn.Linear(hidden_dim, 64)
        self.v = nn.Linear(64, 1, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h: (batch, seq_len, hidden_dim), mask: (batch, seq_len) True for valid tokens
        score = torch.tanh(self.att_linear(h))  # (batch, seq_len, 64)
        score = self.v(score).squeeze(-1)       # (batch, seq_len)
        score = score.masked_fill(~mask, -1e4)
        alpha = F.softmax(score, dim=-1).unsqueeze(-1)  # (batch, seq_len, 1)
        context = torch.sum(h * alpha, dim=1)           # (batch, hidden_dim)
        return context


# ==============================================================================
# CHAMPION 1: BIDIRECTIONAL LSTM WITH CONV STEM & ATTENTION/MAX/MEAN POOLING
# ==============================================================================
class BiLSTMAttention(nn.Module):
    """
    Upgraded Bidirectional LSTM with:
    1. In-domain self-supervised PPMI-SVD subword embedding initialization
    2. Spatial Dropout (drops entire embedding feature channels)
    3. Multi-Scale Conv1D stem (residual feature enhancement)
    4. 2-layer Bidirectional LSTM
    5. Tri-Pooling concatenation: Attention Pooling + Max Pooling + Mean Pooling
    6. Multi-Sample Dropout classification head
    """
    def __init__(
        self,
        vocab_size: int = 8192,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.25,
        padding_idx: int = 0,
        pretrained_embeddings: Optional[np.ndarray] = None
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        else:
            try:
                emb_data = get_or_train_in_domain_embeddings(vocab_size=vocab_size, embed_dim=embedding_dim)
                self.embedding.weight.data.copy_(torch.from_numpy(emb_data))
            except Exception as e:
                print(f"Warning: Could not initialize PPMI-SVD embeddings: {e}")
                nn.init.normal_(self.embedding.weight, mean=0.0, std=0.05)

        self.spatial_drop = nn.Dropout1d(p=dropout)
        self.conv = nn.Sequential(
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU()
        )
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = AttentionPooling(hidden_dim * 2)
        # Tri-pooling: Attn (2*hidden) + Max (2*hidden) + Mean (2*hidden) = 6*hidden
        self.pool_proj = nn.Sequential(
            nn.Linear(hidden_dim * 6, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
        # Multi-sample dropout heads
        self.drop_heads = nn.ModuleList([nn.Dropout(dropout) for _ in range(5)])
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, seq_len)
        mask = (input_ids != 0)
        emb = self.embedding(input_ids)  # (batch, seq_len, embed_dim)

        # Spatial dropout across channels
        emb = self.spatial_drop(emb.transpose(1, 2)).transpose(1, 2)

        # Conv1D feature stem with residual
        conv_out = self.conv(emb.transpose(1, 2)).transpose(1, 2)
        h_in = emb + conv_out

        lstm_out, _ = self.lstm(h_in)  # (batch, seq_len, 2 * hidden_dim)

        # 1. Attention Pooling
        attn_pool = self.attention(lstm_out, mask)
        # 2. Masked Max Pooling
        masked_lstm = lstm_out.masked_fill(~mask.unsqueeze(-1), -1e4)
        max_pool = torch.max(masked_lstm, dim=1)[0]
        # 3. Masked Mean Pooling
        denom = torch.clamp(mask.unsqueeze(-1).sum(dim=1), min=1.0)
        mean_pool = (lstm_out * mask.unsqueeze(-1)).sum(dim=1) / denom

        pooled = torch.cat([attn_pool, max_pool, mean_pool], dim=-1)
        h = self.pool_proj(pooled)

        if self.training:
            logits_list = [self.classifier(drop(h)) for drop in self.drop_heads]
            return torch.stack(logits_list, dim=0).mean(dim=0)
        else:
            return self.classifier(h)


# ==============================================================================
# CHAMPION 2: TWITTER TRANSFORMER ENCODER (FROM SCRATCH)
# ==============================================================================
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, d_ff: int = 256, dropout: float = 0.15):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class TwitterTransformerEncoder(nn.Module):
    """
    Compact Multi-Head Self-Attention Transformer Encoder trained strictly from scratch:
    1. In-domain PPMI-SVD subword embedding initialization
    2. Sinusoidal Positional Encoding + Spatial Dropout
    3. Pre-LayerNorm Transformer blocks
    4. Tri-Pooling: Attention Pooling + Max Pooling + Mean Pooling (identical to BiLSTM)
    5. Multi-Sample Dropout classification head
    """
    def __init__(
        self,
        vocab_size: int = 8192,
        max_seq_len: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        num_classes: int = 3,
        dropout: float = 0.15,
        padding_idx: int = 0,
        pretrained_embeddings: Optional[np.ndarray] = None
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        if pretrained_embeddings is not None:
            self.token_emb.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        else:
            try:
                emb_data = get_or_train_in_domain_embeddings(vocab_size=vocab_size, embed_dim=d_model)
                self.token_emb.weight.data.copy_(torch.from_numpy(emb_data))
            except Exception as e:
                print(f"Warning: Could not initialize PPMI-SVD embeddings: {e}")
                nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.05)

        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.spatial_drop = nn.Dropout1d(dropout)

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.attention = AttentionPooling(d_model)
        self.pool_proj = nn.Sequential(
            nn.Linear(d_model * 3, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        self.drop_heads = nn.ModuleList([nn.Dropout(dropout) for _ in range(5)])
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        mask = (input_ids != 0)
        key_padding_mask = ~mask

        x = self.token_emb(input_ids)
        x = self.spatial_drop(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.pos_emb(positions)

        for layer in self.layers:
            x = layer(x, key_padding_mask)

        x = self.ln_final(x)

        attn_pool = self.attention(x, mask)
        masked_x = x.masked_fill(~mask.unsqueeze(-1), -1e4)
        max_pool = torch.max(masked_x, dim=1)[0]
        denom = torch.clamp(mask.unsqueeze(-1).sum(dim=1), min=1.0)
        mean_pool = (x * mask.unsqueeze(-1)).sum(dim=1) / denom

        pooled = torch.cat([attn_pool, max_pool, mean_pool], dim=-1)
        h = self.pool_proj(pooled)

        if self.training:
            logits_list = [self.classifier(drop(h)) for drop in self.drop_heads]
            return torch.stack(logits_list, dim=0).mean(dim=0)
        else:
            return self.classifier(h)


def build_model(model_name: str, input_dim: int = 8192, num_classes: int = 3, **kwargs) -> nn.Module:
    model_name = model_name.lower()
    if model_name in ["loglin", "logistic"]:
        return LogLinearModel(input_dim, num_classes=num_classes)
    elif model_name in ["shallow_mlp", "smlp", "mlp"]:
        return ShallowMLP(input_dim, hidden_dim=256, num_classes=num_classes, dropout=kwargs.get("dropout", 0.2))
    elif model_name in ["deep_mlp", "deepmlp"]:
        return DeepResidualMLP(input_dim, hidden_dim=256, num_classes=num_classes, dropout=kwargs.get("dropout", 0.25))
    elif model_name in ["bilstm", "bilstm_attention", "lstm"]:
        return BiLSTMAttention(
            vocab_size=input_dim,
            embedding_dim=128,
            hidden_dim=128,
            num_classes=num_classes,
            dropout=kwargs.get("dropout", 0.25),
            pretrained_embeddings=kwargs.get("pretrained_embeddings", None)
        )
    elif model_name in ["transformer", "transformer_encoder", "custom_transformer", "transformer_scratch"]:
        return TwitterTransformerEncoder(
            vocab_size=input_dim,
            max_seq_len=64,
            d_model=128,
            n_heads=4,
            n_layers=2,
            num_classes=num_classes,
            dropout=kwargs.get("dropout", 0.15),
            pretrained_embeddings=kwargs.get("pretrained_embeddings", None)
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing PyTorch models on {device}...")

    sparse_input = torch.randn(8, 8192).to(device)
    seq_input = torch.randint(0, 8192, (8, 64)).to(device)

    for name in ["loglin", "shallow_mlp", "deep_mlp", "bilstm", "transformer"]:
        m = build_model(name, input_dim=8192).to(device)
        inp = seq_input if name in ["bilstm", "transformer"] else sparse_input
        out = m(inp)
        params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"Model: {name:20s} | Trainable Params: {params:,} | Output shape: {list(out.shape)}")
