"""
Feature Representation Spaces and Sequence Encoding for TweetEval Sentiment.
Implements:
1. Negation Scope Marking (Das & Chen 2007; Pang et al. 2002)
2. Token Indicator (Binary Presence)
3. Bag-of-Words (Term Frequency Counts)
4. Subword TF-IDF (L2-Normalized TF-IDF with smoothed IDF)
5. Subword N-Gram TF-IDF (Unigram + Bigram subwords with sublinear TF scaling 1 + log(tf))
6. Subword N-Gram BoW (Unigram + Bigram counts with sublinear TF)
7. Sequence Tensor Encoder (Fixed-length padded token sequences for BiLSTM and Transformer)
"""

from typing import List, Tuple, Optional, Dict
from collections import Counter
import re
import numpy as np
import scipy.sparse as sp
import torch
from tokenizers import Tokenizer
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import load_dataset

FIGURES_DIR = Path("Assignment1/slides/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

NEGATION_CUES = {
    "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
    "cannot", "cant", "wont", "dont", "isnt", "arent", "wasnt", "werent",
    "hasnt", "havent", "hadnt", "couldnt", "shouldnt", "wouldnt", "without"
}
PUNCT_SPLIT = re.compile(r"^[.,!?;:]+$")


def mark_negation_scope(text: str) -> str:
    """
    Appends '_NEG' suffix to words governed by a negation cue up to punctuation.
    Follows standard sentiment analysis convention (Das & Chen 2007; Pang et al. 2002).
    """
    tokens = text.split()
    neg_active = False
    marked_tokens = []
    for token in tokens:
        clean_token = token.strip(".,!?;:\"'()[]{}")
        lower_token = clean_token.lower()
        has_punct = any(token.endswith(p) for p in [".", ",", "!", "?", ";", ":"])
        
        if lower_token in NEGATION_CUES or lower_token.endswith("n't") or lower_token.endswith("nt") and lower_token in ["dont", "cant", "wont", "didnt"]:
            neg_active = True
            marked_tokens.append(token)
        elif neg_active:
            marked_tokens.append(token + "_NEG")
            if has_punct:
                neg_active = False
        else:
            marked_tokens.append(token)
    return " ".join(marked_tokens)


class TokenIndicator:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()

    def transform(self, texts: List[str]) -> sp.csr_matrix:
        encodings = self.tokenizer.encode_batch(texts)
        rows, cols = [], []
        for row_idx, enc in enumerate(encodings):
            for token_id in set(enc.ids):
                if token_id < self.vocab_size:
                    rows.append(row_idx)
                    cols.append(token_id)
        data = np.ones(len(rows), dtype=np.float32)
        return sp.csr_matrix((data, (rows, cols)), shape=(len(texts), self.vocab_size))


class BagOfWords:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()

    def transform(self, texts: List[str]) -> sp.csr_matrix:
        encodings = self.tokenizer.encode_batch(texts)
        rows, cols, data = [], [], []
        for row_idx, enc in enumerate(encodings):
            counts = {}
            for token_id in enc.ids:
                if token_id < self.vocab_size:
                    counts[token_id] = counts.get(token_id, 0) + 1
            for token_id, c in counts.items():
                rows.append(row_idx)
                cols.append(token_id)
                data.append(c)
        return sp.csr_matrix((np.array(data, dtype=np.float32), (rows, cols)), shape=(len(texts), self.vocab_size))


class SubwordTfidf:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.idf_ = None
        self.is_fitted = False

    def fit(self, texts: List[str]):
        encodings = self.tokenizer.encode_batch(texts)
        N = len(texts)
        df = np.zeros(self.vocab_size, dtype=np.float64)
        for enc in encodings:
            for token_id in set(enc.ids):
                if token_id < self.vocab_size:
                    df[token_id] += 1
        self.idf_ = np.log((1.0 + N) / (1.0 + df)) + 1.0
        self.is_fitted = True
        return self

    def transform(self, texts: List[str]) -> sp.csr_matrix:
        if not self.is_fitted:
            raise ValueError("SubwordTfidf must be fitted first!")
        encodings = self.tokenizer.encode_batch(texts)
        rows, cols, data = [], [], []
        for row_idx, enc in enumerate(encodings):
            counts = {}
            for token_id in enc.ids:
                if token_id < self.vocab_size:
                    counts[token_id] = counts.get(token_id, 0) + 1
            for token_id, c in counts.items():
                rows.append(row_idx)
                cols.append(token_id)
                data.append(c * self.idf_[token_id])

        tfidf_mat = sp.csr_matrix((np.array(data, dtype=np.float32), (rows, cols)), shape=(len(texts), self.vocab_size))
        norm = sp.linalg.norm(tfidf_mat, axis=1)
        norm[norm == 0] = 1.0
        norm_inv = sp.diags(1.0 / np.array(norm).ravel())
        return norm_inv @ tfidf_mat

    def fit_transform(self, texts: List[str]) -> sp.csr_matrix:
        self.fit(texts)
        return self.transform(texts)


class SubwordNgramTfidf:
    """
    Subword N-Gram TF-IDF with (1, 2)-grams, sublinear term-frequency scaling
    tf_scaled = 1 + log(tf), and smoothed inverse document frequency.
    Extracts unigram subwords + frequent subword bigram transitions.
    """
    def __init__(self, tokenizer: Tokenizer, max_features: int = 16384, sublinear_tf: bool = True):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.bigram_vocab: Dict[Tuple[int, int], int] = {}
        self.idf_: Optional[np.ndarray] = None
        self.total_dim = self.vocab_size
        self.is_fitted = False

    def fit(self, texts: List[str]):
        encodings = self.tokenizer.encode_batch(texts)
        N = len(texts)
        
        # Mine most frequent subword bigrams
        max_bigrams = max(0, self.max_features - self.vocab_size)
        bg_counter = Counter()
        for enc in encodings:
            ids = enc.ids
            for i in range(len(ids) - 1):
                t1, t2 = ids[i], ids[i+1]
                if t1 < self.vocab_size and t2 < self.vocab_size:
                    bg_counter[(t1, t2)] += 1
        
        self.bigram_vocab = {bg: idx for idx, (bg, _) in enumerate(bg_counter.most_common(max_bigrams))}
        self.total_dim = self.vocab_size + len(self.bigram_vocab)
        
        # Compute document frequencies
        df = np.zeros(self.total_dim, dtype=np.float64)
        for enc in encodings:
            seen = set()
            ids = enc.ids
            for t in ids:
                if t < self.vocab_size:
                    seen.add(t)
            for i in range(len(ids) - 1):
                bg = (ids[i], ids[i+1])
                if bg in self.bigram_vocab:
                    seen.add(self.vocab_size + self.bigram_vocab[bg])
            for feat_idx in seen:
                df[feat_idx] += 1
                
        self.idf_ = np.log((1.0 + N) / (1.0 + df)) + 1.0
        self.is_fitted = True
        return self

    def transform(self, texts: List[str]) -> sp.csr_matrix:
        if not self.is_fitted:
            raise ValueError("SubwordNgramTfidf must be fitted first!")
        encodings = self.tokenizer.encode_batch(texts)
        rows, cols, data = [], [], []
        
        for row_idx, enc in enumerate(encodings):
            counts = Counter()
            ids = enc.ids
            for t in ids:
                if t < self.vocab_size:
                    counts[t] += 1
            for i in range(len(ids) - 1):
                bg = (ids[i], ids[i+1])
                if bg in self.bigram_vocab:
                    counts[self.vocab_size + self.bigram_vocab[bg]] += 1
            
            for feat_idx, cnt in counts.items():
                tf_val = (1.0 + np.log(float(cnt))) if self.sublinear_tf else float(cnt)
                weight = tf_val * self.idf_[feat_idx]
                rows.append(row_idx)
                cols.append(feat_idx)
                data.append(weight)
                
        mat = sp.csr_matrix((np.array(data, dtype=np.float32), (rows, cols)), shape=(len(texts), self.total_dim))
        norm = sp.linalg.norm(mat, axis=1)
        norm[norm == 0] = 1.0
        norm_inv = sp.diags(1.0 / np.array(norm).ravel())
        return norm_inv @ mat

    def fit_transform(self, texts: List[str]) -> sp.csr_matrix:
        self.fit(texts)
        return self.transform(texts)


class SubwordNgramBoW:
    """
    Subword (1, 2)-gram Bag-of-Words with sublinear term-frequency scaling.
    """
    def __init__(self, tokenizer: Tokenizer, max_features: int = 16384, sublinear_tf: bool = True):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.bigram_vocab: Dict[Tuple[int, int], int] = {}
        self.total_dim = self.vocab_size
        self.is_fitted = False

    def fit(self, texts: List[str]):
        encodings = self.tokenizer.encode_batch(texts)
        max_bigrams = max(0, self.max_features - self.vocab_size)
        bg_counter = Counter()
        for enc in encodings:
            ids = enc.ids
            for i in range(len(ids) - 1):
                t1, t2 = ids[i], ids[i+1]
                if t1 < self.vocab_size and t2 < self.vocab_size:
                    bg_counter[(t1, t2)] += 1
        self.bigram_vocab = {bg: idx for idx, (bg, _) in enumerate(bg_counter.most_common(max_bigrams))}
        self.total_dim = self.vocab_size + len(self.bigram_vocab)
        self.is_fitted = True
        return self

    def transform(self, texts: List[str]) -> sp.csr_matrix:
        encodings = self.tokenizer.encode_batch(texts)
        rows, cols, data = [], [], []
        for row_idx, enc in enumerate(encodings):
            counts = Counter()
            ids = enc.ids
            for t in ids:
                if t < self.vocab_size:
                    counts[t] += 1
            for i in range(len(ids) - 1):
                bg = (ids[i], ids[i+1])
                if bg in self.bigram_vocab:
                    counts[self.vocab_size + self.bigram_vocab[bg]] += 1
            for feat_idx, cnt in counts.items():
                val = (1.0 + np.log(float(cnt))) if self.sublinear_tf else float(cnt)
                rows.append(row_idx)
                cols.append(feat_idx)
                data.append(val)
        return sp.csr_matrix((np.array(data, dtype=np.float32), (rows, cols)), shape=(len(texts), self.total_dim))

    def fit_transform(self, texts: List[str]) -> sp.csr_matrix:
        self.fit(texts)
        return self.transform(texts)


class SequenceEncoder:
    """
    Encodes text into a fixed-length LongTensor of token IDs.
    Padded with 0 (<pad>) and truncated to max_seq_len.
    Used for sequential models (BiLSTM, Transformer).
    """
    def __init__(self, tokenizer: Tokenizer, max_seq_len: int = 64):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def encode(self, texts: List[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(texts)
        tensor = np.zeros((len(texts), self.max_seq_len), dtype=np.int64)
        for i, enc in enumerate(encodings):
            ids = enc.ids[:self.max_seq_len]
            tensor[i, :len(ids)] = ids
        return tensor

    def transform(self, texts: List[str]) -> np.ndarray:
        return self.encode(texts)


def analyze_representations(tokenizer: Tokenizer, texts: List[str], labels: Optional[np.ndarray] = None):
    """
    Deep empirical analysis of Representation Spaces:
    Panel 1: Top Class-Discriminative Subwords & Bigrams (Weighted Log-Odds Ratio with Dirichlet Prior, Monroe et al. 2008).
    Panel 2: Singular Value Decay / Information Spectrum (Energy Concentration sigma_k / sigma_1)
    """
    if labels is None:
        dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")
        labels = np.array(dataset["train"]["label"][:len(texts)])

    # 1. Feature Representation Matrices (sample 4000 for fast SVD)
    sample_n = min(4000, len(texts))
    sample_texts = texts[:sample_n]
    sample_labels = labels[:sample_n]

    ind_enc = TokenIndicator(tokenizer)
    bow_enc = BagOfWords(tokenizer)
    tfidf_enc = SubwordTfidf(tokenizer).fit(sample_texts)
    ngram_tfidf_enc = SubwordNgramTfidf(tokenizer, max_features=16384, sublinear_tf=True).fit(sample_texts)

    X_ind = ind_enc.transform(sample_texts)
    X_bow = bow_enc.transform(sample_texts)
    X_tfidf = tfidf_enc.transform(sample_texts)
    X_ngram = ngram_tfidf_enc.transform(sample_texts)

    # 2. Compute Monroe et al. (2008) Weighted Log-Odds Ratio for Positive (class 2) vs Negative (class 0)
    pos_idx = np.where(sample_labels == 2)[0]
    neg_idx = np.where(sample_labels == 0)[0]

    pos_enc = tokenizer.encode_batch([sample_texts[i] for i in pos_idx])
    neg_enc = tokenizer.encode_batch([sample_texts[i] for i in neg_idx])

    pos_counts = Counter()
    for enc in pos_enc:
        for t in enc.ids:
            if t >= 6 and t < tokenizer.get_vocab_size():
                pos_counts[t] += 1

    neg_counts = Counter()
    for enc in neg_enc:
        for t in enc.ids:
            if t >= 6 and t < tokenizer.get_vocab_size():
                neg_counts[t] += 1

    n_pos = sum(pos_counts.values())
    n_neg = sum(neg_counts.values())
    prior = 0.1
    V = tokenizer.get_vocab_size()
    vocab_set = set(list(pos_counts.keys()) + list(neg_counts.keys()))

    z_scores = {}
    for t in vocab_set:
        y_pos = pos_counts[t]
        y_neg = neg_counts[t]
        if y_pos + y_neg < 8:
            continue
        tok_str = tokenizer.id_to_token(t).replace("Ġ", " ").replace("Ċ", "").strip()
        if not tok_str or len(tok_str) < 2 or tok_str in ["user", "http"]:
            continue
        odds_pos = (y_pos + prior) / (n_pos - y_pos + prior * V)
        odds_neg = (y_neg + prior) / (n_neg - y_neg + prior * V)
        delta = np.log(odds_pos) - np.log(odds_neg)
        var = 1.0 / (y_pos + prior) + 1.0 / (y_neg + prior)
        z_scores[tok_str] = delta / np.sqrt(var)

    sorted_z = sorted(z_scores.items(), key=lambda x: x[1])
    top_neg = sorted_z[:8]  # lowest z -> highly negative
    top_pos = sorted_z[-8:] # highest z -> highly positive

    # 3. Compute SVD spectrum (sigma_k / sigma_1)
    k_svd = 40
    _, s_ind, _ = sp.linalg.svds(X_ind, k=k_svd)
    _, s_bow, _ = sp.linalg.svds(X_bow, k=k_svd)
    _, s_tfidf, _ = sp.linalg.svds(X_tfidf, k=k_svd)
    _, s_ngram, _ = sp.linalg.svds(X_ngram, k=k_svd)

    s_ind = np.sort(s_ind)[::-1] / s_ind.max()
    s_bow = np.sort(s_bow)[::-1] / s_bow.max()
    s_tfidf = np.sort(s_tfidf)[::-1] / s_tfidf.max()
    s_ngram = np.sort(s_ngram)[::-1] / s_ngram.max()

    # 4. Create the 2-Panel Figure for Slide 3
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # Panel 1: Top Class-Discriminative Subwords (Log-Odds Ratio)
    terms = [item[0] for item in top_neg] + [item[0] for item in top_pos]
    z_vals = [item[1] for item in top_neg] + [item[1] for item in top_pos]
    colors = ["#b91c1c"] * len(top_neg) + ["#0f766e"] * len(top_pos)

    y_pos_arr = np.arange(len(terms))
    axes[0].barh(y_pos_arr, z_vals, color=colors, edgecolor="black", alpha=0.88, height=0.7)
    axes[0].set_yticks(y_pos_arr)
    axes[0].set_yticklabels(terms, fontsize=7.5, fontweight="bold")
    axes[0].axvline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel(r"Monroe et al. Log-Odds $Z$-Score", fontsize=8, fontweight="bold")
    axes[0].set_title(r"Top Polarity Markers (Neg $\leftarrow \rightarrow$ Pos)", fontsize=8.5, fontweight="bold")
    axes[0].tick_params(axis="x", labelsize=7.5)

    # Panel 2: Singular Value Information Spectrum
    ranks = np.arange(1, k_svd + 1)
    axes[1].plot(ranks, s_ind, label="Token Indicator", color="#64748b", linestyle=":", linewidth=1.8)
    axes[1].plot(ranks, s_bow, label="Bag-of-Words", color="#1e293b", linestyle="--", linewidth=1.8)
    axes[1].plot(ranks, s_tfidf, label="Subword TF-IDF", color="#b91c1c", linewidth=2.0)
    axes[1].plot(ranks, s_ngram, label="N-gram TF-IDF (1,2-gr)", color="#0f766e", linewidth=2.0)
    axes[1].set_xlabel("Latent Semantic Dimension $k$", fontsize=8, fontweight="bold")
    axes[1].set_ylabel(r"Relative Singular Value $\sigma_k / \sigma_1$", fontsize=8, fontweight="bold")
    axes[1].set_title("Information Spectrum & Energy Decay", fontsize=8.5, fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=7.0, frameon=True)
    axes[1].tick_params(axis="both", labelsize=7.5)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "representations_comparison.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "representations_comparison.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved upgraded representations_comparison.pdf / .png (2-Panel Figure)")

    return {
        "top_neg": top_neg,
        "top_pos": top_pos,
        "singular_values": {
            "tfidf_top5": s_tfidf[:5].tolist(),
            "ngram_top5": s_ngram[:5].tolist()
        }
    }


if __name__ == "__main__":
    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    tok = Tokenizer.from_file("Assignment1/results/tokenizers/byte_bpe_8192.json")
    analyze_representations(tok, dataset["train"]["text"])
