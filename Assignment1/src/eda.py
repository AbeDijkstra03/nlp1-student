"""
Exploratory Data Analysis (EDA) for TweetEval Sentiment Analysis
Kaggle Grandmaster-style deep empirical analysis using Polars, Matplotlib, and Seaborn.
"""

from pathlib import Path
import re
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# Set styling for publication / Beamer slides
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.dpi'] = 300

FIGURES_DIR = Path("Assignment1/slides/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path("Assignment1/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL_COLORS = {"Negative": "#d9534f", "Neutral": "#f0ad4e", "Positive": "#5cb85c"}
COLOR_LIST = ["#d9534f", "#f0ad4e", "#5cb85c"]


def load_all_splits():
    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    dfs = {}
    for split in ["train", "validation", "test"]:
        df = pl.from_pandas(dataset[split].to_pandas())
        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        df = df.with_columns([
            pl.col("label").replace_strict(label_map, return_dtype=pl.String).alias("label_name"),
            pl.col("text").str.len_chars().alias("char_len"),
            pl.col("text").str.split(" ").list.len().alias("word_len"),
            pl.col("text").str.count_matches(r"@user").alias("user_mentions"),
            pl.col("text").str.count_matches(r"http\S+").alias("urls"),
            pl.col("text").str.count_matches(r"#\S+").alias("hashtags"),
            pl.col("text").str.count_matches(r"!+").alias("exclamations"),
            pl.col("text").str.count_matches(r"\?+").alias("questions"),
            pl.col("text").str.count_matches(r"(?i)\b(not|never|no|n't|cant|cannot|wont)\b").alias("negations"),
            pl.col("text").str.count_matches(r"\b[A-Z]{2,}\b").alias("uppercase_words"),
        ])
        dfs[split] = df
    return dfs


def analyze_distributions(dfs):
    print("=" * 60)
    print("1. SPLIT SIZES & CLASS DISTRIBUTIONS")
    print("=" * 60)
    summary_rows = []
    for split, df in dfs.items():
        total = len(df)
        counts = df["label_name"].value_counts()
        row = {"Split": split, "Total": total}
        for lbl in ["Negative", "Neutral", "Positive"]:
            c = counts.filter(pl.col("label_name") == lbl)["count"]
            cnt = c[0] if len(c) > 0 else 0
            pct = 100.0 * cnt / total
            row[f"{lbl} (%)"] = f"{cnt} ({pct:.1f}%)"
            row[f"{lbl}_pct"] = pct
        summary_rows.append(row)
    
    summary_df = pl.DataFrame(summary_rows)
    print(summary_df.select(["Split", "Total", "Negative (%)", "Neutral (%)", "Positive (%)"]))
    
    # Plot Distribution Shift
    fig, ax = plt.subplots(figsize=(7, 3.8))
    splits = ["train", "validation", "test"]
    x = np.arange(len(splits))
    width = 0.25
    
    for i, lbl in enumerate(["Negative", "Neutral", "Positive"]):
        vals = [summary_df.filter(pl.col("Split") == s)[f"{lbl}_pct"][0] for s in splits]
        ax.bar(x + (i - 1) * width, vals, width, label=lbl, color=COLOR_LIST[i], edgecolor='black', alpha=0.88)
        for idx, v in enumerate(vals):
            ax.text(x[idx] + (i - 1) * width, v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(["Train (N=45.6k)", "Validation (N=2.0k)", "Test (N=12.3k)"])
    ax.set_ylabel("Class Proportion (%)")
    ax.set_title("TweetEval Class Distribution Shift: Train/Val vs. Test", fontweight='bold')
    ax.set_ylim(0, 60)
    ax.legend(title="Sentiment", loc="upper left", frameon=True)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "label_distribution_shift.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "label_distribution_shift.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved label_distribution_shift.pdf / .png")


def analyze_tweet_features(train_df):
    print("=" * 60)
    print("2. TWEET LENGTH & STYLOMETRIC FEATURES (TRAIN)")
    print("=" * 60)
    stats = train_df.group_by("label_name").agg([
        pl.len().alias("count"),
        pl.col("word_len").mean().alias("mean_words"),
        pl.col("word_len").median().alias("median_words"),
        pl.col("word_len").max().alias("max_words"),
        pl.col("exclamations").mean().alias("mean_exclamations"),
        pl.col("negations").mean().alias("mean_negations"),
        pl.col("uppercase_words").mean().alias("mean_caps"),
        pl.col("user_mentions").mean().alias("mean_mentions"),
    ]).sort("label_name")
    print(stats)
    
    # Save table to latex
    stats_pd = stats.to_pandas()
    stats_pd.to_csv(RESULTS_DIR / "eda_feature_stats.csv", index=False)
    
    # Visualizing feature distributions
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    
    # Word length boxplot
    sns.boxplot(data=train_df.to_pandas(), x="label_name", y="word_len", hue="label_name", legend=False, ax=axes[0], palette=LABEL_COLORS, order=["Negative", "Neutral", "Positive"])
    axes[0].set_title("Word Count Distribution", fontweight='bold')
    axes[0].set_xlabel("Sentiment")
    axes[0].set_ylabel("Number of Words")
    axes[0].set_ylim(0, 45)
    
    # Negations per tweet
    neg_data = train_df.group_by("label_name").agg(pl.col("negations").mean().alias("neg_rate")).to_pandas()
    sns.barplot(data=neg_data, x="label_name", y="neg_rate", hue="label_name", legend=False, ax=axes[1], palette=LABEL_COLORS, order=["Negative", "Neutral", "Positive"], edgecolor='black')
    axes[1].set_title("Average Negations / Tweet", fontweight='bold')
    axes[1].set_xlabel("Sentiment")
    axes[1].set_ylabel("Avg Negations ('not', 'never', etc.)")
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
    
    # Exclamations per tweet
    exc_data = train_df.group_by("label_name").agg(pl.col("exclamations").mean().alias("exc_rate")).to_pandas()
    sns.barplot(data=exc_data, x="label_name", y="exc_rate", hue="label_name", legend=False, ax=axes[2], palette=LABEL_COLORS, order=["Negative", "Neutral", "Positive"], edgecolor='black')
    axes[2].set_title("Average Exclamations / Tweet", fontweight='bold')
    axes[2].set_xlabel("Sentiment")
    axes[2].set_ylabel("Avg Exclamations ('!')")
    for p in axes[2].patches:
        axes[2].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "tweet_length_and_features.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "tweet_length_and_features.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved tweet_length_and_features.pdf / .png")


def analyze_vocabulary_and_heaps(train_df):
    print("=" * 60)
    print("3. VOCABULARY GROWTH & HEAP'S LAW")
    print("=" * 60)
    
    texts = train_df["text"].to_list()
    # Sample subsets to compute Heaps law
    token_counts = []
    vocab_sizes = []
    seen_words = set()
    total_tokens = 0
    
    for i, t in enumerate(texts):
        # standard whitespace tokenization
        words = re.findall(r"\b\w+\b", t.lower())
        total_tokens += len(words)
        seen_words.update(words)
        if (i + 1) % 1000 == 0 or i == len(texts) - 1:
            token_counts.append(total_tokens)
            vocab_sizes.append(len(seen_words))
            
    token_counts = np.array(token_counts)
    vocab_sizes = np.array(vocab_sizes)
    
    # Fit Heaps law: V = k * N^beta  => log(V) = log(k) + beta * log(N)
    log_n = np.log(token_counts)
    log_v = np.log(vocab_sizes)
    slope, intercept = np.polyfit(log_n, log_v, 1)
    k_fit = np.exp(intercept)
    beta_fit = slope
    print(f"Heap's Law Fit: V = {k_fit:.2f} * N^{beta_fit:.3f}")
    print(f"Total corpus tokens: {total_tokens:,}, Unique words (whitespace): {len(seen_words):,}")

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.scatter(token_counts / 1000, vocab_sizes / 1000, color="#1f77b4", s=18, label="Empirical Vocabulary Growth")
    fitted_v = k_fit * (token_counts ** beta_fit)
    ax.plot(token_counts / 1000, fitted_v / 1000, color="crimson", linestyle="--",
            label=f"Heap's Law: $V = {k_fit:.1f} \\cdot N^{{{beta_fit:.2f}}}$")
    ax.set_xlabel("Total Tokens ($N \\times 10^3$)")
    ax.set_ylabel("Vocabulary Size ($V \\times 10^3$)")
    ax.set_title("Heap's Law: Empirical Vocabulary Saturation in TweetEval", fontweight='bold')
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "heaps_law_vocab.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "heaps_law_vocab.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved heaps_law_vocab.pdf / .png")
    
    return {
        "total_tokens": total_tokens,
        "raw_vocab_size": len(seen_words),
        "heaps_k": k_fit,
        "heaps_beta": beta_fit
    }


def run_full_eda():
    dfs = load_all_splits()
    analyze_distributions(dfs)
    analyze_tweet_features(dfs["train"])
    heap_info = analyze_vocabulary_and_heaps(dfs["train"])
    print("\nEDA Completed successfully. All figures generated in:", FIGURES_DIR)


if __name__ == "__main__":
    run_full_eda()
