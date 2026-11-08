"""
generate_minimal_figures.py
Generates crystal-clear, publication-quality diagnostic plots (300 DPI PDF & PNG)
for the minimal pedagogical story (Topics 1 to 6) without any deep learning bloat.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

FIGURES_DIR = Path("Assignment1/slides/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Styling
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["figure.dpi"] = 300

C_SLATE = "#1e293b"
C_TEAL = "#0f766e"
C_CRIMSON = "#b91c1c"
C_GRAY = "#64748b"
C_LIGHT_BG = "#f8fafc"
C_ORANGE = "#d97706"
C_BLUE = "#2563eb"

# ==============================================================================
# 1. minimal_class_distribution
# ==============================================================================
def plot_class_distribution():
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    
    splits = ["Train (N=45.6k)", "Validation (N=2.0k)", "Test (N=12.3k)"]
    neg_pct = [15.5, 15.6, 32.3]
    neu_pct = [45.3, 43.5, 48.3]
    pos_pct = [39.1, 41.0, 19.3]
    
    x = np.arange(len(splits))
    width = 0.26
    
    rects1 = ax.bar(x - width, neg_pct, width, label="Negative", color=C_CRIMSON, edgecolor="black", alpha=0.88)
    rects2 = ax.bar(x, neu_pct, width, label="Neutral", color=C_GRAY, edgecolor="black", alpha=0.88)
    rects3 = ax.bar(x + width, pos_pct, width, label="Positive", color=C_TEAL, edgecolor="black", alpha=0.88)
    
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")
            
    # Add shift callout
    ax.annotate("Neg doubles (15.5% -> 32.3%)\nPos halves (39.1% -> 19.3%)",
                xy=(x[2] - width, 32.3), xytext=(x[1] - 0.1, 52),
                arrowprops=dict(arrowstyle="->", color=C_CRIMSON, lw=1.5),
                fontsize=8.5, fontweight="bold", color=C_CRIMSON,
                bbox=dict(boxstyle="round,pad=0.3", fc="#fef2f2", ec=C_CRIMSON, lw=1))
    
    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontweight="bold", fontsize=9.5)
    ax.set_ylabel("Class Proportion (%)", fontweight="bold")
    ax.set_ylim(0, 62)
    ax.set_title("TweetEval Sentiment: Prior Shift from Train/Val to Test Set", fontweight="bold", fontsize=11)
    ax.legend(title="Sentiment Class", loc="upper left", frameon=True, fontsize=8.5, title_fontsize=9)
    fig.tight_layout()
    
    fig.savefig(FIGURES_DIR / "minimal_class_distribution.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "minimal_class_distribution.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved minimal_class_distribution.pdf / .png")


# ==============================================================================
# 2. minimal_vocab_sweep
# ==============================================================================
def plot_vocab_sweep():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.2))
    
    # Left: Tokens per tweet & Expansion ratio
    vocabs = [2048, 4096, 8192, 16384]
    tokens = [37.28, 32.68, 29.64, 27.55]
    ratios = [1.92, 1.68, 1.52, 1.42]
    
    color1 = C_SLATE
    color2 = C_TEAL
    
    ax1.plot(vocabs, tokens, marker="o", color=color1, lw=2, label="Tokens / Tweet")
    ax1.set_xlabel("Vocabulary Size ($V$)", fontweight="bold")
    ax1.set_ylabel("Subwords per Tweet", color=color1, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(vocabs)
    ax1.set_xticklabels(["2k", "4k", "8k", "16k"])
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(vocabs, ratios, marker="s", color=color2, lw=2, linestyle="--", label="Expansion Ratio")
    ax1_twin.set_ylabel("Subwords / Word Ratio", color=color2, fontweight="bold")
    ax1_twin.tick_params(axis="y", labelcolor=color2)
    ax1_twin.set_ylim(1.2, 2.1)
    
    # Mark 8k
    ax1.axvline(8192, color=C_ORANGE, linestyle=":", lw=1.8)
    ax1.annotate("Optimal V=8,192\n(1.52 subw/wd, 0% UNK)",
                 xy=(8192, 29.64), xytext=(3500, 34),
                 arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=1.5),
                 fontsize=8, fontweight="bold", color=C_ORANGE,
                 bbox=dict(boxstyle="round,pad=0.2", fc="#fffbeb", ec=C_ORANGE))
    ax1.set_title("BPE Subword Granularity Trade-Off", fontweight="bold", fontsize=10)
    
    # Right: Heaps Law & Vocabulary Expansion
    N = np.linspace(1000, 63650, 200)
    # Heaps Law: V(N) = 17.14 * N^0.572
    V_heaps = 17.14 * (N ** 0.572)
    
    ax2.plot(N / 1000, V_heaps, color=C_CRIMSON, lw=2, label=r"Raw Words: $V(N) = 17.14 N^{0.572}$")
    ax2.axhline(8192, color=C_TEAL, lw=2, linestyle="--", label="Byte-BPE Cap ($V=8,192$)")
    ax2.fill_between(N / 1000, 8192, V_heaps, where=(V_heaps >= 8192), color=C_CRIMSON, alpha=0.12, label="Unbounded OOV / Sparsity")
    
    ax2.set_xlabel("Corpus Size ($N$ in thousands of tokens)", fontweight="bold")
    ax2.set_ylabel("Vocabulary Size ($V$)", fontweight="bold")
    ax2.set_title("Heap's Law vs. Fixed Byte-BPE", fontweight="bold", fontsize=10)
    ax2.legend(loc="upper left", fontsize=7.5, frameon=True)
    ax2.annotate("43,395 raw types\nat full corpus!",
                 xy=(63.65, 43.395), xytext=(40, 36),
                 arrowprops=dict(arrowstyle="->", color=C_CRIMSON, lw=1.2),
                 fontsize=7.5, fontweight="bold", color=C_CRIMSON)
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "minimal_vocab_sweep.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "minimal_vocab_sweep.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved minimal_vocab_sweep.pdf / .png")


# ==============================================================================
# 3. minimal_representations_polarity
# ==============================================================================
def plot_representations_polarity():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.2))
    
    # Left: 5-Fold CV Macro-F1 across the 3 representations
    reps = ["Token\nIndicator", "Bag-of-Words\n(BoW)", "Subword\nTF-IDF"]
    f1_loglin = [0.6259, 0.6263, 0.6058]
    err_loglin = [0.0038, 0.0055, 0.0028]
    
    f1_shallow = [0.6258, 0.6270, 0.6091]
    err_shallow = [0.0045, 0.0054, 0.0035]
    
    x = np.arange(len(reps))
    width = 0.35
    
    bar1 = ax1.bar(x - width/2, f1_loglin, width, yerr=err_loglin, capsize=4,
                   label="Log-Linear", color=C_SLATE, edgecolor="black", alpha=0.88)
    bar2 = ax1.bar(x + width/2, f1_shallow, width, yerr=err_shallow, capsize=4,
                   label="Shallow MLP", color=C_TEAL, edgecolor="black", alpha=0.88)
    
    ax1.set_ylabel("5-Fold CV Macro-F1", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(reps, fontweight="bold", fontsize=8.5)
    ax1.set_ylim(0.57, 0.65)
    ax1.set_title("5-Fold CV Across Representations", fontweight="bold", fontsize=10)
    ax1.legend(loc="lower right", fontsize=8, frameon=True)
    
    # Annotation for BoW vs TF-IDF & Indicator
    ax1.text(0.5, 0.635, "Indicator ~ BoW (98.8% in ROPE)\nTF-IDF drops (BoW wins P=99.4%)",
             ha="center", fontsize=7.5, fontweight="bold", color=C_CRIMSON,
             bbox=dict(boxstyle="round,pad=0.2", fc="#fef2f2", ec=C_CRIMSON, lw=0.8))
    
    # Right: Top Polarity Subwords from Log-Linear weights
    neg_words = ["worst", "disappointed", "stupid", "sucks", "terrible", "hate", "horrible", "not"]
    neg_weights = [-4.06, -3.68, -3.53, -3.49, -2.93, -2.75, -2.62, -2.48]
    
    pos_words = ["amazing", "Great", "Happy", "Excited", "awesome", "glad", "beautiful", "love"]
    pos_weights = [3.69, 3.58, 3.26, 3.25, 3.15, 2.91, 2.85, 2.70]
    
    all_words = neg_words[::-1] + pos_words
    all_weights = neg_weights[::-1] + pos_weights
    colors = [C_CRIMSON] * len(neg_words) + [C_TEAL] * len(pos_words)
    
    y = np.arange(len(all_words))
    ax2.barh(y, all_weights, color=colors, edgecolor="black", alpha=0.88, height=0.7)
    ax2.set_yticks(y)
    ax2.set_yticklabels(all_words, fontsize=7.5, fontweight="bold")
    ax2.axvline(0, color="gray", linestyle="--", lw=0.8)
    ax2.set_xlabel(r"Log-Linear Weight $W_{c, i}$", fontweight="bold", fontsize=9)
    ax2.set_title(r"Top Polarity Markers (Neg $\leftarrow \rightarrow$ Pos)", fontweight="bold", fontsize=10)
    ax2.tick_params(axis="x", labelsize=8)
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "minimal_representations_polarity.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "minimal_representations_polarity.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved minimal_representations_polarity.pdf / .png")


# ==============================================================================
# 4. minimal_architectures_ablation
# ==============================================================================
def plot_architectures_ablation():
    fig, ax1 = plt.subplots(figsize=(6.8, 3.4))
    
    models = ["Log-Linear\nSoftmax", "Shallow MLP\n(1 Hidden + Drop)", "Deep Residual\nMLP (4 ResBlocks)"]
    f1_means = [0.6263, 0.6270, 0.6122]
    f1_stds = [0.0055, 0.0054, 0.0063]
    params = [24.6, 2100.0, 2230.0] # in thousands
    
    x = np.arange(len(models))
    width = 0.42
    
    bars = ax1.bar(x, f1_means, width, yerr=f1_stds, capsize=5,
                   color=[C_SLATE, C_TEAL, C_CRIMSON], edgecolor="black", alpha=0.88)
    
    for i, b in enumerate(bars):
        h = b.get_height()
        p_str = f"{params[i]:.0f}k params" if params[i] < 1000 else f"{params[i]/1000:.2f}M params"
        ax1.text(b.get_x() + b.get_width() / 2, 0.585,
                 f"{f1_means[i]:.4f}\n({p_str})",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="white")
        
    ax1.set_ylabel("5-Fold CV Macro-F1", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="bold", fontsize=9.5)
    ax1.set_ylim(0.57, 0.65)
    ax1.set_title("Architectural Comparison & Deep MLP Ablation on BoW", fontweight="bold", fontsize=11)
    
    # Annotations
    ax1.annotate("Practically Equivalent\n(99.1% in ROPE: [-0.01, +0.01])\nOccam's Razor: 85x smaller!",
                 xy=(0.5, 0.627), xytext=(0.1, 0.638),
                 arrowprops=dict(arrowstyle="<->", color=C_TEAL, lw=1.5),
                 fontsize=8, fontweight="bold", color=C_TEAL,
                 bbox=dict(boxstyle="round,pad=0.2", fc="#f0fdfa", ec=C_TEAL))
    
    ax1.annotate("Deep MLP Degrades!\nP(Shallow > Deep) = 93.8%\nOverfits on 99.7% sparse BoW",
                 xy=(2.0, 0.613), xytext=(1.45, 0.598),
                 arrowprops=dict(arrowstyle="->", color=C_CRIMSON, lw=1.5),
                 fontsize=8, fontweight="bold", color=C_CRIMSON,
                 bbox=dict(boxstyle="round,pad=0.2", fc="#fef2f2", ec=C_CRIMSON))
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "minimal_architectures_ablation.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "minimal_architectures_ablation.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved minimal_architectures_ablation.pdf / .png")


# ==============================================================================
# 5. minimal_bayesian_rope
# ==============================================================================
def plot_bayesian_rope():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.2))
    rope = 0.01
    
    # Panel A: Shallow MLP vs Log-Linear
    # mean_diff: 0.0007, std: 0.00206
    df = 4
    mu1 = 0.0007
    sigma1 = 0.00206
    dist1 = stats.t(df=df, loc=mu1, scale=sigma1)
    
    x1 = np.linspace(-0.02, 0.02, 1000)
    y1 = dist1.pdf(x1)
    
    ax1.plot(x1, y1, color=C_SLATE, lw=2.2, label=r"Posterior $p(\mu | \mathcal{D})$")
    ax1.axvspan(-rope, rope, color=C_TEAL, alpha=0.22, label=f"ROPE [$-0.01, +0.01$]")
    ax1.axvline(0, color="gray", linestyle=":", lw=1.0)
    ax1.axvline(mu1, color=C_ORANGE, linestyle="--", lw=1.5, label=f"Mean diff: +{mu1:.4f}")
    
    y1_max = y1.max()
    ax1.text(0, y1_max * 0.5, "P(ROPE) = 99.1%\nPractically Equivalent\n(Occam's Razor: LogLin)",
             ha="center", va="center", color=C_TEAL, fontweight="bold", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.2", fc="#f0fdfa", ec=C_TEAL, lw=0.8))
    
    ax1.set_xlabel("Mean Diff (Shallow MLP - Log-Linear)", fontweight="bold", fontsize=8.5)
    ax1.set_ylabel("Posterior Density", fontweight="bold", fontsize=8.5)
    ax1.set_title("A: Shallow MLP vs. Log-Linear (BoW)", fontweight="bold", fontsize=9.5)
    ax1.legend(loc="upper right", fontsize=7.2, frameon=True)
    ax1.set_xlim(-0.018, 0.018)
    
    # Panel B: BoW vs TF-IDF (LogLin)
    # mean_diff: 0.0205, std: 0.00238
    mu2 = 0.0205
    sigma2 = 0.00238
    dist2 = stats.t(df=df, loc=mu2, scale=sigma2)
    
    x2 = np.linspace(-0.01, 0.035, 1000)
    y2 = dist2.pdf(x2)
    
    ax2.plot(x2, y2, color=C_SLATE, lw=2.2, label=r"Posterior $p(\mu | \mathcal{D})$")
    ax2.axvspan(-rope, rope, color=C_TEAL, alpha=0.22, label=f"ROPE [$-0.01, +0.01$]")
    ax2.axvline(0, color="gray", linestyle=":", lw=1.0)
    ax2.axvline(mu2, color=C_ORANGE, linestyle="--", lw=1.5, label=f"Mean diff: +{mu2:.4f}")
    
    y2_max = y2.max()
    ax2.text(0.022, y2_max * 0.55, "P(BoW > TF-IDF) = 99.4%\nBoW Decisively Wins!\n(L2 norm hurts)",
             ha="center", va="center", color=C_SLATE, fontweight="bold", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.2", fc="#fffbeb", ec=C_ORANGE, lw=0.8))
    
    ax2.set_xlabel("Mean Diff (BoW - TF-IDF)", fontweight="bold", fontsize=8.5)
    ax2.set_ylabel("Posterior Density", fontweight="bold", fontsize=8.5)
    ax2.set_title("B: Bag-of-Words vs. Subword TF-IDF", fontweight="bold", fontsize=9.5)
    ax2.legend(loc="upper left", fontsize=7.2, frameon=True)
    ax2.set_xlim(-0.008, 0.033)
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "minimal_bayesian_rope.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "minimal_bayesian_rope.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved minimal_bayesian_rope.pdf / .png")


# ==============================================================================
# 6. minimal_confusion_matrices
# ==============================================================================
def plot_confusion_matrices():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.4))
    
    classes = ["Negative", "Neutral", "Positive"]
    
    # Exact normalized confusion matrices from final_test_evaluation.json
    cm_loglin = np.array([
        [54.0, 36.0, 10.0],
        [21.3, 61.1, 17.6],
        [ 8.6, 29.8, 61.6]
    ])
    
    cm_shallow = np.array([
        [56.1, 34.5,  9.4],
        [23.5, 61.4, 15.1],
        [ 9.1, 30.5, 60.4]
    ])
    
    # Heatmap 1: Log-Linear
    sns.heatmap(cm_loglin, annot=True, fmt=".1f", cmap="Blues", ax=ax1, cbar=False,
                annot_kws={"size": 10, "weight": "bold"}, vmin=0, vmax=100)
    ax1.set_xticklabels(classes, fontsize=8.5, fontweight="bold")
    ax1.set_yticklabels(classes, fontsize=8.5, fontweight="bold", rotation=0)
    ax1.set_xlabel("Predicted Label", fontweight="bold", fontsize=9)
    ax1.set_ylabel("True Label", fontweight="bold", fontsize=9)
    ax1.set_title("Log-Linear (BoW)\nTest F1: 0.5799 | Acc: 58.9%", fontweight="bold", fontsize=10)
    
    # Heatmap 2: Shallow MLP
    sns.heatmap(cm_shallow, annot=True, fmt=".1f", cmap="Blues", ax=ax2, cbar=False,
                annot_kws={"size": 10, "weight": "bold"}, vmin=0, vmax=100)
    ax2.set_xticklabels(classes, fontsize=8.5, fontweight="bold")
    ax2.set_yticklabels(classes, fontsize=8.5, fontweight="bold", rotation=0)
    ax2.set_xlabel("Predicted Label", fontweight="bold", fontsize=9)
    ax2.set_ylabel("True Label", fontweight="bold", fontsize=9)
    ax2.set_title("Shallow MLP (BoW)\nTest F1: 0.5862 | Acc: 59.5%", fontweight="bold", fontsize=10)
    
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "minimal_confusion_matrices.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "minimal_confusion_matrices.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved minimal_confusion_matrices.pdf / .png")


if __name__ == "__main__":
    plot_class_distribution()
    plot_vocab_sweep()
    plot_representations_polarity()
    plot_architectures_ablation()
    plot_bayesian_rope()
    plot_confusion_matrices()
    print("ALL 6 MINIMAL FIGURES SUCCESSFULLY GENERATED!")

