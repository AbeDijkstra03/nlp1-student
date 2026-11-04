"""
Bayesian Classifier Comparison with Region of Practical Equivalence (ROPE).
Implements Correlated Bayesian t-test (Benavoli et al., 2017) to compare:
- Architectures: Deep MLP vs Shallow MLP vs Log-Linear
- Representations: TF-IDF vs Bag-of-Words vs Token Indicator
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import baycomp

FIGURES_DIR = Path("Assignment1/slides/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path("Assignment1/results")


class BayesianRopeComparator:
    """
    Bayesian Correlated t-test wrapper for cross-validation evaluation.
    Accounts for the statistical dependence between CV folds.
    """
    def __init__(self, sys_a_scores: List[float], sys_b_scores: List[float], name_a: str = "System A", name_b: str = "System B"):
        self.name_a = name_a
        self.name_b = name_b
        self.sys_a = np.array(sys_a_scores)
        self.sys_b = np.array(sys_b_scores)
        # 1 run of 5-fold CV: test/train ratio = 1/(5-1) = 0.25
        self._ct = baycomp.CorrelatedTTest(self.sys_a, self.sys_b, rope=0.0, runs=1)
        self.df = self._ct.df
        self.mean_diff = self._ct.mean  # (B - A)
        self.std = np.sqrt(self._ct.var)
        self.dist = st.t(self.df, loc=self.mean_diff, scale=self.std)

    def compute_probs(self, rope: float = 0.01) -> Dict[str, float]:
        """
        Compute posterior probabilities:
        P(A > B) = P(diff < -rope)
        P(ROPE) = P(-rope <= diff <= rope)
        P(B > A) = P(diff > rope)
        """
        p_a = float(self.dist.cdf(-rope))
        p_rope = float(self.dist.cdf(rope) - self.dist.cdf(-rope))
        p_b = float(1.0 - self.dist.cdf(rope))
        effect_size = float(self.mean_diff / max(self.std, 1e-8))
        hdi_95 = [float(x) for x in self.dist.interval(0.95)]

        return {
            "name_a": self.name_a,
            "name_b": self.name_b,
            "mean_diff": float(self.mean_diff),
            "std": float(self.std),
            "effect_size": effect_size,
            "hdi_95": hdi_95,
            "rope": rope,
            "p_a_wins": p_a,
            "p_rope": p_rope,
            "p_b_wins": p_b
        }

    def plot_posterior(self, rope: float = 0.01, filename: str = "bayesian_rope.pdf"):
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        
        ci = self.dist.interval(0.999)
        x_min = min(ci[0], -rope * 3.2)
        x_max = max(ci[1], rope * 3.2)
        x = np.linspace(x_min, x_max, 1000)
        y = self.dist.pdf(x)

        ax.plot(x, y, color="#2c3e50", lw=2.2, label=r"Posterior $p(\mu | \mathcal{D})$")
        ax.axvspan(-rope, rope, color="#16a085", alpha=0.25, label=f"ROPE [$-{rope}, +{rope}$]")

        probs = self.compute_probs(rope=rope)
        y_max = y.max()

        # Annotate probabilities cleanly inside plot bounds
        ax.text(0.5 * (x_min - rope), y_max * 0.42, f"P({self.name_a} wins)\n{probs['p_a_wins']*100:.1f}%",
                ha="center", va="center", color="#c0392b", fontweight="bold", fontsize=8)
        ax.text(0, y_max * 0.68, f"Equivalent (ROPE)\n{probs['p_rope']*100:.1f}%",
                ha="center", va="center", color="#16a085", fontweight="bold", fontsize=8)
        ax.text(rope + 0.002, y_max * 0.42, f"P({self.name_b} wins)\n{probs['p_b_wins']*100:.1f}%",
                ha="left", va="center", color="#2980b9", fontweight="bold", fontsize=8)

        ax.axvline(0, color="gray", linestyle=":", alpha=0.7)
        ax.axvline(self.mean_diff, color="#e67e22", linestyle="--", label=f"Mean Diff = {self.mean_diff:+.3f}")

        ax.set_xlabel(f"Mean Difference ({self.name_b} - {self.name_a})", fontweight="bold")
        ax.set_ylabel("Posterior Density", fontweight="bold")
        ax.set_title(f"Bayesian ROPE Analysis: {self.name_b} vs. {self.name_a}", fontweight="bold")
        ax.legend(loc="upper right", frameon=True, fontsize=8.5)
        fig.tight_layout()

        out_path = FIGURES_DIR / filename
        fig.savefig(out_path, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved Bayesian plot to {out_path}")


def run_full_bayesian_analysis(cv_summary_path: str = "Assignment1/results/cv_results_summary.json", rope: float = 0.01):
    with open(cv_summary_path, "r") as f:
        summary = json.load(f)

    comparisons = [
        # Sequence Models vs Best BoW Baselines
        ("shallow_mlp_bow", "bilstm_attention", "Shallow MLP (BoW)", "BiLSTM + Attention", "bayesian_rope_bilstm_vs_shallow.pdf"),
        ("loglin_bow", "bilstm_attention", "Log-Linear (BoW)", "BiLSTM + Attention", "bayesian_rope_bilstm_vs_loglin.pdf"),
        ("deep_mlp_bow", "bilstm_attention", "Deep MLP (BoW)", "BiLSTM + Attention", "bayesian_rope_bilstm_vs_deep.pdf"),
        ("transformer_scratch", "bilstm_attention", "Transformer Scratch", "BiLSTM + Attention", "bayesian_rope_bilstm_vs_transformer.pdf"),
        ("shallow_mlp_bow", "transformer_scratch", "Shallow MLP (BoW)", "Transformer Scratch", "bayesian_rope_transformer_vs_shallow.pdf"),
        # Classical Architecture comparisons on BoW
        ("shallow_mlp_bow", "deep_mlp_bow", "Shallow MLP (BoW)", "Deep MLP (BoW)", "bayesian_rope_deep_vs_shallow.pdf"),
        ("loglin_bow", "shallow_mlp_bow", "Log-Linear (BoW)", "Shallow MLP (BoW)", "bayesian_rope_shallow_vs_loglin.pdf"),
        # Representation comparisons
        ("loglin_tfidf", "loglin_bow", "TF-IDF (LogLin)", "BoW (LogLin)", "bayesian_rope_bow_vs_tfidf.pdf"),
        ("loglin_indicator", "loglin_bow", "Token Indicator (LogLin)", "BoW (LogLin)", "bayesian_rope_bow_vs_indicator.pdf"),
    ]

    all_comparison_results = []
    print("\n" + "="*80)
    print(f"BAYESIAN ESTIMATION WITH ROPE ({rope=:.3f})")
    print("="*80)

    for sys_a_key, sys_b_key, name_a, name_b, fig_name in comparisons:
        sys_a_scores = summary[sys_a_key]["fold_macro_f1"]
        sys_b_scores = summary[sys_b_key]["fold_macro_f1"]

        comp = BayesianRopeComparator(sys_a_scores, sys_b_scores, name_a=name_a, name_b=name_b)
        probs = comp.compute_probs(rope=rope)
        comp.plot_posterior(rope=rope, filename=fig_name)
        all_comparison_results.append(probs)

        print(f"\nComparison: {name_b} vs. {name_a}")
        print(f"  Mean Difference: {probs['mean_diff']:+.4f} (Std: {probs['std']:.4f}, Effect Size: {probs['effect_size']:.2f})")
        print(f"  95% HDI: [{probs['hdi_95'][0]:.4f}, {probs['hdi_95'][1]:.4f}]")
        print(f"  P({name_a} > {name_b}): {probs['p_a_wins']*100:.2f}% | P(ROPE): {probs['p_rope']*100:.2f}% | P({name_b} > {name_a}): {probs['p_b_wins']*100:.2f}%")

    with open(RESULTS_DIR / "bayesian_rope_summary.json", "w") as f:
        json.dump(all_comparison_results, f, indent=2)

    return all_comparison_results


if __name__ == "__main__":
    run_full_bayesian_analysis()

