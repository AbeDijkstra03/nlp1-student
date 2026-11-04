"""
Master Reproduction Script for NLP1 Assignment 1 / Project 1.
Runs the entire end-to-end pipeline:
1. Exploratory Data Analysis (EDA)
2. BPE Tokenizer training & vocabulary evaluation
3. Feature Representation Spaces analysis
4. 5-Fold Stratified Cross-Validation (All Architectures & Representations)
5. Bayesian ROPE estimation with baycomp
6. Final test set evaluation & qualitative error analysis
7. Automatic LaTeX presentation compilation & PNG slide rendering
"""

import sys
from pathlib import Path
import time
import subprocess
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Assignment1.src.eda import run_full_eda
from Assignment1.src.tokenizer import (
    run_tokenizer_pipeline,
    train_tweet_tokenizer,
    compare_vocab_sizes
)
from Assignment1.src.features import analyze_representations
from Assignment1.src.train import run_cross_validation
from Assignment1.src.bayesian_eval import run_full_bayesian_analysis
from Assignment1.src.error_analysis import evaluate_final_models

from datasets import load_dataset
from tokenizers import Tokenizer
import numpy as np


def main():
    print("=" * 80)
    print("NLP1 PROJECT 1: END-TO-END EXPERIMENTAL PIPELINE")
    print("=" * 80)
    total_start = time.time()

    # Step 1: EDA
    print("\n>>> STEP 1: Deep Exploratory Data Analysis (Polars)...")
    run_full_eda()

    # Step 2: Tokenizers
    print("\n>>> STEP 2: BPE Tokenizer Training & Vocab Exploration...")
    run_tokenizer_pipeline()

    tok_path = Path("Assignment1/results/tokenizers/byte_bpe_8192.json")
    if not tok_path.exists():
        tok_path = Path("Assignment1/results/tokenizers/bpe_vocab_8192.json")
    tok = Tokenizer.from_file(str(tok_path))

    # Step 3: Representations
    print("\n>>> STEP 3: Representation Spaces (Indicator, BoW, TF-IDF, N-Gram)...")
    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    train_corpus = dataset["train"]["text"]
    train_labels = np.array(dataset["train"]["label"])
    analyze_representations(tok, train_corpus, labels=train_labels)

    # Step 4: Cross-Validation
    print("\n>>> STEP 4: 5-Fold Stratified Cross-Validation (All Configurations)...")
    cv_summary = run_cross_validation(train_corpus, train_labels, tok, n_splits=5)

    # Step 5: Bayesian Analysis with ROPE
    print("\n>>> STEP 5: Bayesian Estimation with ROPE (baycomp)...")
    bayesian_results = run_full_bayesian_analysis(rope=0.01)

    # Step 6: Final Test Set Evaluation & Error Analysis
    print("\n>>> STEP 6: Final Test Evaluation & Error Analysis...")
    final_test_results = evaluate_final_models()

    # Step 7: Automatic LaTeX Presentation Compilation & PNG Slide Rendering
    print("\n>>> STEP 7: Compiling Presentation Slides (LaTeX -> PDF -> PNGs)...")
    slides_dir = Path("Assignment1/slides")
    tex_file = slides_dir / "presentation.tex"
    if tex_file.exists() and shutil.which("pdflatex"):
        # Run pdflatex twice to resolve references and labels
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "presentation.tex"], cwd=str(slides_dir), check=True)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "presentation.tex"], cwd=str(slides_dir), check=True)
        print("  Compiled presentation.pdf successfully!")

        # Render high-resolution PNG slides (300 DPI)
        if shutil.which("pdftoppm"):
            # Render slide_page
            subprocess.run(["pdftoppm", "-png", "-r", "300", "presentation.pdf", "slide_page"], cwd=str(slides_dir), check=True)
            # Render slide_preview
            subprocess.run(["pdftoppm", "-png", "-r", "300", "presentation.pdf", "slide_preview"], cwd=str(slides_dir), check=True)
            print("  Rendered slide_page-*.png and slide_preview-*.png successfully!")

    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.1f}s!")
    print("All figures saved in: Assignment1/slides/figures/")
    print("All numeric results saved in: Assignment1/results/")
    print("All presentation slides compiled in: Assignment1/slides/")
    print("=" * 80)


if __name__ == "__main__":
    main()
