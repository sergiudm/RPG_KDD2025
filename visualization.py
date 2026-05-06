"""
Visualization script for RPG_KDD2025 experiment results.
This script parses log files and creates a comprehensive visualization of results across all datasets.
"""

import os
import re
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (16, 10)
plt.rcParams["font.size"] = 10


def parse_log_file(log_path):
    """
    Parse a log file to extract training information and results.

    Returns:
        dict: Dictionary containing parsed information
    """
    results = {
        "dataset": None,
        "hyperparams": {},
        "train_losses": [],
        "val_metrics": defaultdict(list),
        "best_epoch": None,
        "best_val_score": None,
        "test_results": {},
        "epochs": [],
    }

    # Extract dataset name from filename
    filename = os.path.basename(log_path)
    category_match = re.search(r"--category=([^_]+(?:_and_[^_]+)*)", filename)
    if category_match:
        results["dataset"] = category_match.group(1).replace("_", " ")

    # Extract hyperparameters from filename
    hyperparam_patterns = {
        "lr": r"--lr=([\d.]+)",
        "temperature": r"--temperature=([\d.]+)",
        "n_codebook": r"--n_codebook=(\d+)",
        "num_beams": r"--num_beams=(\d+)",
        "n_edges": r"--n_edges=(\d+)",
        "propagation_steps": r"--propagation_steps=(\d+)",
    }

    for param, pattern in hyperparam_patterns.items():
        match = re.search(pattern, filename)
        if match:
            results["hyperparams"][param] = match.group(1)

    # Parse log content
    try:
        with open(log_path, "r") as f:
            for line in f:
                # Parse training loss
                train_loss_match = re.search(
                    r"\[Epoch (\d+)\] Train Loss: ([\d.]+)", line
                )
                if train_loss_match:
                    epoch = int(train_loss_match.group(1))
                    loss = float(train_loss_match.group(2))
                    results["train_losses"].append(loss)
                    results["epochs"].append(epoch)

                # Parse validation results
                val_match = re.search(
                    r"\[Epoch \d+\] Val Results: OrderedDict\(({.*})\)", line
                )
                if val_match:
                    metrics_str = val_match.group(1)
                    # Extract metrics
                    ndcg5_match = re.search(r"'ndcg@5': ([\d.e-]+)", metrics_str)
                    ndcg10_match = re.search(r"'ndcg@10': ([\d.e-]+)", metrics_str)
                    recall5_match = re.search(r"'recall@5': ([\d.e-]+)", metrics_str)
                    recall10_match = re.search(r"'recall@10': ([\d.e-]+)", metrics_str)

                    if ndcg5_match:
                        results["val_metrics"]["ndcg@5"].append(
                            float(ndcg5_match.group(1))
                        )
                    if ndcg10_match:
                        results["val_metrics"]["ndcg@10"].append(
                            float(ndcg10_match.group(1))
                        )
                    if recall5_match:
                        results["val_metrics"]["recall@5"].append(
                            float(recall5_match.group(1))
                        )
                    if recall10_match:
                        results["val_metrics"]["recall@10"].append(
                            float(recall10_match.group(1))
                        )

                # Parse best epoch info
                best_epoch_match = re.search(
                    r"Best epoch: (\d+), Best val score: ([\d.e-]+)", line
                )
                if best_epoch_match:
                    results["best_epoch"] = int(best_epoch_match.group(1))
                    results["best_val_score"] = float(best_epoch_match.group(2))

                # Parse test results
                test_match = re.search(r"Test Results: OrderedDict\(({.*})\)", line)
                if test_match:
                    metrics_str = test_match.group(1)
                    ndcg5_match = re.search(r"'ndcg@5': ([\d.e-]+)", metrics_str)
                    ndcg10_match = re.search(r"'ndcg@10': ([\d.e-]+)", metrics_str)
                    recall5_match = re.search(r"'recall@5': ([\d.e-]+)", metrics_str)
                    recall10_match = re.search(r"'recall@10': ([\d.e-]+)", metrics_str)
                    visited_match = re.search(
                        r"'n_visited_items': ([\d.]+)", metrics_str
                    )

                    if ndcg5_match:
                        results["test_results"]["ndcg@5"] = float(ndcg5_match.group(1))
                    if ndcg10_match:
                        results["test_results"]["ndcg@10"] = float(
                            ndcg10_match.group(1)
                        )
                    if recall5_match:
                        results["test_results"]["recall@5"] = float(
                            recall5_match.group(1)
                        )
                    if recall10_match:
                        results["test_results"]["recall@10"] = float(
                            recall10_match.group(1)
                        )
                    if visited_match:
                        results["test_results"]["n_visited_items"] = float(
                            visited_match.group(1)
                        )

    except Exception as e:
        print(f"Error parsing {log_path}: {e}")

    return results


def plot_single_result(results, output_dir, model=None):
    """
    Create a visualization for a single dataset.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(14, 10))

    color_primary = "#2E86AB"
    color_secondary = "#A23B72"

    # 1. Training Loss Over Epochs (Top Left)
    ax1 = plt.subplot(2, 2, 1)
    if results["train_losses"] and results["epochs"]:
        ax1.plot(
            results["epochs"],
            results["train_losses"],
            color=color_primary,
            linewidth=2,
            marker="o",
            markersize=3,
        )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss Over Epochs", fontweight="bold", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 2. Validation NDCG (Top Right)
    ax2 = plt.subplot(2, 2, 2)
    if results["val_metrics"]["ndcg@5"]:
        epochs = list(range(1, len(results["val_metrics"]["ndcg@5"]) + 1))
        ax2.plot(epochs, results["val_metrics"]["ndcg@5"], label="NDCG@5", color=color_primary, linewidth=2, marker="o", markersize=3)
    if results["val_metrics"]["ndcg@10"]:
        epochs = list(range(1, len(results["val_metrics"]["ndcg@10"]) + 1))
        ax2.plot(epochs, results["val_metrics"]["ndcg@10"], label="NDCG@10", color=color_secondary, linewidth=2, marker="s", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("NDCG")
    ax2.set_title("Validation NDCG Over Epochs", fontweight="bold", fontsize=12)
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Validation Recall (Bottom Left)
    ax3 = plt.subplot(2, 2, 3)
    if results["val_metrics"]["recall@5"]:
        epochs = list(range(1, len(results["val_metrics"]["recall@5"]) + 1))
        ax3.plot(epochs, results["val_metrics"]["recall@5"], label="Recall@5", color=color_primary, linewidth=2, marker="o", markersize=3)
    if results["val_metrics"]["recall@10"]:
        epochs = list(range(1, len(results["val_metrics"]["recall@10"]) + 1))
        ax3.plot(epochs, results["val_metrics"]["recall@10"], label="Recall@10", color=color_secondary, linewidth=2, marker="s", markersize=3)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Recall")
    ax3.set_title("Validation Recall Over Epochs", fontweight="bold", fontsize=12)
    ax3.legend(loc="best", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. Info Panel (Bottom Right)
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis("off")

    info_text = f"Dataset: {results['dataset']}\n\n"
    info_text += "Hyperparameters:\n"
    for param, value in results["hyperparams"].items():
        info_text += f"  {param}: {value}\n"

    if results["best_epoch"] is not None:
        info_text += f"\nBest epoch: {results['best_epoch']}\n"
        info_text += f"Best val score: {results['best_val_score']:.6f}\n"

    if results["test_results"]:
        info_text += "\nTest Results:\n"
        for metric, value in results["test_results"].items():
            info_text += f"  {metric}: {value:.6f}\n"

    ax4.text(
        0.05,
        0.95,
        info_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.suptitle(
        f"{model} — {results['dataset']}",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.99])

    output_path = os.path.join(output_dir, "results_visualization.pdf")
    plt.savefig(output_path, bbox_inches="tight")
    print(f"  Saved individual PDF: {output_path}")
    plt.close(fig)


def plot_all_results(log_paths, output_dir, model=None):
    """
    Create a comprehensive visualization of all experiment results.
    """
    log_files = []
    for path in log_paths:
        if os.path.isfile(path):
            log_files.append(path)
        elif os.path.isdir(path):
            log_files.extend(glob.glob(os.path.join(path, "*.log")))

    if not log_files:
        print("No log files found in the provided paths")
        return

    print(f"Found {len(log_files)} log files")

    # Parse all log files
    all_results = []
    for log_file in sorted(log_files):
        print(f"Parsing {os.path.basename(log_file)}...")
        results = parse_log_file(log_file)
        if results["dataset"]:
            all_results.append(results)

    if not all_results:
        print("No valid results found")
        return

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))

    # Define colors for each dataset
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_results)))

    # 1. Training Loss Over Epochs (Top Left)
    ax1 = plt.subplot(3, 3, 1)
    for i, results in enumerate(all_results):
        if results["train_losses"] and results["epochs"]:
            ax1.plot(
                results["epochs"],
                results["train_losses"],
                label=results["dataset"],
                color=colors[i],
                linewidth=2,
                alpha=0.8,
            )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss Over Epochs", fontweight="bold", fontsize=12)
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Validation NDCG@5 Over Epochs (Top Middle)
    ax2 = plt.subplot(3, 3, 2)
    for i, results in enumerate(all_results):
        if results["val_metrics"]["ndcg@5"]:
            epochs = list(range(1, len(results["val_metrics"]["ndcg@5"]) + 1))
            ax2.plot(
                epochs,
                results["val_metrics"]["ndcg@5"],
                label=results["dataset"],
                color=colors[i],
                linewidth=2,
                alpha=0.8,
            )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("NDCG@5")
    ax2.set_title("Validation NDCG@5 Over Epochs", fontweight="bold", fontsize=12)
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Validation NDCG@10 Over Epochs (Top Right)
    ax3 = plt.subplot(3, 3, 3)
    for i, results in enumerate(all_results):
        if results["val_metrics"]["ndcg@10"]:
            epochs = list(range(1, len(results["val_metrics"]["ndcg@10"]) + 1))
            ax3.plot(
                epochs,
                results["val_metrics"]["ndcg@10"],
                label=results["dataset"],
                color=colors[i],
                linewidth=2,
                alpha=0.8,
            )
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("NDCG@10")
    ax3.set_title("Validation NDCG@10 Over Epochs", fontweight="bold", fontsize=12)
    ax3.legend(loc="best", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. Validation Recall@5 Over Epochs (Middle Left)
    ax4 = plt.subplot(3, 3, 4)
    for i, results in enumerate(all_results):
        if results["val_metrics"]["recall@5"]:
            epochs = list(range(1, len(results["val_metrics"]["recall@5"]) + 1))
            ax4.plot(
                epochs,
                results["val_metrics"]["recall@5"],
                label=results["dataset"],
                color=colors[i],
                linewidth=2,
                alpha=0.8,
            )
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Recall@5")
    ax4.set_title("Validation Recall@5 Over Epochs", fontweight="bold", fontsize=12)
    ax4.legend(loc="best", fontsize=9)
    ax4.grid(True, alpha=0.3)

    # 5. Validation Recall@10 Over Epochs (Middle Middle)
    ax5 = plt.subplot(3, 3, 5)
    for i, results in enumerate(all_results):
        if results["val_metrics"]["recall@10"]:
            epochs = list(range(1, len(results["val_metrics"]["recall@10"]) + 1))
            ax5.plot(
                epochs,
                results["val_metrics"]["recall@10"],
                label=results["dataset"],
                color=colors[i],
                linewidth=2,
                alpha=0.8,
            )
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Recall@10")
    ax5.set_title("Validation Recall@10 Over Epochs", fontweight="bold", fontsize=12)
    ax5.legend(loc="best", fontsize=9)
    ax5.grid(True, alpha=0.3)

    # 6. Test NDCG Comparison (Middle Right)
    ax6 = plt.subplot(3, 3, 6)
    datasets_with_test = [r for r in all_results if r["test_results"]]
    if datasets_with_test:
        x_pos = np.arange(len(datasets_with_test))
        width = 0.35

        ndcg5_scores = [r["test_results"].get("ndcg@5", 0) for r in datasets_with_test]
        ndcg10_scores = [
            r["test_results"].get("ndcg@10", 0) for r in datasets_with_test
        ]

        bars1 = ax6.bar(
            x_pos - width / 2, ndcg5_scores, width, label="NDCG@5", alpha=0.8
        )
        bars2 = ax6.bar(
            x_pos + width / 2, ndcg10_scores, width, label="NDCG@10", alpha=0.8
        )

        ax6.set_xlabel("Dataset")
        ax6.set_ylabel("NDCG Score")
        ax6.set_title("Test NDCG Comparison", fontweight="bold", fontsize=12)
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(
            [r["dataset"] for r in datasets_with_test], rotation=45, ha="right"
        )
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax6.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # 7. Test Recall Comparison (Bottom Left)
    ax7 = plt.subplot(3, 3, 7)
    if datasets_with_test:
        x_pos = np.arange(len(datasets_with_test))
        width = 0.35

        recall5_scores = [
            r["test_results"].get("recall@5", 0) for r in datasets_with_test
        ]
        recall10_scores = [
            r["test_results"].get("recall@10", 0) for r in datasets_with_test
        ]

        bars1 = ax7.bar(
            x_pos - width / 2, recall5_scores, width, label="Recall@5", alpha=0.8
        )
        bars2 = ax7.bar(
            x_pos + width / 2, recall10_scores, width, label="Recall@10", alpha=0.8
        )

        ax7.set_xlabel("Dataset")
        ax7.set_ylabel("Recall Score")
        ax7.set_title("Test Recall Comparison", fontweight="bold", fontsize=12)
        ax7.set_xticks(x_pos)
        ax7.set_xticklabels(
            [r["dataset"] for r in datasets_with_test], rotation=45, ha="right"
        )
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax7.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # 8. Best Validation Score and Best Epoch (Bottom Middle)
    ax8 = plt.subplot(3, 3, 8)
    datasets_with_best = [r for r in all_results if r["best_val_score"] is not None]
    if datasets_with_best:
        x_pos = np.arange(len(datasets_with_best))
        best_scores = [r["best_val_score"] for r in datasets_with_best]

        bars = ax8.bar(
            x_pos, best_scores, alpha=0.8, color=colors[: len(datasets_with_best)]
        )
        ax8.set_xlabel("Dataset")
        ax8.set_ylabel("Best Validation Score (NDCG@10)")
        ax8.set_title("Best Validation Performance", fontweight="bold", fontsize=12)
        ax8.set_xticks(x_pos)
        ax8.set_xticklabels(
            [r["dataset"] for r in datasets_with_best], rotation=45, ha="right"
        )
        ax8.grid(True, alpha=0.3, axis="y")

        # Add best epoch labels on bars
        for i, (bar, result) in enumerate(zip(bars, datasets_with_best)):
            height = bar.get_height()
            ax8.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}\n(epoch {result['best_epoch']})",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # 9. Hyperparameters Summary (Bottom Right)
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis("off")

    # Create hyperparameter summary text
    summary_text = "Hyperparameters Summary\n" + "=" * 40 + "\n\n"
    for results in all_results:
        summary_text += f"{results['dataset']}:\n"
        for param, value in results["hyperparams"].items():
            summary_text += f"  {param}: {value}\n"
        if results["best_epoch"] is not None:
            summary_text += f"  best_epoch: {results['best_epoch']}\n"
        if results["best_val_score"] is not None:
            summary_text += f"  best_val_score: {results['best_val_score']:.6f}\n"
        summary_text += "\n"

    ax9.text(
        0.05,
        0.95,
        summary_text,
        transform=ax9.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    # Overall title
    fig.suptitle(
        f"{model} Model Performance Across All Datasets",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.99])

    # Also save as PDF for better quality
    output_path_pdf = f"{output_dir}/results_visualization.pdf"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path_pdf, bbox_inches="tight")
    print(f"Visualization saved to: {output_path_pdf}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    for results in all_results:
        print(f"\n{results['dataset']}:")
        print(f"  Total epochs: {len(results['train_losses'])}")
        if results["best_epoch"]:
            print(f"  Best epoch: {results['best_epoch']}")
            print(f"  Best validation score: {results['best_val_score']:.6f}")
        if results["test_results"]:
            print("  Test Results:")
            for metric, value in results["test_results"].items():
                print(f"    {metric}: {value:.6f}")


def extract_metadata_from_filename(filepath):
    """Extract model, category, and timestamp from a log filename."""
    filename = os.path.basename(filepath)
    model_m = re.search(r"--model=([^_]+)", filename)
    cat_m = re.search(r"--category=([^_]+(?:_and_[^_]+)*)", filename)
    time_m = re.search(
        r"-([A-Z][a-z]{2}-\d{2}-\d{4}_\d{2}-\d{2}(?:-[a-z0-9]+)?)\.log", filename
    )
    model = model_m.group(1) if model_m else "UnknownModel"
    category = cat_m.group(1) if cat_m else "UnknownCategory"
    timestamp = time_m.group(1) if time_m else "UnknownTime"
    return model, category, timestamp


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize RPG experiment results")
    parser.add_argument(
        "--log_paths",
        nargs="+",
        default=[
            # "logs/AmazonReviews2014/DiffAR/genrec_default-main.py_--model=DiffAR_--category=Toys_and_Games_--lr=0.003_--temperature=0.03-May-05-2026_18-52-833a9d.log",
            # "logs/AmazonReviews2014/DiffAR/genrec_default-main.py_--model=DiffAR_--category=Sports_and_Outdoors_--lr=0.003_--temperature=0.03-May-05-2026_18-53-91572e.log",
            # "logs/AmazonReviews2014/DiffAR/genrec_default-main.py_--model=DiffAR_--category=CDs_and_Vinyl_--lr=0.001_--temperature=0.03-May-05-2026_18-53-2eeead.log",
            # "logs/AmazonReviews2014/DiffAR/genrec_default-main.py_--model=DiffAR_--category=Beauty_--lr=0.01_--temperature=0.03_--n_codebook=32_--num_beams=20_--n_edges=200_--propagation_steps=3-May-05-2026_18-31-d5929c.log",
            # "logs/AmazonReviews2014/RPG/genrec_default-main.py_--category=Beauty_--lr=0.01_--temperature=0.03_--n_codebook=32_--num_beams=20_--n_edges=200_--propagation_steps=3-Apr-27-2026_20-30-ba460a.log"
            # # "logs/AmazonReviews2014/RPG/genrec_default-main.py_--category=CDs_and_Vinyl_--lr=0.001_--temperature=0.03_--n_codebook=64_--num_beams=20_--n_edges=500_--propagation_steps=5-May-06-2026_01-21-70824e.log",
            # "logs/AmazonReviews2014/RPG/genrec_default-main.py_--category=Sports_and_Outdoors_--lr=0.003_--temperature=0.03_--n_codebook=16_--num_beams=100_--n_edges=30_--propagation_steps=5-May-06-2026_01-21-52057a.log",
            # "logs/AmazonReviews2014/RPG/genrec_default-main.py_--category=Toys_and_Games_--lr=0.003_--temperature=0.03_--n_codebook=16_--num_beams=200_--n_edges=20_--propagation_steps=3-May-06-2026_01-22-2b595c.log",
            "logs/AmazonReviews2014/RPGDiff"
        ],
        help="List of full paths to the log files (or directories)",
    )

    args = parser.parse_args()

    # Resolve log files (files or directories)
    log_files = []
    for path in args.log_paths:
        if os.path.isfile(path):
            log_files.append(path)
        elif os.path.isdir(path):
            log_files.extend(glob.glob(os.path.join(path, "*.log")))

    if not log_files:
        print("No log files found in the provided paths")
        exit(1)

    print(f"Found {len(log_files)} log files")

    # Determine model from first log file
    model, _, _ = extract_metadata_from_filename(log_files[0])

    # Step 1: Generate individual per-dataset PDFs
    all_results = []
    for log_file in sorted(log_files):
        print(f"Processing {os.path.basename(log_file)}...")
        results = parse_log_file(log_file)
        if results["dataset"]:
            all_results.append(results)
            mdl, cat, ts = extract_metadata_from_filename(log_file)
            individual_output_dir = os.path.join("vis_results", mdl, cat, ts)
            plot_single_result(results, individual_output_dir, model=mdl)

    if not all_results:
        print("No valid results found")
        exit(1)

    # Step 2: Generate combined comparison PDF
    first_ts = None
    for log_file in sorted(log_files):
        _, _, ts = extract_metadata_from_filename(log_file)
        first_ts = ts
        break

    combined_output_dir = os.path.join("vis_results", model, f"combined_{first_ts}")
    plot_all_results(log_files, combined_output_dir, model)
