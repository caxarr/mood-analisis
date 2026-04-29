import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Visual style

def set_report_style() -> None:
    """Clean academic/report style for all figures."""
    plt.style.use("default")
    plt.rcParams.update({
        "figure.figsize": (10.5, 6.2),
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.edgecolor": "#D0D0D0",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E6E6E6",
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.size": 11,
        "legend.frameon": False,
        "legend.fontsize": 10,
    })


def _safe_name(text: str) -> str:
    return (
        text.strip().lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )

# Plot helpers

def _save_correlation_plot(corr: pd.Series, target: str, dataset_name: str, out_dir: Path) -> None:
    corr_plot = corr.drop(labels=[target], errors="ignore").sort_values()
    colors = ["#B65E5A" if v < 0 else "#4C78A8" for v in corr_plot.values]

    fig, ax = plt.subplots()
    ax.barh(corr_plot.index, corr_plot.values, color=colors, edgecolor="none")
    ax.axvline(0, color="#444444", linewidth=1)
    ax.set_title(f"{dataset_name}: Correlation with {target}")
    ax.set_xlabel("Pearson correlation")
    ax.set_ylabel("Feature")

    for i, v in enumerate(corr_plot.values):
        ha = "left" if v >= 0 else "right"
        offset = 0.015 if v >= 0 else -0.015
        ax.text(v + offset, i, f"{v:.2f}", va="center", ha=ha, fontsize=9)

    plt.tight_layout()
    fig.savefig(out_dir / f"{_safe_name(dataset_name)}_correlation.png", bbox_inches="tight")
    plt.close(fig)


def _save_variance_plot(var_ratio: np.ndarray, dataset_name: str, out_dir: Path) -> None:
    pcs = np.arange(1, len(var_ratio) + 1)
    cumulative = np.cumsum(var_ratio)

    fig, ax = plt.subplots()
    bars = ax.bar(pcs, var_ratio, color="#4C78A8", width=0.7, edgecolor="none", label="Explained variance")
    ax.plot(pcs, cumulative, marker="o", linewidth=2.0, color="#222222", label="Cumulative variance")
    ax.set_xticks(pcs)
    ax.set_ylim(0, min(1.05, max(1.0, cumulative.max() + 0.05)))
    ax.set_title(f"{dataset_name}: Explained variance by principal components")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance ratio")

    for rect, v in zip(bars, var_ratio):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.015, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)

    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / f"{_safe_name(dataset_name)}_variance.png", bbox_inches="tight")
    plt.close(fig)


def _save_pc1_loadings_plot(pc1: pd.Series, dataset_name: str, out_dir: Path) -> None:
    pc1_plot = pc1.sort_values()
    colors = ["#B65E5A" if v < 0 else "#59A14F" for v in pc1_plot.values]

    fig, ax = plt.subplots()
    ax.barh(pc1_plot.index, pc1_plot.values, color=colors, edgecolor="none")
    ax.axvline(0, color="#444444", linewidth=1)
    ax.set_title(f"{dataset_name}: PC1 loadings")
    ax.set_xlabel("Loading")
    ax.set_ylabel("Feature")

    for i, v in enumerate(pc1_plot.values):
        ha = "left" if v >= 0 else "right"
        offset = 0.015 if v >= 0 else -0.015
        ax.text(v + offset, i, f"{v:.2f}", va="center", ha=ha, fontsize=9)

    plt.tight_layout()
    fig.savefig(out_dir / f"{_safe_name(dataset_name)}_pc1_loadings.png", bbox_inches="tight")
    plt.close(fig)


def _save_pc_scatter(scores: np.ndarray, dataset_name: str, out_dir: Path) -> None:
    if scores.shape[1] < 2:
        return

    fig, ax = plt.subplots()
    ax.scatter(scores[:, 0], scores[:, 1], s=32, alpha=0.75, color="#4C78A8", edgecolors="none")
    ax.axhline(0, color="#BDBDBD", linewidth=1)
    ax.axvline(0, color="#BDBDBD", linewidth=1)
    ax.set_title(f"{dataset_name}: PC1 vs PC2 scores")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    fig.savefig(out_dir / f"{_safe_name(dataset_name)}_pc1_vs_pc2.png", bbox_inches="tight")
    plt.close(fig)


def _save_heatmap(corr_matrix: pd.DataFrame, dataset_name: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    im = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_matrix.index)
    ax.set_title(f"{dataset_name}: Correlation matrix")

    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            val = corr_matrix.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(val) > 0.55 else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation")
    plt.tight_layout()
    fig.savefig(out_dir / f"{_safe_name(dataset_name)}_heatmap.png", bbox_inches="tight")
    plt.close(fig)

# Analysis pipeline

def analyze_dataset(df: pd.DataFrame, target: str, dataset_name: str, output_dir: str = "data/plots") -> None:
    """
    Runs a clean report-style analysis for one dataset:
    - numeric subset
    - correlations with target
    - full correlation matrix
    - PCA explained variance
    - PC1 loadings
    - PC1 vs PC2 scatter
    - saves CSV summaries
    """
    set_report_style()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print(f"DATASET: {dataset_name}")
    print(f"TARGET:  {target}")
    print("=" * 72)

    numeric_df = df.select_dtypes(include="number").dropna().copy()
    if target not in numeric_df.columns:
        print(f"Target '{target}' not found among numeric columns.")
        return

    features = [c for c in numeric_df.columns if c != target]
    if len(features) < 2:
        print("Not enough numeric features for PCA.")
        return

    X = numeric_df[features]
    corr_matrix = numeric_df[features + [target]].corr(numeric_only=True)
    corr_target = corr_matrix[target].sort_values(ascending=False)

    print("\nCorrelations with target:")
    print(corr_target)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    scores = pca.fit_transform(X_scaled)
    var_ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(var_ratio)

    print("\nExplained variance ratio:")
    for i, value in enumerate(var_ratio, start=1):
        print(f"PC{i}: {value:.4f} | cumulative: {cumulative[i-1]:.4f}")

    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=[f"PC{i}" for i in range(1, len(features) + 1)],
    )

    pc1 = loadings["PC1"].sort_values(key=lambda s: s.abs(), ascending=False)

    print("\nPC1 loadings:")
    print(pc1)

    # Save tables
    safe = _safe_name(dataset_name)
    corr_target.to_csv(out_dir / f"{safe}_correlations.csv", header=["correlation"])
    loadings.to_csv(out_dir / f"{safe}_pca_loadings.csv")
    pd.DataFrame({
        "component": [f"PC{i}" for i in range(1, len(var_ratio) + 1)],
        "explained_variance_ratio": var_ratio,
        "cumulative_variance_ratio": cumulative,
    }).to_csv(out_dir / f"{safe}_variance.csv", index=False)

    # Save plots
    _save_correlation_plot(corr_target, target, dataset_name, out_dir)
    _save_heatmap(corr_matrix.loc[features + [target], features + [target]], dataset_name, out_dir)
    _save_variance_plot(var_ratio, dataset_name, out_dir)
    _save_pc1_loadings_plot(pc1, dataset_name, out_dir)
    _save_pc_scatter(scores, dataset_name, out_dir)

    print(f"\nSaved plots and tables to: {out_dir.resolve()}")
