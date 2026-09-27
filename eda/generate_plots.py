"""Renders the EDA charts referenced in explore_data.py and README.md as
saved PNG files in eda/plots/.

The plotting code in explore_data.py originally used plt.show() (for
interactive viewing during EDA) and was never saved to disk. This script
reuses the exact same chart definitions - same data, same logic, same
findings - and just saves them as files so they can be embedded in the
README and report. No new analysis is performed here.

The one addition (market_index_train_vs_validation.png) visualizes a
finding that was already computed numerically during EDA (the train vs
validation market_index distribution shift) but never plotted.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
PLOTS_DIR = Path(__file__).parent / "plots"


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_DIR / "train-test.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")

    # 1. Distance vs Posted Rate
    plt.figure(figsize=(9, 6))
    plt.scatter(train["distance"], train["posted_rate"], alpha=0.2, s=10)
    plt.xlabel("Distance")
    plt.ylabel("Posted Rate")
    plt.title("Distance vs Posted Rate")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "distance_vs_posted_rate.png", dpi=150)
    plt.close()

    # 2. Posted Rate Distribution
    plt.figure(figsize=(9, 6))
    plt.hist(train["posted_rate"], bins=50)
    plt.xlabel("Posted Rate")
    plt.ylabel("Number of Loads")
    plt.title("Posted Rate Distribution")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "posted_rate_distribution.png", dpi=150)
    plt.close()

    # 3. Posted Rate by Equipment
    plt.figure(figsize=(8, 6))
    train.boxplot(column="posted_rate", by="equipment")
    plt.xlabel("Equipment")
    plt.ylabel("Posted Rate")
    plt.title("Posted Rate by Equipment")
    plt.suptitle("")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "posted_rate_by_equipment.png", dpi=150)
    plt.close()

    # 4. Market Index vs Posted Rate
    plt.figure(figsize=(9, 6))
    plt.scatter(train["market_index"], train["posted_rate"], alpha=0.2, s=10)
    plt.xlabel("Market Index")
    plt.ylabel("Posted Rate")
    plt.title("Market Index vs Posted Rate")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "market_index_vs_posted_rate.png", dpi=150)
    plt.close()

    # 5. Quote Signal vs Posted Rate
    plt.figure(figsize=(9, 6))
    plt.scatter(train["quote_signal"], train["posted_rate"], alpha=0.2, s=10)
    plt.xlabel("Quote Signal")
    plt.ylabel("Posted Rate")
    plt.title("Quote Signal vs Posted Rate")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "quote_signal_vs_posted_rate.png", dpi=150)
    plt.close()

    # 6. Weight vs Posted Rate (negative weights visibly separate on the left)
    plt.figure(figsize=(9, 6))
    plt.scatter(train["weight"], train["posted_rate"], alpha=0.2, s=10)
    plt.xlabel("Weight")
    plt.ylabel("Posted Rate")
    plt.title("Weight vs Posted Rate")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "weight_vs_posted_rate.png", dpi=150)
    plt.close()

    # 7. Train vs Validation market_index distribution shift
    plt.figure(figsize=(9, 6))
    plt.hist(train["market_index"], bins=40, alpha=0.5, label="Train", density=True)
    plt.hist(validation["market_index"], bins=40, alpha=0.5, label="Validation", density=True)
    plt.xlabel("Market Index")
    plt.ylabel("Density")
    plt.title("Market Index: Train vs Validation Distribution")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "market_index_train_vs_validation.png", dpi=150)
    plt.close()

    print(f"Saved 7 EDA charts to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
