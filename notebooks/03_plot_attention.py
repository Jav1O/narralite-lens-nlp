import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils_plot import compute_mean_attention_share, plot_attention_share


def main():
    csv_path = ROOT / "results" / "attention_scores.csv"
    out_path = ROOT / "plots" / "attention_share.png"

    mean_df = compute_mean_attention_share(csv_path)
    print("Mean attention share:")
    print(mean_df)

    plot_attention_share(mean_df, save_path=out_path)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
