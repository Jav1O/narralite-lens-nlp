from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def compute_mean_attention_share(csv_path: Path) -> pd.DataFrame:
    """
    Lee el CSV de attention_scores y devuelve un DataFrame con:
      - scenario
      - character
      - attention_score medio
      - share (normalizado dentro de cada escenario, entre 0 y 1)
    """
    df = pd.read_csv(csv_path)

    mean = df.groupby(["scenario", "character"])["attention_score"].mean().reset_index()
    mean["share"] = mean["attention_score"] / mean.groupby("scenario")["attention_score"].transform("sum")

    return mean


def plot_attention_share(mean_df: pd.DataFrame, save_path: Optional[Path] = None):
    """
    Dibuja un gráfico de barras de la atención por personaje y escenario.

    - mean_df debe tener columnas: scenario, character, share
    - si save_path no es None, guarda la figura en esa ruta
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # Creamos etiquetas tipo "enchanted_forest\nLuna"
    labels = [
        f"{row['scenario']}\n{row['character']}" for _, row in mean_df.iterrows()
    ]
    values = mean_df["share"].values

    ax.bar(labels, values)

    ax.set_ylabel("Attention share")
    ax.set_ylim(0, 1)
    ax.set_title("Character-level attention share per scenario")

    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    # Si quieres ver la gráfica al ejecutar el script:
    # plt.show()

    plt.close(fig)
