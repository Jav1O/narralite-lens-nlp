import os
import sys
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.prompts import example_scenarios
from src.generate import load_model
from src.analyze_attention import compute_attention_character_scores


def main():
    # 1) Cargar modelo
    tokenizer, model, device = load_model()

    # 2) Mapa de escenarios -> lista de personajes
    scenarios = example_scenarios()
    scen_map = {s.name: s for s in scenarios}

    # 3) Cargar historias generadas
    data_path = ROOT / "data" / "stories_raw.jsonl"
    records = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    # 4) Crear carpeta de resultados
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_csv = results_dir / "attention_scores.csv"

    rows = []

    # 5) Para cada historia, calcular atención por personaje
    for rec in tqdm(records, desc="Analyzing attention"):
        scen_name = rec["scenario"]
        prompt = rec["prompt"]
        text = rec["text"]
        seed = rec["seed"]

        scenario = scen_map[scen_name]
        character_names = [ch.name for ch in scenario.characters]

        scores = compute_attention_character_scores(
            tokenizer=tokenizer,
            model=model,
            prompt=prompt,
            full_text=text,
            character_names=character_names,
            device=device,
        )

        for name, score in scores.items():
            rows.append(
                {
                    "scenario": scen_name,
                    "seed": seed,
                    "character": name,
                    "attention_score": score,
                }
            )

    # 6) Guardar en CSV
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved attention scores to {out_csv}")


if __name__ == "__main__":
    main()
