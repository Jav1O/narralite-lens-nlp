import os
import sys
import json
from pathlib import Path

# Añadir la carpeta raíz del proyecto al path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.prompts import example_scenarios, build_prompt
from src.generate import load_model, generate_story
from tqdm import tqdm

def main():
    tokenizer, model, device = load_model()
    scenarios = example_scenarios()
    seeds = [0, 1, 2]

    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    out_path = data_dir / "stories_raw.jsonl"
    records = []

    for scen in tqdm(scenarios, desc="Scenarios"):
        for seed in seeds:
            prompt = build_prompt(scen)
            out = generate_story(
                tokenizer=tokenizer,
                model=model,
                prompt=prompt,
                device=device,
                seed=seed,
                max_new_tokens=120,
                temperature=0.8,
                top_p=0.9,
                return_attentions=False,
            )
            rec = {
                "scenario": scen.name,
                "seed": seed,
                "prompt": out["prompt"],
                "text": out["text"],
            }
            records.append(rec)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} stories to {out_path}")

if __name__ == "__main__":
    main()
