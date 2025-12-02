NLP-FINAL-PROJECT
github repository for the nlp project, i am Erasmus+ estudent and i will give my best to do this project.
# Narrative Lens: Analyzing Character Influence in Story Generation

This repository contains the code for the final project of the **Natural Language Processing** course.

The goal of the project is to analyze how a Transformer-based language model (DistilGPT-2) distributes *narrative focus* among characters while generating short stories from structured prompts. We use self-attention and gradient-based saliency to obtain character-level interpretability metrics.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
