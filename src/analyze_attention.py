from typing import Dict, List, Tuple
import torch


def _find_name_spans(text: str, name: str) -> List[Tuple[int, int]]:
    """
    Encuentra todas las apariciones de `name` dentro de `text` y devuelve
    una lista de pares (start, end) en índices de carácter.
    """
    spans: List[Tuple[int, int]] = []
    start = 0
    while True:
        idx = text.find(name, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(name)))
        start = idx + len(name)
    return spans


def _get_offsets(enc) -> List[Tuple[int, int]]:
    """
    Normaliza el formato de offset_mapping para que siempre sea
    una lista de pares (start, end), sin dimensión de batch.
    """
    offsets = enc["offset_mapping"]

    # Puede venir como tensor -> lo pasamos a lista
    if isinstance(offsets, torch.Tensor):
        offsets = offsets.tolist()

    # Si viene como lista anidada [ [ (start, end), ... ] ], cogemos el primer elemento
    # (porque solo estamos tokenizando una cadena)
    if (
        len(offsets) > 0
        and isinstance(offsets[0], (list, tuple))
        and len(offsets[0]) > 0
        and isinstance(offsets[0][0], (list, tuple))
    ):
        offsets = offsets[0]

    return offsets  # lista de (start, end)


def get_character_token_indices(tokenizer, prompt: str, character_names: List[str]) -> Dict[str, List[int]]:
    """
    Devuelve, para cada personaje, los índices de tokens en el PROMPT donde aparece su nombre,
    usando offsets de caracteres en vez de comparar tokens directamente.
    """
    enc = tokenizer(
        prompt,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = _get_offsets(enc)  # lista de (start, end)

    mapping: Dict[str, List[int]] = {name: [] for name in character_names}

    for name in character_names:
        name_spans = _find_name_spans(prompt, name)
        if not name_spans:
            continue

        for tok_idx, (tok_start, tok_end) in enumerate(offsets):
            # Si el token no cubre ningún carácter real, lo ignoramos
            if tok_end <= tok_start:
                continue

            # Comprobamos solapamiento con alguna ocurrencia del nombre
            for span_start, span_end in name_spans:
                # solapamiento si los intervalos [tok_start, tok_end) y [span_start, span_end)
                # se cruzan en algún punto
                if tok_start < span_end and tok_end > span_start:
                    mapping[name].append(tok_idx)
                    break  # pasamos al siguiente token

    return mapping


def compute_attention_character_scores(
    tokenizer,
    model,
    prompt: str,
    full_text: str,
    character_names: List[str],
    device: str,
) -> Dict[str, float]:
    """
    Calcula un score de atención por personaje, usando las self-attentions del modelo.

    - prompt: solo el prompt (Characters + Setting + Task + Story:)
    - full_text: prompt + historia generada
    - character_names: lista de nombres, p.ej. ["Luna", "Orion"]
    """
    model.eval()

    # 1) Tokenizar prompt (sin special tokens) para obtener su longitud en tokens
    enc_prompt = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    prompt_ids = enc_prompt.input_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    # Mapa de tokens del prompt que pertenecen a cada personaje
    char_token_indices = get_character_token_indices(tokenizer, prompt, character_names)

    # 2) Tokenizar texto completo (sin special tokens)
    enc_full = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    full_ids = enc_full.input_ids.to(device)
    attention_mask = enc_full.attention_mask.to(device)
    seq_len = full_ids.shape[1]

    # 3) Ejecutar modelo con attentions
    with torch.no_grad():
        outputs = model(
            full_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

    attentions = outputs.attentions  # tupla de (num_layers) cada una (batch, heads, seq, seq) o None

    # Por si acaso, fallback si algo viene a None
    if attentions is None:
        n = len(character_names)
        return {name: 1.0 / n for name in character_names}

    attentions = [a for a in attentions if a is not None]
    if len(attentions) == 0:
        n = len(character_names)
        return {name: 1.0 / n for name in character_names}

    # 4) Apilar y promediar sobre capas y cabezas
    attn = torch.stack(attentions, dim=0)  # (L, B, H, S, S)
    attn = attn[:, 0]                      # (L, H, S, S)
    attn_mean = attn.mean(dim=(0, 1))      # (S, S)

    # 5) Filas de tokens generados (después del prompt)
    gen_start = prompt_len
    gen_indices = list(range(gen_start, seq_len))

    scores = {name: 0.0 for name in character_names}
    total_mass = 0.0

    for i in gen_indices:
        row = attn_mean[i]  # (S,)
        total_mass += float(row[:prompt_len].sum().item())

        for name, idxs in char_token_indices.items():
            if not idxs:
                continue
            val = float(row[idxs].sum().item())
            scores[name] += val

    if total_mass > 0:
        for name in scores:
            scores[name] /= total_mass

    return scores
