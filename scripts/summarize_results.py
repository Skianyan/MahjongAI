#!/usr/bin/env python3
"""
Resumen de experimentos MahjongAI (offline + local + fallback).

Uso:
  python scripts/summarize_results.py --results-dir results

Supone archivos como:
  results/e1_offline.json
  results/e1_local.json
  results/e2_offline.json
  results/e2_local.json
  ...
  results/fallback_local.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt_float(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    results_dir = args.results_dir
    fallback_path = results_dir / "fallback_local.json"
    if not fallback_path.exists():
        raise FileNotFoundError(f"No existe {fallback_path}")

    fallback_local = load_json(fallback_path)
    fb_score = float(fallback_local.get("mean_score", 0.0))
    fb_rank = float(fallback_local.get("mean_rank", 0.0))
    fb_fp = float(fallback_local.get("first_place_rate", 0.0))

    rows: list[dict] = []
    for offline_path in sorted(results_dir.glob("*_offline.json")):
        stem = offline_path.name.removesuffix("_offline.json")
        local_path = results_dir / f"{stem}_local.json"
        if not local_path.exists():
            continue

        off = load_json(offline_path)
        loc = load_json(local_path)

        overall = off.get("overall", {})
        row = {
            "exp": stem,
            "off_examples": int(overall.get("examples", 0)),
            "off_top1": float(overall.get("top1_accuracy", 0.0)),
            "off_topk": float(overall.get("topk_accuracy", 0.0)),
            "off_illegal": float(overall.get("raw_top1_illegal_rate", 0.0)),
            "loc_games": int(loc.get("games", 0)),
            "loc_score": float(loc.get("mean_score", 0.0)),
            "loc_rank": float(loc.get("mean_rank", 0.0)),
            "loc_fp": float(loc.get("first_place_rate", 0.0)),
            "loc_timeout": int(loc.get("timeout_actions", 0)),
            "loc_tie": float(loc.get("tie_rate", 0.0)),
            "loc_unfinished": float(loc.get("unfinished_game_rate", 0.0)),
        }
        row["d_score"] = row["loc_score"] - fb_score
        row["d_rank"] = row["loc_rank"] - fb_rank
        row["d_fp"] = row["loc_fp"] - fb_fp
        rows.append(row)

    if not rows:
        print("No se encontraron pares *_offline.json + *_local.json en", results_dir)
        return

    # Orden: mejor delta rank (más negativo), luego delta score (más alto), luego delta fp (más alto)
    rows.sort(key=lambda r: (r["d_rank"], -r["d_score"], -r["d_fp"]))

    headers = [
        "exp",
        "off_examples",
        "off_top1",
        "off_topk",
        "off_illegal",
        "loc_games",
        "loc_score",
        "loc_rank",
        "loc_fp",
        "loc_timeout",
        "loc_tie",
        "loc_unfinished",
        "d_score",
        "d_rank",
        "d_fp",
    ]

    def render(v, key: str) -> str:
        if key in {
            "off_top1",
            "off_topk",
            "off_illegal",
            "loc_rank",
            "loc_fp",
            "loc_tie",
            "loc_unfinished",
            "d_rank",
            "d_fp",
        }:
            return fmt_float(float(v), 4)
        if key in {"loc_score", "d_score"}:
            return fmt_float(float(v), 1)
        return str(v)

    table = [headers]
    for r in rows:
        table.append([render(r[h], h) for h in headers])

    col_widths = [max(len(row[i]) for row in table) for i in range(len(headers))]

    def line(row: list[str]) -> str:
        return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))

    print("\nFallback local (referencia):")
    print(
        f"  mean_score={fb_score:.1f}  mean_rank={fb_rank:.4f}  first_place_rate={fb_fp:.4f}\n"
    )

    print(line(table[0]))
    print("-+-".join("-" * w for w in col_widths))
    for row in table[1:]:
        print(line(row))

    best = rows[0]
    print("\nMejor candidato (según orden actual):", best["exp"])
    print(
        "Criterio usado: menor d_rank, luego mayor d_score, luego mayor d_fp "
        "(comparado contra fallback)."
    )


if __name__ == "__main__":
    main()