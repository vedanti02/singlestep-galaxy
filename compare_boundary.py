"""Compare boundary_error.csv files across runs / checkpoints.

Prints a side-by-side table of per-distance `rel_to_zero` plus the
inner-cube aggregate. Useful for confirming that an arch change
(e.g. the c_lf_pt=0 fix) actually improves the inference path.

Usage:
    python compare_boundary.py path/to/boundary_error.csv [path2] [...]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path


def load(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def inner_agg(rows: list[dict], buf: int = 4) -> tuple[float, float, float]:
    """Mean over distances >= buf, weighted by per-bin count."""
    n_total = sum(int(r["count"]) for r in rows if int(r["distance_from_edge"]) >= buf)
    m = sum(float(r["mean_sq_error"]) * int(r["count"])
            for r in rows if int(r["distance_from_edge"]) >= buf) / max(n_total, 1)
    z = sum(float(r["baseline_zero"]) * int(r["count"])
            for r in rows if int(r["distance_from_edge"]) >= buf) / max(n_total, 1)
    return m, z, (m / z if z > 0 else float("nan"))


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print(__doc__)
        sys.exit(2)

    tables = {p: load(p) for p in paths}
    distances = sorted({int(r["distance_from_edge"]) for rows in tables.values()
                        for r in rows})

    # Header
    label_w = max(len(p.parent.name + "/" + p.name) for p in paths) + 2
    print("dist", " " * (label_w - 4),
          *[f"{p.parent.parent.name}/{p.parent.name}".ljust(40)[:40] for p in paths])
    print(" " * label_w, *[f"{'rel_zero'.center(40)}" for _ in paths])

    # Rows
    for d in distances:
        line = f"{d:>4} {'':<{label_w-4}}"
        for p in paths:
            row = next((r for r in tables[p]
                        if int(r["distance_from_edge"]) == d), None)
            if row is None:
                cell = "  --  "
            else:
                cell = f"{float(row['rel_to_zero']):>8.4f}"
            line += f" {cell:<40}"
        print(line)

    print()
    print("INNER-CUBE AGGREGATE (distance >= buf=4):")
    for p in paths:
        m, z, r = inner_agg(tables[p])
        verdict = "BEATS zero" if r < 1.0 else "WORSE than zero"
        print(f"  {str(p):>60s}  model_mse={m:.4e}  zero_mse={z:.4e}  "
              f"rel={r:.4f}  -> {verdict}")


if __name__ == "__main__":
    main()
