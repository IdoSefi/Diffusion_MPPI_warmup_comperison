#!/usr/bin/env python3
"""
Plot mean env steps vs n_actions from the wrapper's combined_all_arches.json.
Supports multiple input JSON files to compare curves across different runs/architectures.

- One curve per architecture (experiment) per file.
- Y: mean episode steps (lower is better)
- X: apply_first_n_actions (n_actions)
- Shaded band: configurable (default: 95% CI over episodes per point)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt


def _pretty_name(exp_key: str) -> str:
    k = exp_key.strip().lower()
    if k == "mppi_only":
        return "MPPI only"
    if k.startswith("warmstart_diffusion_"):
        arch = k.split("warmstart_diffusion_", 1)[1]
        return f"MPPI + Diffusion warmstart ({arch.upper()})"
    # fallback: make it readable
    return exp_key.replace("_", " ")


def _band_from_steps(steps: np.ndarray, band: str) -> Tuple[float, float, float]:
    """
    Returns (mean, lo, hi) for shading.
    band ∈ {"none","std","sem","ci95"} computed over episodes for this point.
    """
    mean = float(np.mean(steps)) if steps.size else float("nan")
    if steps.size <= 1 or band == "none":
        return mean, mean, mean

    std = float(np.std(steps, ddof=1))
    sem = std / math.sqrt(steps.size)

    if band == "std":
        delta = std
    elif band == "sem":
        delta = sem
    elif band == "ci95":
        delta = 1.96 * sem
    else:
        raise ValueError(f"Unknown band: {band}")

    return mean, mean - delta, mean + delta


def load_series(json_paths: List[Path], band: str) -> Dict[str, Dict[str, Any]]:
    """
    Output format:
      series[unique_key] = {
          "x": [...], 
          "y": [...], 
          "lo": [...], 
          "hi": [...],
          "label": "Display Name"
      }
    """
    series: Dict[str, Dict[str, Any]] = {}
    
    # If multiple files are provided, we append the filename to the label 
    # to distinguish curves (e.g. "MPPI only (seed1)" vs "MPPI only (seed2)")
    multiple_inputs = len(json_paths) > 1

    for path in json_paths:
        if not path.exists():
            print(f"Warning: {path} does not exist, skipping.")
            continue

        with path.open("r") as f:
            root = json.load(f)

        experiments = root.get("experiments", {})

        for exp_key, exp in experiments.items():
            runs = exp.get("runs", [])
            xs, ys, los, his = [], [], [], []

            for r in runs:
                if r.get("status") != "ok":
                    continue

                n_actions = int(r["n_actions"])

                # Prefer per-episode steps if present (best for band); otherwise fall back to summary mean_steps.
                ep_steps = None
                results = r.get("results", {})
                if isinstance(results, dict):
                    episodes = results.get("episodes", [])
                    if isinstance(episodes, list) and episodes:
                        ep_steps = np.array([e["steps"] for e in episodes], dtype=np.float32)

                if ep_steps is not None and ep_steps.size > 0:
                    mean, lo, hi = _band_from_steps(ep_steps, band=band)
                else:
                    mean = float(r.get("summary", {}).get("mean_steps", float("nan")))
                    lo, hi = mean, mean  # no episodes => no band

                xs.append(n_actions)
                ys.append(mean)
                los.append(lo)
                his.append(hi)

            if not xs:
                continue

            # sort by x
            order = np.argsort(xs)
            
            # Construct label and unique key
            base_name = _pretty_name(exp_key)
            if multiple_inputs:
                label = f"{base_name} ({path.stem})"
                unique_key = f"{path.stem}_{exp_key}"
            else:
                label = base_name
                unique_key = exp_key

            series[unique_key] = {
                "x": [xs[i] for i in order],
                "y": [ys[i] for i in order],
                "lo": [los[i] for i in order],
                "hi": [his[i] for i in order],
                "label": label
            }

    return series


def plot_steps(series: Dict[str, Dict[str, Any]], out_path: Path, title: str, band: str) -> None:
    # W&B-like clean style (no seaborn dependency required; this is a matplotlib bundled style)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10.5, 4.2), dpi=130)

    # Sort keys to ensure stable legend order (optional but nice)
    sorted_keys = sorted(series.keys())

    for key in sorted_keys:
        s = series[key]
        x = np.array(s["x"], dtype=np.int32)
        y = np.array(s["y"], dtype=np.float32)
        lo = np.array(s["lo"], dtype=np.float32)
        hi = np.array(s["hi"], dtype=np.float32)
        label = s["label"]

        if x.size == 0:
            continue

        (line,) = ax.plot(
            x, y,
            linewidth=2.6,
            marker="o",
            markersize=4.5,
            label=label,
        )

        if band != "none":
            ax.fill_between(x, lo, hi, alpha=0.18)

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("n_actions (apply_first_n_actions)", fontsize=11)
    ax.set_ylabel("Env steps (mean over episodes)", fontsize=11)

    # Collect all unique x-ticks across all series
    all_xticks = set()
    for s in series.values():
        all_xticks.update(s["x"])
    
    if all_xticks:
        ax.set_xticks(sorted(all_xticks))
        
    ax.tick_params(axis="both", labelsize=10)

    # Clean spines like W&B
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Subtle grid
    ax.grid(True, which="major", alpha=0.28)
    ax.grid(True, which="minor", alpha=0.12)
    ax.minorticks_on()

    # Legend on top
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=3,
        frameon=False,
        fontsize=10,
        handlelength=2.2,
        columnspacing=1.2,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=str, nargs='+', required=True, 
                   help="Path(s) to combined_all_arches.json files. You can provide multiple files.")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for plots")
    p.add_argument("--title", type=str, default="Mean env steps vs n_actions", help="Plot title")
    p.add_argument("--band", type=str, default="ci95", choices=["none", "std", "sem", "ci95"],
                   help="Shaded band type computed over episodes for each point")
    args = p.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    out_dir = Path(args.out_dir)

    series = load_series(input_paths, band=args.band)

    if not series:
        print("No valid data found in inputs.")
        return

    out_png = out_dir / "steps_vs_n_actions.png"
    out_pdf = out_dir / "steps_vs_n_actions.pdf"
    plot_steps(series, out_png, args.title, band=args.band)
    plot_steps(series, out_pdf, args.title, band=args.band)

    print(f"[plot] wrote: {out_png}")
    print(f"[plot] wrote: {out_pdf}")


if __name__ == "__main__":
    main()