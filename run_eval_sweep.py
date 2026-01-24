#!/usr/bin/env python3
"""
Sweep eval_vanilla_mppi.py over apply_first_n_actions and (selected) architectures,
and write combined JSON summaries.

- n_actions in {3,6,...,30}
- videos only for n_actions in {3,15,30} (all episodes of those runs will be recorded)
- per-run logs_dir: <out>/<arch>/n_actions_XX/
- per-arch combined JSON: <out>/<arch>/combined_arch.json
- global combined JSON: <out>/combined_all_arches.json
- progress is saved after every run

Logging:
- Writes a top-level sweep log: <out>/sweep.log
- Also writes per-run console output: <run_dir>/console.log
- Everything is still printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# -----------------------------
# USER-EDITABLE PARAMETERS
# -----------------------------

RUN_MODE = "mlp"  # or "all"

DIFF_CKPTS = {
    "mlp": "/home/user_229/Diffusion_MPPI_comp/checkpoints_dir/eval_saturday_morning/mlp_very_good_horizon_100/val_best_4.pt",
    "cnn": "/home/user_229/Diffusion_MPPI_comp/results_cnn_turbo/val_best_22.pt",
    "transformer": "/home/user_229/Diffusion_MPPI_comp/checkpoints_dir/eval_saturday_morning/transformer_bad/val_best_4.pt",
}

WARMSTART_TIME_LIMIT = 0.21
DIFF_NUM_INFERENCE_STEPS = 30

N_ACTIONS_LIST = list(range(6, 31, 3))
VIDEO_N_ACTIONS = {6, 15, 30}

EPISODES = 80


# -----------------------------
# INTERNALS
# -----------------------------

@dataclass(frozen=True)
class Experiment:
    name: str
    plan_method: str
    diff_arch: Optional[str]


class TeeLogger:
    """Writes messages to stdout and to a file (line-based)."""
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("a", encoding="utf-8")

    def log(self, msg: str) -> None:
        print(msg, flush=True)
        self._f.write(msg + "\n")
        self._f.flush()

    def write_raw(self, text: str) -> None:
        """For streaming subprocess output lines verbatim."""
        sys.stdout.write(text)
        sys.stdout.flush()
        self._f.write(text)
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_eval_script(eval_script: str) -> Path:
    p = Path(eval_script)
    if p.exists():
        return p.resolve()
    here = Path(__file__).resolve().parent
    alt = here / eval_script
    if alt.exists():
        return alt.resolve()
    return Path(eval_script).resolve()


def _require_ckpt(arch: str) -> str:
    ckpt = DIFF_CKPTS.get(arch, "")
    if not ckpt or ckpt.startswith("/path/to/"):
        raise ValueError(f"DIFF_CKPTS['{arch}'] is not set. Insert the checkpoint path.")
    return ckpt


def _build_cmd(
    eval_script: Path,
    logs_dir: Path,
    exp: Experiment,
    n_actions: int,
    episodes: int,
    warmstart_time_limit: float,
) -> List[str]:
    cmd = [
        sys.executable,
        str(eval_script),
        "--episodes", str(episodes),
        "--logs_dir", str(logs_dir),
        "--plan_method", exp.plan_method,
        "--warmstart_time_limit", str(warmstart_time_limit),
        "--apply_first_n_actions", str(n_actions),
    ]

    if exp.plan_method == "mppi_warmstart_by_diffusion":
        assert exp.diff_arch in ("mlp", "cnn", "transformer")
        ckpt = _require_ckpt(exp.diff_arch)
        cmd += [
            "--diff_arch", exp.diff_arch,
            "--diff_ckpt", ckpt,
            "--diff_num_inference_steps", str(DIFF_NUM_INFERENCE_STEPS),
        ]

    if n_actions in VIDEO_N_ACTIONS:
        cmd += ["--save_video"]

    return cmd


def _select_experiments() -> List[Experiment]:
    all_experiments = [
        Experiment(name="mppi_only", plan_method="mppi_only", diff_arch=None),
        Experiment(name="warmstart_diffusion_cnn", plan_method="mppi_warmstart_by_diffusion", diff_arch="cnn"),
        Experiment(name="warmstart_diffusion_transformer", plan_method="mppi_warmstart_by_diffusion", diff_arch="transformer"),
        Experiment(name="warmstart_diffusion_mlp", plan_method="mppi_warmstart_by_diffusion", diff_arch="mlp"),
    ]
    if RUN_MODE == "mppi_and_mlp":
        return [e for e in all_experiments if e.name in {"mppi_only", "warmstart_diffusion_mlp"}]
    if RUN_MODE == "all":
        return all_experiments
    if RUN_MODE == "cnn":
        return [e for e in all_experiments if e.name in {"warmstart_diffusion_cnn"}]
    if RUN_MODE == "mlp":
        return [e for e in all_experiments if e.name in {"warmstart_diffusion_mlp"}]
    raise ValueError(f"Unknown RUN_MODE='{RUN_MODE}'. Use 'mppi_and_mlp' or 'all'.")


def _run_cmd_tee(cmd: List[str], run_log_path: Path, logger: TeeLogger) -> None:
    """
    Run a command and tee its combined stdout/stderr to:
    - stdout
    - sweep.log (via logger)
    - per-run console.log (run_log_path)
    """
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    with run_log_path.open("w", encoding="utf-8") as rf:
        rf.write("COMMAND:\n" + " ".join(cmd) + "\n\n")
        rf.flush()

        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None

        for line in p.stdout:
            # Tee verbatim line to stdout + sweep.log + run console.log
            logger.write_raw(line)
            rf.write(line)
            rf.flush()

        rc = p.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_script", type=str, default="eval_vanilla_mppi.py")
    ap.add_argument("--out_root", type=str, default="eval_sweeps")
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--warmstart_time_limit", type=float, default=WARMSTART_TIME_LIMIT)
    args = ap.parse_args()

    eval_script = _resolve_eval_script(args.eval_script)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"sweep_vanilla_mppi_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = TeeLogger(out_dir / "sweep.log")
    logger.log(f"[sweep] out_dir = {out_dir}")
    logger.log(f"[sweep] sweep.log = {out_dir / 'sweep.log'}")
    logger.log(f"[sweep] eval_script = {eval_script}")
    logger.log(f"[sweep] RUN_MODE = {RUN_MODE}")
    logger.log(f"[sweep] episodes = {args.episodes}, warmstart_time_limit = {args.warmstart_time_limit}")
    logger.log(f"[sweep] n_actions_list = {N_ACTIONS_LIST}, video_n_actions = {sorted(VIDEO_N_ACTIONS)}")

    experiments = _select_experiments()

    combined_all: Dict[str, Any] = {
        "created_at": ts,
        "eval_script": str(eval_script),
        "run_mode": RUN_MODE,
        "sweep": {
            "n_actions_list": N_ACTIONS_LIST,
            "video_n_actions": sorted(VIDEO_N_ACTIONS),
            "episodes": args.episodes,
            "warmstart_time_limit": args.warmstart_time_limit,
            "diff_num_inference_steps": DIFF_NUM_INFERENCE_STEPS,
        },
        "experiments": {},
    }
    combined_all_path = out_dir / "combined_all_arches.json"

    try:
        for exp in experiments:
            logger.log(f"\n[arch] START {exp.name}")
            exp_dir = out_dir / exp.name
            exp_dir.mkdir(parents=True, exist_ok=True)

            arch_combined_path = exp_dir / "combined_arch.json"
            if arch_combined_path.exists():
                arch_combined = _load_json(arch_combined_path)
            else:
                arch_combined = {
                    "created_at": ts,
                    "experiment": asdict(exp),
                    "diff_ckpt": (DIFF_CKPTS.get(exp.diff_arch) if exp.diff_arch else None),
                    "runs": [],
                }

            done_ns = set()
            for r in arch_combined.get("runs", []):
                try:
                    done_ns.add(int(r.get("n_actions")))
                except Exception:
                    pass

            for n_actions in N_ACTIONS_LIST:
                run_dir = exp_dir / f"n_actions_{n_actions:02d}"
                run_dir.mkdir(parents=True, exist_ok=True)
                run_json_path = run_dir / "vanilla_mppi_results.json"
                run_console_log = run_dir / "console.log"

                if run_json_path.exists() and n_actions not in done_ns:
                    logger.log(f"[run] ingest existing: {exp.name} n_actions={n_actions}")
                    run_data = _load_json(run_json_path)
                    arch_combined["runs"].append({
                        "n_actions": n_actions,
                        "logs_dir": str(run_dir),
                        "status": "ok",
                        "result_path": str(run_json_path),
                        "summary": run_data.get("summary", {}),
                        "results": run_data,
                    })
                    done_ns.add(n_actions)

                elif not run_json_path.exists():
                    logger.log(f"[run] execute: {exp.name} n_actions={n_actions} (video={'yes' if n_actions in VIDEO_N_ACTIONS else 'no'})")
                    cmd = _build_cmd(
                        eval_script=eval_script,
                        logs_dir=run_dir,
                        exp=exp,
                        n_actions=n_actions,
                        episodes=args.episodes,
                        warmstart_time_limit=args.warmstart_time_limit,
                    )
                    logger.log("[cmd] " + " ".join(cmd))

                    try:
                        _run_cmd_tee(cmd, run_console_log, logger)

                        run_data = _load_json(run_json_path)
                        arch_combined["runs"].append({
                            "n_actions": n_actions,
                            "logs_dir": str(run_dir),
                            "status": "ok",
                            "result_path": str(run_json_path),
                            "summary": run_data.get("summary", {}),
                            "results": run_data,
                        })
                        done_ns.add(n_actions)
                        logger.log(f"[run] ok: {exp.name} n_actions={n_actions}")

                    except Exception as e:
                        arch_combined["runs"].append({
                            "n_actions": n_actions,
                            "logs_dir": str(run_dir),
                            "status": "failed",
                            "error": repr(e),
                            "result_path": str(run_json_path),
                        })
                        logger.log(f"[run] FAILED: {exp.name} n_actions={n_actions} error={repr(e)}")

                # Persist after each run
                _atomic_write_json(arch_combined_path, arch_combined)
                combined_all["experiments"][exp.name] = arch_combined
                _atomic_write_json(combined_all_path, combined_all)

            logger.log(f"[arch] DONE {exp.name}")

    finally:
        logger.log(f"\n[sweep] DONE. Global combined JSON: {combined_all_path}")
        logger.close()


if __name__ == "__main__":
    main()
