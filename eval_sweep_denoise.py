#!/usr/bin/env python3
"""
Sweep wrapper for run_diffusion_and_mppi.py

Produces:
  - Per-run folders (each contains vanilla_mppi_results.json + run.log + optional videos/)
  - Per-arch combined JSON
  - Global combined JSON (all arches)
Writes combined JSONs after every run so partial progress is preserved.
"""

from __future__ import annotations

import json
import sys
import time
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# =========================
# User-editable parameters
# =========================

# Choose sweep type:
#   "n_actions"     : sweep apply_first_n_actions in {3..30 step 3}, fixed diff_num_inference_steps
#   "denoise_steps" : sweep diff_num_inference_steps in {30..100 step 10}, fixed apply_first_n_actions
SWEEP_MODE = "denoise_steps" #"n_actions"  # <-- change to "denoise_steps" for the new test

# Choose which architectures to run:
#   "mppi_and_mlp" : only mppi_only + diffusion warmstart (mlp)
#   "all"          : mppi_only + diffusion warmstart (mlp/cnn/transformer)
ARCH_PROFILE = "all_but_mppi"  # <-- set to "all" when ready

# Path to eval script (put this wrapper next to run_diffusion_and_mppi.py to keep default)
EVAL_SCRIPT_PATH = (Path(__file__).resolve().parent / "run_diffusion_and_mppi.py")

# Output root
OUT_BASE_DIR = Path("eval_sweeps")

# Common eval params
EPISODES_PER_RUN = 80
BASE_SEED = 42
PLAN_ITERATION = 1

# Plan method for diffusion-based arches
DIFFUSION_PLAN_METHOD = "diffusion_and_one_MPPI_refine"  # or "mppi_warmstart_by_diffusion"

# Warmstart budget used by run_diffusion_and_mppi.py (you said you'll insert)
WARMSTART_TIME_LIMIT = 0.25  

# Checkpoints (you said you'll insert)
DIFF_CKPTS = {
    "mlp": "/home/user_229/Diffusion_MPPI_comp/checkpoints_dir/eval_saturday_morning/mlp_very_good_horizon_100/val_best_4.pt",
    "cnn": "/home/user_229/Diffusion_MPPI_comp/results_cnn_turbo/val_best_22.pt",
    "transformer": "/home/user_229/Diffusion_MPPI_comp/results_trans_large/val_best_14.pt",
}

# Sweep config: n_actions sweep
N_ACTIONS_VALUES = list(range(3, 31, 3))
FIXED_DIFF_INFERENCE_STEPS_FOR_N_ACTIONS_SWEEP = 30 # <-- TODO
VIDEO_N_ACTIONS = {3, 15, 30}  # save videos only for these n_actions

# Sweep config: denoise sweep
DENOISE_STEPS_VALUES = list(range(30, 101, 10))
FIXED_N_ACTIONS_FOR_DENOISE_SWEEP = 10  # <-- TODO: set your hardcoded n_actions


# =========================
# Internals
# =========================

@dataclass(frozen=True)
class ArchSpec:
    key: str                 # folder + curve name
    plan_method: str         # "mppi_only" / "mppi_warmstart_by_diffusion" / "diffusion_and_one_MPPI_refine"
    diff_arch: Optional[str] # None for mppi_only; else "mlp"/"cnn"/"transformer"


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def _run_and_tee(cmd: List[str], log_path: Path) -> None:
    """
    Runs cmd, tees merged stdout/stderr to both console and log_path.
    Raises RuntimeError on non-zero exit.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()

        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()

        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"Command failed (rc={rc}): {' '.join(cmd)}")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _get_arches() -> List[ArchSpec]:
    all_arches = [
        ArchSpec(key="mppi_only", plan_method="mppi_only", diff_arch=None),
        ArchSpec(key="warmstart_mlp", plan_method=DIFFUSION_PLAN_METHOD, diff_arch="mlp"),
        ArchSpec(key="warmstart_cnn", plan_method=DIFFUSION_PLAN_METHOD, diff_arch="cnn"),
        ArchSpec(key="warmstart_transformer", plan_method=DIFFUSION_PLAN_METHOD, diff_arch="transformer"),
    ]
    if ARCH_PROFILE == "mppi_and_mlp":
        return [a for a in all_arches if a.key in {"mppi_only", "warmstart_mlp"}]
    if ARCH_PROFILE == "all":
        return all_arches
    if ARCH_PROFILE == "all_but_mppi":
        return [a for a in all_arches if a.key != "mppi_only"]
    if ARCH_PROFILE == "cnn":
        return [a for a in all_arches if a.key  == "warmstart_cnn"]
    if ARCH_PROFILE == "mlp":
        return [a for a in all_arches if a.key  == "warmstart_mlp"]
    if ARCH_PROFILE == "transformer":
        return [a for a in all_arches if a.key  == "warmstart_transformer"]
    raise ValueError(f"Unknown ARCH_PROFILE={ARCH_PROFILE}")


def _sweep_values() -> tuple[str, List[int], Dict[str, Any]]:
    """
    Returns: (param_name, values, fixed_params_dict)
    """
    if SWEEP_MODE == "n_actions":
        return (
            "apply_first_n_actions",
            N_ACTIONS_VALUES,
            {"diff_num_inference_steps": FIXED_DIFF_INFERENCE_STEPS_FOR_N_ACTIONS_SWEEP},
        )
    if SWEEP_MODE == "denoise_steps":
        return (
            "diff_num_inference_steps",
            DENOISE_STEPS_VALUES,
            {"apply_first_n_actions": FIXED_N_ACTIONS_FOR_DENOISE_SWEEP},
        )
    raise ValueError(f"Unknown SWEEP_MODE={SWEEP_MODE}")


def _build_cmd(
    arch: ArchSpec,
    logs_dir: Path,
    sweep_param_name: str,
    sweep_value: int,
    fixed_params: Dict[str, Any],
) -> List[str]:
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT_PATH),
        "--episodes", str(EPISODES_PER_RUN),
        "--seed", str(BASE_SEED),
        "--plan_iteration", str(PLAN_ITERATION),
        "--logs_dir", str(logs_dir),
        "--plan_method", arch.plan_method,
        "--warmstart_time_limit", str(WARMSTART_TIME_LIMIT),
    ]

    # Set sweep and fixed params
    params = dict(fixed_params)
    params[sweep_param_name] = sweep_value

    # These args exist for all modes; harmless for mppi_only
    cmd += ["--apply_first_n_actions", str(params.get("apply_first_n_actions", 1))]
    cmd += ["--diff_num_inference_steps", str(params.get("diff_num_inference_steps", 100))]

    # Diffusion args only needed when diffusion is used
    if arch.diff_arch is not None:
        ckpt = DIFF_CKPTS.get(arch.diff_arch)
        if not ckpt or "ABS/PATH/TO" in ckpt:
            raise ValueError(f"Missing DIFF_CKPTS['{arch.diff_arch}'] (set it in the script).")
        cmd += ["--diff_arch", arch.diff_arch, "--diff_ckpt", ckpt]

    # Save videos only for the n_actions sweep at {3,15,30}
    if SWEEP_MODE == "n_actions" and params["apply_first_n_actions"] in VIDEO_N_ACTIONS:
        cmd += ["--save_video"]

    return cmd


def _init_combined(sweep_dir: Path, sweep_param_name: str, fixed_params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "eval_script": str(EVAL_SCRIPT_PATH),
            "sweep_dir": str(sweep_dir),
            "episodes_per_run": EPISODES_PER_RUN,
            "base_seed": BASE_SEED,
            "plan_iteration": PLAN_ITERATION,
            "warmstart_time_limit": WARMSTART_TIME_LIMIT,
        },
        "sweep": {
            "mode": SWEEP_MODE,
            "param_name": sweep_param_name,
            "fixed_params": fixed_params,
        },
        "architectures": {},  # arch_key -> {plan_method, diff_arch, diff_ckpt, runs_by_value}
    }


def _upsert_run(combined: Dict[str, Any], arch: ArchSpec, sweep_value: int, run_dir: Path, results: Dict[str, Any]) -> None:
    arch_block = combined["architectures"].setdefault(
        arch.key,
        {
            "plan_method": arch.plan_method,
            "diff_arch": arch.diff_arch,
            "diff_ckpt": (DIFF_CKPTS.get(arch.diff_arch) if arch.diff_arch else None),
            "runs_by_value": {},  # str(sweep_value) -> run_entry
        },
    )

    ep_steps = [int(e["steps"]) for e in results.get("episodes", [])]
    ep_success = [bool(e.get("success", False)) for e in results.get("episodes", [])]

    run_entry = {
        "sweep_value": int(sweep_value),
        "run_dir": str(run_dir),
        "results_json": str(run_dir / "vanilla_mppi_results.json"),
        "summary": results.get("summary", {}),
        "episode_steps": ep_steps,
        "episode_success": ep_success,
    }
    arch_block["runs_by_value"][str(sweep_value)] = run_entry


def main() -> None:
    arches = _get_arches()
    sweep_param_name, values, fixed_params = _sweep_values()

    sweep_dir = OUT_BASE_DIR / f"sweep_vanilla_mppi_{_now_tag()}_{SWEEP_MODE}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    sweep_log_path = sweep_dir / "sweep.log"
    with sweep_log_path.open("w", encoding="utf-8") as slog:

        def log(msg: str) -> None:
            print(msg)
            slog.write(msg + "\n")
            slog.flush()

        log(f"[sweep] mode={SWEEP_MODE} param={sweep_param_name} values={values}")
        log(f"[sweep] arches={[a.key for a in arches]}")
        log(f"[sweep] out_dir={sweep_dir}")
        log(f"[sweep] eval_script={EVAL_SCRIPT_PATH}")

        combined_all_path = sweep_dir / "combined_all_arches.json"
        combined_all = _init_combined(sweep_dir, sweep_param_name, fixed_params)

        for arch in arches:
            arch_dir = sweep_dir / arch.key
            arch_dir.mkdir(parents=True, exist_ok=True)

            combined_arch_path = arch_dir / f"combined_{arch.key}.json"
            combined_arch = _init_combined(arch_dir, sweep_param_name, fixed_params)

            log(f"\n[arch] {arch.key} (plan_method={arch.plan_method} diff_arch={arch.diff_arch})")

            for v in values:
                run_dir = arch_dir / f"{sweep_param_name}_{int(v):03d}"
                run_dir.mkdir(parents=True, exist_ok=True)

                results_path = run_dir / "vanilla_mppi_results.json"
                run_log_path = run_dir / "run.log"

                # Resume-friendly: if results exist, just load and update combined JSONs
                if results_path.exists():
                    log(f"[run] SKIP existing {arch.key} {sweep_param_name}={v} -> {results_path}")
                    results = _load_json(results_path)
                else:
                    cmd = _build_cmd(
                        arch=arch,
                        logs_dir=run_dir,
                        sweep_param_name=sweep_param_name,
                        sweep_value=int(v),
                        fixed_params=fixed_params,
                    )
                    log(f"[run] START {arch.key} {sweep_param_name}={v}")
                    t0 = time.time()
                    _run_and_tee(cmd, run_log_path)
                    dt = time.time() - t0
                    log(f"[run] DONE  {arch.key} {sweep_param_name}={v} wall={dt:.1f}s")
                    results = _load_json(results_path)

                # Update combined JSONs
                _upsert_run(combined_arch, arch, int(v), run_dir, results)
                _upsert_run(combined_all, arch, int(v), run_dir, results)

                # Persist progress after every run
                _atomic_write_json(combined_arch_path, combined_arch)
                _atomic_write_json(combined_all_path, combined_all)
                log(f"[json] updated {combined_arch_path.name} and {combined_all_path.name}")

        log("\n[sweep] completed")


if __name__ == "__main__":
    main()
