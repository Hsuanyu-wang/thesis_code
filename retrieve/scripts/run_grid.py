#!/usr/bin/env python3
import argparse
import itertools
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union


def _normalize_dropout(value: Union[str, float]) -> Optional[float]:
    if isinstance(value, str):
        if value.lower() in ("none", "no", "off", "0"):
            return None
        try:
            return float(value)
        except Exception:
            return None
    try:
        v = float(value)
        return None if v <= 0.0 else v
    except Exception:
        return None


def _make_command(python_exe: str, train_main: str, combo: Dict[str, object]) -> List[str]:
    cmd: List[str] = [python_exe, train_main, "-d", combo["dataset"]]

    # Dropout
    dr = combo.get("dropout_rate", None)
    if dr is not None:
        cmd += ["-dp", "-dpr", str(dr)]

    # Supervision signal on graph (target triples)
    supervision = combo.get("supervision", "none")
    if supervision == "freq_weight":
        cmd += ["-fw"]
    elif supervision == "freq_weight_inv":
        cmd += ["-fwi"]
    elif supervision in ("kge_score", "kge_score_inv"):
        # Use shortest path KGE scoring with model-specific weight mode
        cmd += ["-ksp"]

    # BCE training loss
    bce_mode = combo.get("bce_mode", "none")
    if bce_mode == "spcount":
        cmd += ["-sc"]
    elif bce_mode == "spcount_inv":
        cmd += ["-sci"]
    # smoothing
    sc_smooth = combo.get("spcount_smooth", "none")
    if bce_mode in ("spcount", "spcount_inv") and sc_smooth in ("none", "log", "sqrt"):
        cmd += ["--spcount_smooth", sc_smooth]

    # If supervision needs KGE, include model + weight mode
    if supervision in ("kge_score", "kge_score_inv"):
        cmd += ["-km", combo.get("kge_model", "transe")]
        cmd += ["-kwm", combo.get("kge_weight_mode", "score")]  # score or score_inv

    # Early stop validation mode
    esv = combo.get("early_stop_val", "none")
    if esv in ("and", "or"):
        cmd += ["-esv", esv]

    # Experiment tag (mark grid runs)
    exp_tag = combo.get("exp_tag", None)
    if exp_tag:
        cmd += ["--exp_tag", str(exp_tag)]

    return cmd


def _label_from_combo(idx: int, total: int, combo: Dict[str, object]) -> str:
    ds = combo.get("dataset", "-")
    km = combo.get("kge_model", "none")
    sup = combo.get("supervision", "none")
    bce = combo.get("bce_mode", "none")
    smooth = combo.get("spcount_smooth", "none")
    dr = combo.get("dropout_rate", None)
    drs = f"{dr:.2f}" if isinstance(dr, float) else ("none" if dr is None else str(dr))
    esv = combo.get("early_stop_val", "none")
    parts = [
        f"{idx}/{total}",
        f"d={ds}",
        f"kge={km}",
        f"sup={sup}",
        f"bce={bce}{'' if smooth=='none' else ':'+smooth}",
        f"drop={drs}",
        f"esv={esv}",
    ]
    return " | ".join(parts)


def _combos(args: argparse.Namespace) -> List[Dict[str, object]]:
    datasets = args.datasets

    # Dropout options: list of strings/floats including 'none'
    dropout_rates: List[Optional[float]] = []
    for d in args.dropouts:
        dr = _normalize_dropout(d)
        dropout_rates.append(dr)
    if not dropout_rates:
        dropout_rates = [None]

    kge_models = args.kge_models  # includes 'none'
    supervision_signals = args.supervision
    bce_modes = args.bce_losses
    sp_smooth_options = args.spcount_smooth
    esv_modes = args.early_stop

    grid: List[Dict[str, object]] = []
    for (
        dataset,
        dr,
        kge_model,
        supervision,
        bce_mode,
        sp_smooth,
        esv,
    ) in itertools.product(
        datasets,
        dropout_rates,
        kge_models,
        supervision_signals,
        bce_modes,
        sp_smooth_options,
        esv_modes,
    ):
        # Validate early stop
        esv_norm = esv
        if esv_norm in ("esv_and", "and"):
            esv_norm = "and"
        elif esv_norm in ("esv_or", "esc_or", "or"):
            esv_norm = "or"
        elif esv_norm in ("none", "no", "off"):
            esv_norm = "none"
        else:
            continue

        # Supervision-KGE constraints
        if supervision in ("kge_score", "kge_score_inv"):
            if kge_model == "none":
                continue
            if supervision == "kge_score" and kge_model not in ("distmult", "complex"):
                continue
            if supervision == "kge_score_inv" and kge_model not in ("transe", "rotate"):
                continue
        else:
            # No KGE supervision required → ignore any non-default KGE model "noise"
            if kge_model != "none":
                # keep also variants where KGE is not used if user explicitly asked for model none combinations separately
                pass

        # BCE modes supported
        if bce_mode not in ("none", "spcount", "spcount_inv"):
            continue

        # spcount_smooth only applies when using spcount variants
        if bce_mode == "none" and sp_smooth != "none":
            # normalize by dropping smoothing when not applicable
            sp_smooth = "none"
        if sp_smooth not in ("none", "log", "sqrt"):
            continue

        combo = {
            "dataset": dataset,
            "dropout_rate": dr,
            "kge_model": kge_model,
            "supervision": supervision,
            "bce_mode": bce_mode,
            "spcount_smooth": sp_smooth,
            "early_stop_val": esv_norm,
        }

        # KGE weight mode for supervision
        if supervision == "kge_score":
            combo["kge_weight_mode"] = "score"
        elif supervision == "kge_score_inv":
            combo["kge_weight_mode"] = "score_inv"

        grid.append(combo)

    # Trim to limit if requested
    if args.limit and args.limit > 0:
        grid = grid[: args.limit]
    return grid


def _run_command(cmd: List[str], env: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, check=False, text=True)
        return proc.returncode, proc.stdout
    except Exception as e:
        return 1, f"Execution failed: {e}"


def _stream_command(cmd: List[str], env: Optional[Dict[str, str]] = None) -> int:
    """Run command and stream stdout line-by-line (for sequential mode)."""
    try:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, bufsize=1) as p:
            assert p.stdout is not None
            for line in p.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
            return p.wait()
    except Exception as e:
        sys.stdout.write(f"Execution failed: {e}\n")
        sys.stdout.flush()
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full grid for train_main.py with conflict-safe combinations")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use")
    parser.add_argument("--train_script", default="/home/YX_thesis/retrieve/train_main.py", help="Absolute path to train_main.py")
    parser.add_argument("--datasets", nargs="+", default=["webqsp"], choices=["webqsp", "cwq"], help="Datasets to run")

    # Dropout grid (use values or 'none')
    parser.add_argument("--dropouts", nargs="*", default=["none", 0.1, 0.2, 0.5, 0.7], help="Dropout values: use 'none' or float rates")

    # KGE model grid
    parser.add_argument("--kge_models", nargs="*", default=["none", "transe", "rotate", "distmult", "complex"], choices=["none", "transe", "rotate", "distmult", "complex"], help="KGE model to use for supervision")

    # Supervision signal grid
    parser.add_argument(
        "--supervision",
        nargs="*",
        default=["none", "freq_weight", "freq_weight_inv", "kge_score", "kge_score_inv"],
        choices=["none", "freq_weight", "freq_weight_inv", "kge_score", "kge_score_inv"],
        help="How to build supervision/targets on the graph",
    )

    # BCE loss grid
    parser.add_argument("--bce_losses", nargs="*", default=["none", "spcount", "spcount_inv"], choices=["none", "spcount", "spcount_inv"], help="BCE loss modes")
    parser.add_argument("--spcount_smooth", nargs="*", default=["none", "log", "sqrt"], choices=["none", "log", "sqrt"], help="Smoothing for spcount/spcount_inv")

    # Early stopping grid (accept aliases)
    parser.add_argument("--early_stop", nargs="*", default=["none", "esv_and", "esv_or"], choices=["none", "esv_and", "esv_or", "esc_or"], help="Additional early stop constraint")

    # Execution
    parser.add_argument("--limit", type=int, default=0, help="Optional limit to first N combinations")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--dry_run", action="store_true", help="Only print commands without executing")
    parser.add_argument("--env", nargs="*", default=[], help="Environment variables KEY=VALUE to pass")

    args = parser.parse_args()

    # Validate absolute path expectation
    train_script = args.train_script
    if not os.path.isabs(train_script):
        # Normalize to absolute for robustness
        train_script = os.path.abspath(train_script)

    combos = _combos(args)
    if not combos:
        print("No valid combinations produced after conflict filtering.")
        return 1

    env = os.environ.copy()
    for kv in args.env:
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k] = v

    # Build commands with labels
    commands: List[Tuple[str, List[str]]] = []
    total = len(combos)
    for i, combo in enumerate(combos, 1):
        combo["exp_tag"] = os.environ.get("GRID_EXP_TAG", "grid")
        label = _label_from_combo(i, total, combo)
        cmd = _make_command(args.python, train_script, combo)
        commands.append((label, cmd))

    # Print summary only in dry-run
    if args.dry_run:
        print(f"Total runs: {len(commands)} (parallel={args.parallel})")
        for i, (label, c) in enumerate(commands, 1):
            print(f"[{i}] {label}")
            print(f"    {shlex.join(c)}")

    if args.dry_run:
        return 0

    # Execute
    rc = 0
    if args.parallel <= 1:
        for label, c in commands:
            print(f"===== START {label} =====")
            code, _ = _run_command(c, env)  # run quietly
            status = "DONE" if code == 0 else f"FAIL({code})"
            print(f"===== {status} {label} =====\n")
            if code != 0:
                rc = code
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futures: Dict[object, str] = {}
            for label, c in commands:
                print(f"===== START {label} =====")
                futures[ex.submit(_run_command, c, env)] = label
            for fut in as_completed(futures):
                label = futures[fut]
                code, _ = fut.result()
                status = "DONE" if code == 0 else f"FAIL({code})"
                print(f"===== {status} {label} =====")
                if code != 0:
                    rc = code

    return rc


if __name__ == "__main__":
    raise SystemExit(main())


