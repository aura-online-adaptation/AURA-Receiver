import os
import argparse
import copy
import itertools
import torch

from experiments.framework.read_config import load_config
from experiments.framework.projection_learning import train_projection
from experiments.framework.single_run import run_experiment, save_results


# Parameters that define the learned projection/artifact compatibility.
# These must be copied from the offline/projection config into the online config.
PROJECTION_CRITICAL_PATHS = [
    "model.type",
    "channel.num_users",
    "channel.num_antennas",
    "channel.snr",
    "channel.linear_channel",
    "projection.proj_type",
    "projection.num_wanted",
    "model.OU",
    "model.F",
    "model.F_style",
    "algorithm.covariance_type",
]


def get_nested(cfg, path, default=None):
    keys = path.split(".")
    val = cfg
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return default
    return val


def set_nested(cfg, path, value):
    keys = path.split(".")
    d = cfg
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def has_sweep(config):
    return isinstance(config, dict) and "base_config" in config and bool(config.get("sweep", {}))


def resolve_online_mode(online_config, requested_mode):
    if requested_mode != "auto":
        return requested_mode
    return "multi" if has_sweep(online_config) else "single"


def materialize_offline_config(offline_config):
    """
    Build the actual single offline config that will be used for projection learning.

    Supported:
    1. Plain offline config.
    2. base_config without sweep.
    3. base_config with a sweep that contains exactly one combination.

    Not supported here:
    Offline sweep with multiple combinations. That should be handled later as a
    paired offline-online sweep, not silently collapsed.
    """
    if "base_config" not in offline_config:
        return copy.deepcopy(offline_config), []

    offline_base = copy.deepcopy(offline_config["base_config"])
    sweep = offline_config.get("sweep", {})

    if not sweep:
        return offline_base, []

    keys = list(sweep.keys())
    values = list(sweep.values())
    combos = list(itertools.product(*values))

    if len(combos) != 1:
        raise ValueError(
            "offline_config contains a sweep with multiple combinations. "
            "This runner currently supports one offline configuration followed by single/multi online. "
            "Reduce the offline sweep to one value per key, or use a future paired-sweep runner."
        )

    combo = combos[0]
    applied = []
    for k, v in zip(keys, combo):
        set_nested(offline_base, k, copy.deepcopy(v))
        applied.append(f"{k}={v}")

    return offline_base, applied


def sync_online_base_from_offline(offline_base, online_base):
    """
    Offline/projection config is the source of truth.

    This function updates online_base in-place for all parameters that must match
    the learned artifact. It returns a list of human-readable changes.
    """
    changes = []

    for path in PROJECTION_CRITICAL_PATHS:
        off_val = get_nested(offline_base, path)
        on_val = get_nested(online_base, path)

        if off_val is None:
            continue

        if on_val != off_val:
            changes.append(f"{path}: online={on_val} -> offline={off_val}")
            set_nested(online_base, path, copy.deepcopy(off_val))

    # Compatibility between old plotting/config conventions:
    # sqz_factor is not used to build the model, but many plots/grouping functions use it.
    off_num_wanted = get_nested(offline_base, "projection.num_wanted")
    if off_num_wanted is not None:
        on_sqz = get_nested(online_base, "projection.sqz_factor")
        if on_sqz != off_num_wanted:
            changes.append(f"projection.sqz_factor: online={on_sqz} -> offline projection.num_wanted={off_num_wanted}")
            set_nested(online_base, "projection.sqz_factor", copy.deepcopy(off_num_wanted))

    return changes


def remove_projection_critical_sweep_keys(online_config, offline_base):
    """
    In offline -> multi mode, the online sweep must not override projection-critical
    parameters learned by the offline stage.
    """
    if not has_sweep(online_config):
        return []

    sweep = online_config.get("sweep", {})
    removed = []

    protected = set(PROJECTION_CRITICAL_PATHS + ["projection.sqz_factor"])

    for key in list(sweep.keys()):
        if key in protected:
            old_values = sweep.pop(key)
            if key == "projection.sqz_factor":
                replacement = get_nested(offline_base, "projection.num_wanted")
            else:
                replacement = get_nested(offline_base, key)
            removed.append(f"{key}: removed online sweep values={old_values}; using offline value={replacement}")

    return removed


def main():
    parser = argparse.ArgumentParser(description="End-to-End Runner: Offline Learning -> Online Evaluation")
    parser.add_argument("--offline_config", type=str, required=True, help="Path to offline projection learning config JSON")
    parser.add_argument("--online_config", type=str, required=True, help="Path to online evaluation config JSON")
    parser.add_argument("--output_dir", type=str, default="experiments/data", help="Base output directory for final results")

    # Kept for backwards compatibility. With the new behavior, offline always adapts online.
    parser.add_argument(
        "--allow_mismatch",
        action="store_true",
        help="Backward-compatible flag. Mismatches are now resolved by adapting online to offline.",
    )

    parser.add_argument(
        "--online_mode",
        type=str,
        choices=["auto", "single", "multi"],
        default="auto",
        help="Use single online run, multi online sweep, or auto-detect from online_config.",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=7,
        help="Number of repeated runs per valid combo when online_mode=multi.",
    )
    parser.add_argument(
        "--run_start",
        type=int,
        default=9,
        help="First run index used by save_results when online_mode=multi.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="For online_mode=multi: print combinations after offline learning without running online experiments.",
    )

    args = parser.parse_args()

    print("[1/5] Loading Configurations...")
    offline_config = load_config(args.offline_config)
    online_config = load_config(args.online_config)

    offline_base, applied_offline_sweep = materialize_offline_config(offline_config)
    if applied_offline_sweep:
        print("[Offline] Applied single-combo offline sweep:")
        for item in applied_offline_sweep:
            print(f"  - {item}")

    online_base = online_config.get("base_config", online_config)

    online_mode = resolve_online_mode(online_config, args.online_mode)
    print(f"[Mode] online_mode={online_mode}")

    print("[2/5] Adapting Online Config to Offline/Projection Config...")

    changes = sync_online_base_from_offline(offline_base, online_base)
    if changes:
        print("\n[Sync] Online base config was adapted to the offline/projection config:")
        for c in changes:
            print(f"  - {c}")
    else:
        print("[Sync] Online base config already matches the offline/projection config.")

    if online_mode == "multi":
        removed_sweep_keys = remove_projection_critical_sweep_keys(online_config, offline_base)
        if removed_sweep_keys:
            print("\n[Sync] Removed projection-critical keys from online sweep:")
            for r in removed_sweep_keys:
                print(f"  - {r}")

    # Write synced base back into online_config wrapper if needed.
    if "base_config" in online_config:
        online_config["base_config"] = online_base
    else:
        online_config = {"base_config": online_base, "sweep": {}}

    print("\n[3/5] Running Offline Projection Learning...")
    artifact_dir = train_projection(offline_base)
    torch.autograd.set_detect_anomaly(False)
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)

    if not artifact_dir or not os.path.exists(artifact_dir):
        raise RuntimeError(f"Offline learning failed to return a valid artifact directory: {artifact_dir}")

    print(f"\n[4/5] Injecting Artifact Directory into Online Config...")
    print(f"Artifact Dir: {artifact_dir}")

    if online_mode == "single":
        online_base["model"]["learned matrix"] = True
        online_base["projection"]["load_dir"] = artifact_dir

        if "experiment" not in online_base:
            online_base["experiment"] = {}
        online_base["experiment"]["allow_mismatch"] = True

        print("\n[5/5] Running Online Evaluation...")
        results = run_experiment(online_base)

        print("\n[+] Saving Final Results...")
        save_dir = save_results(config=online_base, results=results, base_dir=args.output_dir)
        print(f"End-to-End Pipeline Complete. Everything is saved in: {save_dir}")

    elif online_mode == "multi":
        print("\n[5/5] Running Online Multi Evaluation...")

        from experiments.framework.multi_runner import run_multi_sweep

        online_base.setdefault("experiment", {})
        online_base["experiment"]["allow_mismatch"] = True
        online_config["base_config"] = online_base

        summary = run_multi_sweep(
            cfg_file=online_config,
            output_dir=args.output_dir,
            num_runs=args.num_runs,
            run_start=args.run_start,
            artifact_dir=artifact_dir,
            dry_run=args.dry_run,
        )
        print(f"End-to-End Multi Pipeline Complete. Summary: {summary}")

    else:
        raise ValueError(f"Unsupported online_mode: {online_mode}")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    main()
