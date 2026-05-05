import os

"""os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
"""
import torch
"""torch.set_num_threads(1)
torch.set_num_interop_threads(1)"""

import json
import copy
import itertools
import ctypes
import argparse


try:
    from experiments.framework.single_run import run_experiment, save_results
except ImportError:
    # Keeps backward compatibility when running this file directly from experiments/framework.
    from single_run import run_experiment, save_results


torch.set_default_dtype(torch.float32)


def pin_process_to_cores(core_ids):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    HANDLE = ctypes.c_void_p
    DWORD_PTR = ctypes.c_size_t
    BOOL = ctypes.c_int

    kernel32.GetCurrentProcess.restype = HANDLE
    kernel32.GetCurrentProcess.argtypes = []

    kernel32.GetProcessAffinityMask.restype = BOOL
    kernel32.GetProcessAffinityMask.argtypes = [
        HANDLE,
        ctypes.POINTER(DWORD_PTR),
        ctypes.POINTER(DWORD_PTR),
    ]

    kernel32.SetProcessAffinityMask.restype = BOOL
    kernel32.SetProcessAffinityMask.argtypes = [HANDLE, DWORD_PTR]

    process_handle = kernel32.GetCurrentProcess()

    process_mask = DWORD_PTR()
    system_mask = DWORD_PTR()

    ok = kernel32.GetProcessAffinityMask(
        process_handle,
        ctypes.byref(process_mask),
        ctypes.byref(system_mask),
    )
    if not ok:
        err = ctypes.get_last_error()
        raise OSError(f"GetProcessAffinityMask failed, WinError={err}")

    requested_mask = 0
    for core_id in core_ids:
        requested_mask |= (1 << core_id)

    if requested_mask & process_mask.value != requested_mask:
        raise ValueError(
            f"Requested cores {core_ids} are not allowed. "
            f"Allowed mask=0x{process_mask.value:x}, requested=0x{requested_mask:x}"
        )

    ok = kernel32.SetProcessAffinityMask(process_handle, DWORD_PTR(requested_mask))
    if not ok:
        err = ctypes.get_last_error()
        raise OSError(
            f"SetProcessAffinityMask failed, WinError={err}, "
            f"allowed=0x{process_mask.value:x}, requested=0x{requested_mask:x}"
        )

    print(
        f"[Affinity] pinned to cores {core_ids} | "
        f"allowed mask=0x{process_mask.value:x} | requested=0x{requested_mask:x}"
    )


def set_nested(config, key, value):
    """Updates nested config values from keys like 'a.b.c'."""
    keys = key.split(".")
    d = config
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def is_valid_combo(cfg):
    model = cfg['model']['type'].lower()
    method = cfg['algorithm']['method'].lower()
    cov = cfg['algorithm']['covariance_type'].lower()
    pulse = cfg['model']['Pulse']
    lm = cfg['model']['learned matrix']
    F = cfg['model']["F"]
    OU = cfg['model']["OU"]

    if method == "sgd":
        return (not pulse) and (not lm) and model == "deepsic" and cov == "full"

    if model == "resnet":
        if cov == "diag":
            return False
        if cov == "full" and (not pulse) and (not lm):
            return True
        if cov == "full" and pulse and lm:
            return True
        return False

    if model == "deepsic":
        """if cov == "full":
            return pulse == lm"""
        if cov in ["diag", "dlr"]:
            return (not pulse) and (not lm)
        """return False"""

    if F and OU or not F and not OU:
        return False

    return True


def update_projection_paths(cfg):
    save_dir = cfg['projection'].get('save_dir', 'experiments/data/Learned_matrices')
    frames = cfg['projection']['frames']
    symbols = cfg['projection']['symbols per frame']
    covtype = cfg['algorithm']['covariance_type']
    snr = cfg['channel']['snr']
    model = cfg['model']['type'].lower()

    pred_type = "F" if cfg['model']['F'] else "OU"
    if pred_type == "F" and cfg.get('model', {}).get('F_style', 'full') == "diag":
        pred_type = "F_diag"

    proj_type = cfg['projection'].get('proj_type', 'DNN').lower() if model == 'resnet' else 'affine'

    linear_str = "linear" if cfg['channel'].get('linear_channel', True) else "nonlinear"

    run_name = f"{model}_{snr}snr_{proj_type}_{frames}x{symbols}_wanted{cfg['projection']['num_wanted']}_{pred_type}_cov{covtype}_{linear_str}"
    cfg['projection']['load_dir'] = os.path.join(save_dir, run_name)


def load_multi_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def iter_sweep_configs(cfg_file):
    """
    Yields (cfg, combo_dict) for every sweep combination.

    This keeps the original sweep behavior:
    cfg_file["base_config"] is copied, then every key in cfg_file["sweep"]
    is applied using dot-separated nested keys.
    """
    base = cfg_file["base_config"]
    sweep = cfg_file.get("sweep", {})

    if not sweep:
        yield copy.deepcopy(base), {}
        return

    keys = list(sweep.keys())
    values = list(sweep.values())

    for combo in itertools.product(*values):
        cfg = copy.deepcopy(base)
        combo_dict = {k: v for k, v in zip(keys, combo)}

        for k, v in combo_dict.items():
            set_nested(cfg, k, v)

        yield cfg, combo_dict


def apply_external_artifact(cfg, artifact_dir):
    """
    Used by run_offline_then_online.py.

    When artifact_dir is provided, the online sweep should use the matrices
    just learned offline instead of guessing load_dir from the naming convention.
    """
    if artifact_dir is None:
        return cfg

    if 'model' not in cfg:
        cfg['model'] = {}
    if 'projection' not in cfg:
        cfg['projection'] = {}

    cfg['model']['learned matrix'] = True
    cfg['projection']['load_dir'] = artifact_dir
    return cfg


def run_multi_sweep(
    cfg_file=None,
    config_path=None,
    output_dir="experiments/data/results",
    num_runs=7,
    run_start=9,
    artifact_dir=None,
    dry_run=False,
):
    """
    Run an online multi sweep.

    Two supported modes:
    1. Standalone:
       run_multi_sweep(config_path="experiments/framework/sweep_config.json", ...)

    2. Integrated with offline:
       artifact_dir is supplied by run_offline_then_online.py, and every online
       config in the sweep uses this artifact as projection.load_dir.
    """
    if cfg_file is None:
        if config_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, "sweep_config.json")
        cfg_file = load_multi_config(config_path)

    os.makedirs(output_dir, exist_ok=True)

    completed = 0
    skipped = 0

    print(f"[Multi] Output directory: {output_dir}")
    if config_path is not None:
        print(f"[Multi] Config path: {config_path}")
    if artifact_dir is not None:
        print(f"[Multi] Using external artifact_dir: {artifact_dir}")

    for cfg, combo_dict in iter_sweep_configs(cfg_file):
        if not is_valid_combo(cfg):
            skipped += 1
            print("Skipping:", combo_dict)
            continue

        cfg = apply_external_artifact(cfg, artifact_dir)

        # Backward-compatible behavior:
        # If one_true_path is false and no external artifact was injected,
        # build projection.load_dir from the standard learned-matrices naming convention.
        if artifact_dir is None and cfg.get('projection', {}).get('one_true_path', True) is False:
            update_projection_paths(cfg)

        print("\n[Multi] Running combo:", combo_dict)
        print("[Multi] projection.load_dir:", cfg.get("projection", {}).get("load_dir", "none"))

        if dry_run:
            completed += 1
            continue

        for i in range(num_runs):
            result = run_experiment(cfg)
            save_idx = run_start + i
            save = save_results(config=cfg, results=result, base_dir=output_dir, i=save_idx)
            print(f"Completed run. Results saved to: {save}")

        completed += 1

    print(f"\n[Multi] Done. Valid combos: {completed}, skipped combos: {skipped}")
    return {"completed": completed, "skipped": skipped, "output_dir": output_dir}


def main():
    """
    Standalone multi runner.

    Backward-compatible default:
    if --config_path is not provided, it uses sweep_config.json next to this file.
    """
    parser = argparse.ArgumentParser(description="Run an online multi sweep from JSON config")
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to sweep config JSON. Default: sweep_config.json next to multi_runner.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/data/results",
        help="Base output directory for multi-run results",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=7,
        help="Number of repeated runs per valid sweep combination",
    )
    parser.add_argument(
        "--run_start",
        type=int,
        default=9,
        help="First run index used in save_results. Default keeps old i+9 behavior.",
    )
    parser.add_argument(
        "--artifact_dir",
        type=str,
        default=None,
        help="Optional learned artifact directory. Used mainly by offline-then-online integration.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print valid/skipped combinations without running experiments",
    )

    args = parser.parse_args()

    run_multi_sweep(
        config_path=args.config_path,
        output_dir=args.output_dir,
        num_runs=args.num_runs,
        run_start=args.run_start,
        artifact_dir=args.artifact_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
