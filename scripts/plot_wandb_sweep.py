#!/usr/bin/env python3
"""Plot tail-loss vs. a hyperparameter for groups of W&B runs.

Plot configurations are loaded from YAML files in scripts/sweep_configs/.
Each YAML file defines one figure with one or more subplot panels.  Pass
specific files via --configs, or omit to load all *.yaml files in that
directory (sorted by filename).

X-axis values are auto-extracted from run names using regex patterns defined in
X_PARAM_PATTERNS.  When a pattern is not found in a run name, the series-level
'default_x_value' is used as a fallback.

Cache: data is stored in .wandb_sweep_cache/ (separate from plot_wandb_loss.py's cache).
"""

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from plotting_utils import set_plot_style
import numpy as np
import yaml
from tqdm import tqdm
import wandb


# ── Defaults ──────────────────────────────────────────────────────────────────

WANDB_ENTITY = "epfl-relay"
WANDB_PROJECT = "megatron_opt_v1"

CACHE_DIR = Path(__file__).resolve().parent / ".wandb_sweep_cache"

METRIC_KEY = "lm loss"
STEP_KEY = "consumed-tokens"

# Regex patterns for auto-extracting x-axis values from run names.
# Each pattern must have exactly one capture group that yields a numeric string.
X_PARAM_PATTERNS: dict[str, str] = {
    "lr":  r"(?<![a-z])lr(\d+\.?\d*(?:e[+-]?\d+)?)",  # lr0.004, lr1e-3
    "mlr": r"mlr(\d+\.?\d*)",                           # mlr2, mlr4, mlr8
    "wd":  r"(?<![a-z])wd(\d+\.?\d*)",                 # wd0, wd0.1
    "m":   r"(?<![a-z])m(\d+\.\d+)",                   # m0.95
    "b":   r"(?<![a-z])b(\d+\.\d+)",                   # b0.9
}

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent / "sweep_configs"



def load_plots(config_paths: list[str] | None) -> list[dict]:
    """Load plot configs from YAML files.

    If config_paths is None or empty, all *.yaml files in DEFAULT_CONFIG_DIR are
    loaded (sorted by filename so the numbering prefix controls order).
    """
    if not config_paths:
        if not DEFAULT_CONFIG_DIR.is_dir():
            raise SystemExit(
                f"No config files given and default directory {DEFAULT_CONFIG_DIR} "
                "does not exist.  Pass --configs explicitly."
            )
        yaml_files = sorted(DEFAULT_CONFIG_DIR.glob("*.yaml"))
        if not yaml_files:
            raise SystemExit(f"No *.yaml files found in {DEFAULT_CONFIG_DIR}")
    else:
        yaml_files = [Path(p) for p in config_paths]

    plots = []
    for yf in yaml_files:
        with open(yf) as f:
            data = yaml.safe_load(f)
        if data is None:
            continue
        data["_source_stem"] = yf.stem
        print(f"Loaded config: {yf.name}")
        plots.append(data)
    return plots


# ── Helpers ───────────────────────────────────────────────────────────────────


def slugify(text: str, max_len: int = 80) -> str:
    """Turn an arbitrary title into a filesystem-friendly slug."""
    s = text.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s[:max_len].rstrip("_")


def make_plot_filename(plot: dict, plot_idx: int) -> str:
    """Build a descriptive filename from the config's source stem and title."""
    stem = plot.get("_source_stem", "")
    title = plot.get("title", "")
    if stem:
        return f"sweep_{stem}.png"
    if title:
        return f"sweep_{slugify(title)}.png"
    return f"sweep_plot_{plot_idx}.png"


def parse_experiment(exp, default_entity: str, default_project: str) -> tuple[str, str, str]:
    """Parse an experiment entry into (entity, project, name)."""
    if isinstance(exp, dict):
        name = exp["name"]
        entity = exp.get("entity", default_entity)
        project = exp.get("project", default_project)
    else:
        name = str(exp)
        entity = default_entity
        project = default_project
    return entity, project, name


def strip_run_suffix(name: str) -> str:
    """Remove a trailing job/run-id suffix like '-1570020' or '-j1552378'."""
    return re.sub(r"-j?\d{5,}$", "", name)


def cache_path_for(run_id: str) -> Path:
    return CACHE_DIR / f"{run_id}.json"


def fetch_run_history(run, keys: list[str]) -> list[dict]:
    """Download the requested keys from a run, using a local JSON cache."""
    cached = cache_path_for(run.id)
    if cached.exists():
        with open(cached) as f:
            return json.load(f)

    rows = []
    for row in run.scan_history(keys=keys, page_size=5000):
        rows.append({k: row.get(k) for k in keys})

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cached, "w") as f:
        json.dump(rows, f)
    return rows


def extract_x_value(run_name: str, x_param: str, default) -> float | None:
    """Extract the x-axis value from a run name using X_PARAM_PATTERNS.

    Returns default if the pattern is not found.  Returns None if default is
    also None (run will be skipped with a warning).
    """
    pattern = X_PARAM_PATTERNS.get(x_param)
    if pattern:
        m = re.search(pattern, run_name)
        if m:
            return float(m.group(1))
    return default


def compute_tail_loss(
    steps: np.ndarray, vals: np.ndarray, tail: int, metric: str
) -> tuple[float, float] | tuple[None, None]:
    """Return (mean, std) over the last `tail` raw data points."""
    if len(vals) == 0:
        return None, None
    tail_k = min(tail, len(vals))
    tail_vals = vals[-tail_k:]
    return float(np.mean(tail_vals)), float(np.std(tail_vals))


def collect_all_runs(plots: list[dict], args) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fetch and cache all runs referenced in PLOTS.

    Returns a dict mapping base experiment name → (steps_array, vals_array).
    Multiple W&B runs with the same base name are concatenated and deduplicated
    (matching the logic in plot_wandb_loss.py).
    """
    # Collect all unique (entity, project, name) specs across every series.
    all_specs: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for plot in plots:
        for sp in plot["subplots"]:
            for series in sp["series"]:
                for run_spec in series["runs"]:
                    key = parse_experiment(run_spec, args.entity, args.project)
                    if key not in seen:
                        seen.add(key)
                        all_specs.append(key)

    # Group by (entity, project) for batched API queries.
    by_project: dict[tuple[str, str], list[str]] = defaultdict(list)
    for entity, project, name in all_specs:
        by_project[(entity, project)].append(name)

    api = wandb.Api()

    # Maps base name → list of wandb run objects (possibly multiple restarts).
    run_groups: dict[str, list] = {name: [] for _, _, name in all_specs}

    for (entity, project), names in tqdm(by_project.items(), desc="Projects", unit="proj"):
        path = f"{entity}/{project}"
        name_regex = "|".join(re.escape(n) for n in names)
        filters = {"display_name": {"$regex": f"^({name_regex})(-j?\\d{{5,}})?$"}}
        runs = api.runs(path, filters=filters, per_page=1000)
        for run in runs:
            base = strip_run_suffix(run.name)
            if base in run_groups:
                run_groups[base].append(run)
            else:
                tqdm.write(f"  [warn] '{run.name}' matched filter but base '{base}' is unknown")

    for name in run_groups:
        run_groups[name].sort(key=lambda r: r.name)

    keys = [args.step_key, args.metric]
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    exp_pbar = tqdm(run_groups.items(), desc="Runs", unit="exp")
    for exp_name, run_group in exp_pbar:
        exp_pbar.set_postfix_str(exp_name[:40])
        if not run_group:
            tqdm.write(f"  [skip] No W&B runs found for '{exp_name}'")
            continue

        all_steps, all_vals = [], []
        for run in run_group:
            tqdm.write(f"  Loading {run.name} ({run.id}) …")
            rows = fetch_run_history(run, keys)
            for row in rows:
                s, v = row.get(args.step_key), row.get(args.metric)
                if s is not None and v is not None:
                    all_steps.append(s)
                    all_vals.append(v)

        if not all_steps:
            tqdm.write(f"  [skip] No data points for '{exp_name}'")
            continue

        order = np.argsort(all_steps)
        all_steps = np.array(all_steps)[order]
        all_vals = np.array(all_vals)[order]

        # Deduplicate: for repeated steps keep the value from the latest run.
        _, unique_idx = np.unique(all_steps[::-1], return_index=True)
        unique_idx = len(all_steps) - 1 - unique_idx
        unique_idx.sort()
        all_steps = all_steps[unique_idx]
        all_vals = all_vals[unique_idx]

        result[exp_name] = (all_steps, all_vals)

    return result


def plot_all(plots: list[dict], run_data: dict, args) -> list:
    """Create one figure per plot entry, with subplots as panels."""
    figures = []
    for plot_idx, plot in enumerate(plots):
        n_subplots = len(plot["subplots"])
        fig, axes = plt.subplots(
            1, n_subplots,
            figsize=(6 * n_subplots, 5),
            squeeze=False,
            sharey=(n_subplots > 1),
        )
        fig.suptitle(plot["title"], fontsize=14, fontweight="bold")

        for sp_idx, sp in enumerate(plot["subplots"]):
            ax = axes[0][sp_idx]
            x_param = sp["x_param"]

            # Assert that every run in this subplot ends at the same consumed-tokens
            # value, so tail losses are all measured at the same training horizon.
            tail_end_steps: dict[str, float] = {}  # run_name → final step
            for series in sp["series"]:
                for run_spec in series["runs"]:
                    _, _, run_name = parse_experiment(run_spec, args.entity, args.project)
                    if run_name in run_data:
                        tail_end_steps[run_name] = float(run_data[run_name][0][-1])

            if tail_end_steps:
                unique_ends = np.array(list(tail_end_steps.values()))
                rel_spread = (unique_ends.max() - unique_ends.min()) / unique_ends.max()
                assert rel_spread < 0.0001, (
                    f"Subplot '{sp['title']}': runs end at different consumed-tokens "
                    f"(spread {rel_spread:.1%}). Per-run final steps:\n"
                    + "\n".join(
                        f"  {name}: {step:,.0f}" for name, step in sorted(tail_end_steps.items())
                    )
                )

            for series in sp["series"]:
                label = series["label"]
                default_x = series.get("default_x_value")
                xs, means, stds = [], [], []

                for run_spec in series["runs"]:
                    _, _, run_name = parse_experiment(run_spec, args.entity, args.project)

                    x_val = extract_x_value(run_name, x_param, default_x)
                    if x_val is None:
                        tqdm.write(
                            f"  [skip] '{run_name}': cannot determine x for param '{x_param}'"
                        )
                        continue

                    if run_name not in run_data:
                        tqdm.write(f"  [skip] '{run_name}': no data (run not found or empty)")
                        continue

                    steps, vals = run_data[run_name]
                    mean, std = compute_tail_loss(steps, vals, args.tail, args.metric)
                    if mean is None:
                        continue

                    xs.append(x_val)
                    means.append(mean)
                    stds.append(std)

                if not xs:
                    continue

                # Sort by x value for a clean line.
                order = np.argsort(xs)
                xs = np.array(xs)[order]
                means = np.array(means)[order]
                stds = np.array(stds)[order]

                if args.no_errbar:
                    line = ax.plot(xs, means, marker="x", linewidth=1.5, label=label)
                    color = line[0].get_color()
                else:
                    ebar = ax.errorbar(
                        xs, means, yerr=stds,
                        marker="x", linewidth=1.5, capsize=4, label=label,
                    )
                    color = ebar[0].get_color()

                if args.plot_min_estimate and len(xs) >= 3:
                    fit_xs = np.log(xs) if args.xscale == "log" else xs
                    coeffs = np.polyfit(fit_xs, means, 2)
                    a, b, _c = coeffs
                    if a > 0:
                        x_min_fit = -b / (2 * a)
                        y_min_fit = np.polyval(coeffs, x_min_fit)
                        x_min_plot = np.exp(x_min_fit) if args.xscale == "log" else x_min_fit
                        ax.plot(
                            x_min_plot, y_min_fit,
                            marker="*", markersize=8, color=color,
                            markeredgecolor="black", markeredgewidth=0.8,
                            zorder=5,
                        )

                obs_min_idx = int(np.argmin(means))
                ax.plot(
                    xs[obs_min_idx], means[obs_min_idx],
                    marker="s", markersize=8, color=color,
                    markeredgecolor="black", markeredgewidth=0.8,
                    zorder=4,
                )

            ax.set_xscale(args.xscale)
            ax.set_xlabel(sp["x_label"], fontsize=11)
            if sp_idx == 0:
                ax.set_ylabel(f"Tail mean {args.metric}", fontsize=11)
            ax.set_title(sp["title"], fontsize=12)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            for spine in ax.spines.values():
                spine.set_visible(True)

        fig.tight_layout()
        figures.append((plot_idx, plot, fig))

    return figures


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs", nargs="*", default=None, metavar="YAML",
        help="YAML config files to plot (default: all *.yaml in scripts/sweep_configs/)",
    )
    parser.add_argument("--entity", default=WANDB_ENTITY, help="Default W&B entity")
    parser.add_argument("--project", default=WANDB_PROJECT, help="Default W&B project")
    parser.add_argument(
        "--metric", default=METRIC_KEY,
        help="Metric key to download (default: 'lm loss')",
    )
    parser.add_argument(
        "--step-key", default=STEP_KEY,
        help="Step key (default: 'consumed-tokens')",
    )
    parser.add_argument(
        "--tail", type=int, default=100,
        help="Number of final raw steps to average for tail loss (default: 100)",
    )
    parser.add_argument(
        "--xscale", default="linear", choices=["linear", "log"],
        help="X-axis scale (default: linear; use 'log' for LR sweeps)",
    )
    parser.add_argument(
        "--no-errbar", action="store_true",
        help="Suppress ±std error bars (default: show them)",
    )
    parser.add_argument(
        "--plot-min-estimate", action="store_true",
        help="Fit a quadratic to each series curve and mark its minimum with a star",
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="Delete .wandb_sweep_cache/ and re-download everything",
    )
    parser.add_argument(
        "--save", default=None, metavar="DIR",
        help="Save figures to DIR/ with names derived from config filenames (e.g. sweep_03_preln_finln_flat.png)",
    )
    args = parser.parse_args()

    if args.clear_cache and CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        print(f"Cleared cache at {CACHE_DIR}")


    rcParams = plt.rcParams
    set_plot_style(rcParams)

    plots = load_plots(args.configs)

    run_data = collect_all_runs(plots, args)

    figures = plot_all(plots, run_data, args)

    if args.save:
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)
        for plot_idx, plot, fig in figures:
            out = save_dir / make_plot_filename(plot, plot_idx)
            fig.savefig(out, dpi=400, bbox_inches="tight")
            print(f"Saved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
