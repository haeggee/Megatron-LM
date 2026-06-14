#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "wandb>=0.19",
#   "pandas>=2.0",
#   "numpy",
#   "matplotlib>=3.8",
#   "seaborn>=0.13",
#   "pyyaml",
# ]
# ///
"""plot_rewarming.py — re-warming (warm-restart) training curves from wandb.

Companion to plot_sweeps.py (reuses its wandb listing/caching machinery). Where
the sweep plotters draw one curve per *group* and pick a best run, this draws an
explicit, ordered list of named curves — the warm-restart.sh 2x2 ({reuse vs fresh
optimizer moments} x {warmup ramp vs jump-to-peak}) plus the source run and the
no-restart reference — and writes TWO figures:

  1. <tag>-loss   — `metric` (default 'lm loss') vs x
  2. <tag>-grad   — `grad_metric` (default 'grad-norm') vs x, log y

x is consumed tokens in B (iter*GBS*seq), or raw iterations with --x iters. The
continuations branch from the source at `branch_iter` (a dashed vertical rule);
the source spans 0->branch, the reference spans the whole run. Curves are smoothed
with a rolling mean (--smooth, default 100; grad norm uses --grad-smooth, default
20, so the warm-up spike survives). Each curve is one or more wandb runs sharing a
base name (sans -<jobid>); fragments/restarts merge exactly as in plot_sweeps.

Usage:
    uv run plot_rewarming.py configs/rewarming.yaml
    uv run plot_rewarming.py configs/rewarming.yaml --x iters --smooth 100
    uv run plot_rewarming.py configs/rewarming.yaml --refresh        # redownload
    uv run plot_rewarming.py configs/rewarming.yaml --offline        # cache only
    uv run plot_rewarming.py configs/rewarming.yaml --list           # show merged names
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

# Reuse the listing/cache/fetch machinery so all plotters share one wandb cache.
from plot_sweeps import (
    ERR,
    OK,
    RST,
    Cfg,
    Group,
    autotune_font_size_title,
    cache_is_fresh,
    cache_paths,
    check_run,
    consumed_tokens,
    download_run,
    list_fragments,
    load_run,
    set_plot_style,
    warn,
)


@dataclass
class Curve:
    runs: list[str]                 # base names (sans -<jobid>); fragments merge
    label: str
    color: str | None = None
    linestyle: str = "-"
    lw: float = 1.5


@dataclass
class RewarmCfg:
    base: Cfg                       # what the reused cache helpers consume
    grad_metric: str
    grad_ylabel: str | None
    branch_iter: int | None
    curves: list[Curve] = field(default_factory=list)


def _hex(c: str | None) -> str | None:
    if c is None:
        return None
    return c if c.startswith("#") else f"#{c}"


def load_cfg(path: Path, args: argparse.Namespace) -> RewarmCfg:
    raw = yaml.safe_load(path.read_text())

    def respath(key: str, default: str) -> Path:
        p = Path(raw.get(key, default))
        return p if p.is_absolute() else (path.parent / p).resolve()

    curves = []
    for c in raw.get("curves") or []:
        runs = list(c.get("runs", []))
        if c.get("run"):
            runs.append(c["run"])
        if not runs:
            sys.exit(f"{ERR}curve {c!r} has no `run`/`runs`{RST}")
        curves.append(Curve(
            runs=runs,
            label=c.get("label", runs[0]),
            color=_hex(c.get("color")),
            linestyle=c.get("linestyle", "-"),
            lw=float(c.get("lw", 1.5)),
        ))
    if not curves:
        sys.exit(f"{ERR}config needs a non-empty `curves` list{RST}")

    grad_metric = args.grad_metric or raw.get("grad_metric", "grad-norm")
    # Each curve becomes a Group so the cache helpers (which key on Group.runs)
    # can drive listing/download unchanged; the x_key machinery is unused here.
    groups = [Group(key=str(i), label=cv.label, runs=cv.runs)
              for i, cv in enumerate(curves)]
    base = Cfg(
        entity=args.entity or raw.get("entity"),
        project=raw.get("project", "apertus-v2-optim-baseline"),
        metric=args.metric or raw.get("metric", "lm loss"),
        x_key="",
        tail_iters=int(raw.get("tail_iters", 50)),
        target_iter=None,
        extra_history_keys=[grad_metric],   # cache grad norm alongside the loss
        cache_dir=respath("cache_dir", "../results/wandb-cache"),
        out_dir=Path(args.out_dir) if args.out_dir else respath("out_dir", "../results/plots"),
        groups=groups,
        tag=raw.get("tag", path.stem),
        title=raw.get("title"),
        xlabel=raw.get("xlabel"),
        ylabel=raw.get("ylabel"),
        subtitle=raw.get("subtitle"),
        subtitle_sweep=raw.get("subtitle_sweep"),
    )
    return RewarmCfg(
        base=base,
        grad_metric=grad_metric,
        grad_ylabel=raw.get("grad_ylabel"),
        branch_iter=raw.get("branch_iter"),
        curves=curves,
    )


# ── x axis ───────────────────────────────────────────────────────────────────

def x_axis(df: pd.DataFrame, config: dict, base: str, x_mode: str) -> tuple[np.ndarray, float]:
    """Returns (x values, branch_iter -> x scale factor) for the given mode."""
    if x_mode == "iters":
        return df["_step"].to_numpy(dtype=float), 1.0
    # tokens in billions = iter * GBS * seq / 1e9
    tok = consumed_tokens(df, config, base) / 1e9
    gbs, seq = config["global_batch_size"], config["seq_length"]
    return tok, gbs * seq / 1e9


# ── plot ─────────────────────────────────────────────────────────────────────

def smoothed(y: pd.Series, win: int) -> pd.Series:
    return y.rolling(win, min_periods=1).mean() if win > 1 else y


def draw(rcfg: RewarmCfg, loaded: list[dict], metric: str, smooth: int,
         x_mode: str, branch_x: float | None, log_y: bool, ylabel: str | None,
         ylim: tuple[float, float] | None, out: Path) -> None:
    cfg = rcfg.base
    set_plot_style(plt.rcParams)
    fig, ax = plt.subplots(figsize=(5, 4))

    post_branch_vals: list[float] = []
    drawn = 0
    for item in loaded:
        df, cv = item["df"], item["curve"]
        if metric not in df.columns:
            warn(f"{item['base']}: '{metric}' not in history — skipping this curve")
            continue
        x, _ = x_axis(df, item["config"], item["base"], x_mode)
        y = smoothed(df[metric], smooth)
        ax.plot(x, y, label=cv.label, color=cv.color, linestyle=cv.linestyle,
                lw=cv.lw)
        drawn += 1
        if branch_x is not None:
            mask = df["_step"].to_numpy() >= (rcfg.branch_iter or 0)
            post_branch_vals += [v for v in y.to_numpy()[mask] if np.isfinite(v)]

    if not drawn:
        warn(f"no curves had '{metric}' — not writing {out.name}")
        plt.close(fig)
        return

    if branch_x is not None:
        ax.axvline(branch_x, color="0.4", linestyle=":", lw=1.2, zorder=0)

    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(NullFormatter())
    elif ylim is not None:
        ax.set_ylim(*ylim)
    elif post_branch_vals:
        # frame the post-branch comparison region; the source run's pre-branch
        # loss cliff runs off the top rather than squashing everything flat.
        lo, hi = min(post_branch_vals), max(post_branch_vals)
        ax.set_ylim(lo - 0.05 * (hi - lo + 0.1), hi + 0.1 * (hi - lo + 0.1))

    ax.set_xlabel("Training Tokens (B)" if x_mode == "tokens" else "Iteration")
    yl = ylabel or metric
    # if smooth > 1:
    #     yl += f" (rolling mean, {smooth})"
    ax.set_ylabel(yl)
    title = cfg.title or f"{cfg.tag}"
    ax.set_title(title, fontsize=autotune_font_size_title(title))
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=True, ncol=1, fontsize=9)

    fig.tight_layout()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    print(f"{OK}wrote{RST} {out.with_suffix('.png')}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path, help="YAML re-warming config")
    ap.add_argument("--metric", help="loss history key (default: yaml / 'lm loss')")
    ap.add_argument("--grad-metric", help="grad-norm history key (default: yaml / 'grad-norm')")
    ap.add_argument("--x", choices=["tokens", "iters"], default="tokens",
                    help="x-axis: consumed tokens (B) or raw iteration (default tokens)")
    ap.add_argument("--smooth", type=int, default=100,
                    help="rolling-mean window for the loss curve (default 100)")
    ap.add_argument("--grad-smooth", type=int, default=20,
                    help="rolling-mean window for the grad-norm curve (default 20)")
    ap.add_argument("--loss-ylim", type=float, nargs=2, metavar=("LO", "HI"),
                    help="explicit loss y-limits (default: auto-frame post-branch region)")
    ap.add_argument("--grad-linear", action="store_true",
                    help="linear grad-norm axis (default log)")
    ap.add_argument("--entity", help="wandb entity (default: yaml / API default)")
    ap.add_argument("--out-dir", help="plot output dir (default: yaml)")
    ap.add_argument("--refresh", action="store_true", help="ignore cache, redownload")
    ap.add_argument("--offline", action="store_true", help="no wandb API calls, cache only")
    ap.add_argument("--list", action="store_true",
                    help="list merged run names matching the curves and exit")
    args = ap.parse_args()

    rcfg = load_cfg(args.config, args)
    cfg = rcfg.base
    columns = sorted({cfg.metric, rcfg.grad_metric})

    api = None
    live: dict[str, list[dict]] = {}
    if not args.offline:
        import wandb
        api = wandb.Api(timeout=120)
        cfg.entity = cfg.entity or api.default_entity
        print(f"querying runs in {cfg.entity}/{cfg.project} (filtered by curve names) ...")
        live = list_fragments(api, cfg, groups=cfg.groups)
        print(f"  {sum(len(v) for v in live.values())} wandb runs -> {len(live)} merged runs")
    elif cfg.entity is None:
        cfg.entity = "offline"

    if args.list:
        for base, frags in sorted(live.items()):
            states = ",".join(f["state"][0] for f in frags)
            print(f"{base}  [{len(frags)} frag {states}] last_step={frags[-1]['last_step']}")
        return 0

    # ── fetch / cache, in curve order ──
    loaded = []
    for cv in rcfg.curves:
        for base in cv.runs:
            _, meta_path = cache_paths(cfg, base)
            if not args.offline:
                if base not in live:
                    warn(f"{base}: listed in config but not found on wandb — skipping")
                    continue
                if args.refresh or not cache_is_fresh(meta_path, live[base], columns):
                    print(f"  downloading {base}")
                    try:
                        download_run(api, cfg, base, live[base], columns)
                    except RuntimeError as e:
                        warn(str(e) + " — skipping")
                        continue
            elif not meta_path.exists():
                warn(f"{base}: not in cache (offline) — skipping")
                continue

            df, config, meta = load_run(cfg, base)
            if cfg.metric not in df.columns:
                warn(f"{base}: metric '{cfg.metric}' not in cached history — skipping")
                continue
            facts = check_run(base, df, meta, cfg.metric)
            print(f"  {base}: iter {int(df['_step'].min())}..{int(df['_step'].max())} "
                  f"({facts['state']}, {facts['n_fragments']} frag)")
            loaded.append({"curve": cv, "base": base, "df": df, "config": config})
            break  # one resolved run per curve (its fragments already merged)

    if not loaded:
        print(f"{ERR}no runs to plot{RST}")
        return 1

    branch_x = None
    if rcfg.branch_iter is not None:
        sf = (1.0 if args.x == "iters"
              else loaded[0]["config"]["global_batch_size"]
              * loaded[0]["config"]["seq_length"] / 1e9)
        branch_x = rcfg.branch_iter * sf

    draw(rcfg, loaded, cfg.metric, args.smooth, args.x, branch_x,
         log_y=False, ylabel=cfg.ylabel,
         ylim=tuple(args.loss_ylim) if args.loss_ylim else None,
         out=cfg.out_dir / f"{cfg.tag}-loss")
    draw(rcfg, loaded, rcfg.grad_metric, args.grad_smooth, args.x, branch_x,
         log_y=not args.grad_linear, ylabel=rcfg.grad_ylabel,
         ylim=None, out=cfg.out_dir / f"{cfg.tag}-grad")
    return 0


if __name__ == "__main__":
    sys.exit(main())
