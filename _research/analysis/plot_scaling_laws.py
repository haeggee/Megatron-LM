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
"""plot_scaling_laws.py — optimizer scaling-law comparison from wandb.

Companion to plot_sweeps.py (and reuses its wandb listing/caching machinery).
Where plot_sweeps draws one curve per group with x = a swept LR, this draws the
*scaling-law* view of sweep-scaling-laws.sh: one line per (model size, optimizer)
pair, x = consumed tokens (or compute FLOPs), y = tail loss. Each point on a line
is one token-budget run; the line connects budgets for that size+optimizer.

Styling is split across the two axes of the grid so they read at a glance:
  • colour      -> optimizer   (from the YAML `optimizers` palette)
  • marker+dash -> model size  (from the YAML `sizes` table)
Two legends are drawn so each channel can be decoded independently.

Runs are matched by an anchored regex per (size, optimizer) cell, built from
`regex_template`; restarts/continuations merge exactly as in plot_sweeps. The
x value of each point is the run's FINAL consumed tokens (iter*GBS*seq), so the
token budget is read off the run itself — no need to parse the name tag. A run
that hasn't reached its intended budget (final_iter < train_samples/GBS) is drawn
with a hollow marker so it's visibly not comparable.

Usage:
    uv run plot_scaling_laws.py configs/scaling-laws.yaml
    uv run plot_scaling_laws.py configs/scaling-laws.yaml --x flops
    uv run plot_scaling_laws.py cfg.yaml --metric 'lm loss' --tail-iters 50
    uv run plot_scaling_laws.py cfg.yaml --refresh        # force redownload
    uv run plot_scaling_laws.py cfg.yaml --offline        # cache only, no API
    uv run plot_scaling_laws.py cfg.yaml --linear-y       # linear loss axis
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

# Reuse the listing/cache/fetch machinery so both scripts share one wandb cache.
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
    tail_stat,
    warn,
)


# ── config ───────────────────────────────────────────────────────────────────

@dataclass
class Optimizer:
    key: str
    label: str
    color: str
    regex: str | None = None        # per-optimizer template override of regex_template


@dataclass
class Size:
    key: str
    label: str
    marker: str
    linestyle: str
    params: float | None = None      # active params (incl. embeddings), for 6ND
    params_ne: float | None = None   # active NON-embedding params (--flops-params non-embed)


@dataclass
class ScalingCfg:
    base: Cfg                      # what the reused cache helpers consume
    optimizers: list[Optimizer]
    sizes: list[Size]
    regex_template: str
    x_mode: str                    # "tokens" | "flops" | "gbs"
    flops_params: str              # "active" | "non-embed" (which N for 6ND)
    linear_y: bool
    title: str | None = None       # plot title override (default derived from x_mode)
    groups: list[Group] = field(default_factory=list)   # (size, opt) cross product
    excludes: dict[str, list[str]] = field(default_factory=dict)  # group key -> drop regexes


def _hex(c: str) -> str:
    return c if c.startswith("#") else f"#{c}"


def load_cfg(path: Path, args: argparse.Namespace) -> ScalingCfg:
    raw = yaml.safe_load(path.read_text())

    def respath(key: str, default: str) -> Path:
        p = Path(raw.get(key, default))
        return p if p.is_absolute() else (path.parent / p).resolve()

    optimizers = []
    for key, o in (raw.get("optimizers") or {}).items():
        o = o or {}
        optimizers.append(Optimizer(key=key, label=o.get("label", key),
                                    color=_hex(o["color"]), regex=o.get("regex")))
    sizes = []
    for key, s in (raw.get("sizes") or {}).items():
        s = s or {}
        sizes.append(Size(key=key, label=s.get("label", key),
                          marker=s.get("marker", "o"),
                          linestyle=s.get("linestyle", "-"),
                          params=s.get("params"),
                          params_ne=s.get("params_ne")))
    if not optimizers or not sizes:
        sys.exit(f"{ERR}config needs non-empty `optimizers` and `sizes`{RST}")

    template = raw.get("regex_template", r"{size}-{opt}-.*t[0-9.]+b")
    cells = raw.get("cells") or {}                 # {"size/opt": {runs, regex, exclude}}
    global_exclude = raw.get("exclude")            # drop any base fullmatching this

    # (size, opt) cross product -> a plot_sweeps.Group with an anchored regex.
    # Precedence for the matching regex: cell.regex > optimizer.regex > template.
    # `runs` adds explicit base names (e.g. the untagged 14.68B LR-sweep baselines
    # that carry no t<budget>b tag); `exclude` drops wrong-config / stray matches.
    groups, excludes = [], {}
    for s in sizes:
        for o in optimizers:
            key = f"{s.key}/{o.key}"
            cell = cells.get(key) or {}
            tmpl = cell.get("regex") or o.regex or template
            groups.append(Group(
                key=key,
                label=f"{s.label} {o.label}",
                regex=tmpl.format(size=re.escape(s.key), opt=re.escape(o.key)),
                runs=list(cell.get("runs", [])),
                color=o.color,
            ))
            excludes[key] = [e for e in (global_exclude, cell.get("exclude")) if e]

    base = Cfg(
        entity=args.entity or raw.get("entity"),
        project=raw.get("project", "apertus-v2-optim-baseline"),
        metric=args.metric or raw.get("metric", "lm loss"),
        x_key="",  # unused here (x is derived tokens/flops, not a config key)
        tail_iters=args.tail_iters or int(raw.get("tail_iters", 50)),
        target_iter=None,
        extra_history_keys=[],
        cache_dir=respath("cache_dir", "../results/wandb-cache"),
        out_dir=Path(args.out_dir) if args.out_dir else respath("out_dir", "../results/plots"),
        groups=groups,
        tag=raw.get("tag", path.stem),
        title=raw.get("title"),
        xlabel=raw.get("xlabel"),
        ylabel=raw.get("ylabel"),
        subtitle=raw.get("subtitle"),
        subtitle_sweep=raw.get("subtitle_sweep")
    )
    return ScalingCfg(
        base=base,
        optimizers=optimizers,
        sizes=sizes,
        regex_template=template,
        x_mode=args.x or raw.get("x", "tokens"),
        flops_params=args.flops_params or raw.get("flops_params", "active"),
        linear_y=args.linear_y or bool(raw.get("linear_y", True)),
        groups=groups,
        excludes=excludes,
    )


# ── x-axis (tokens / compute) ──────────────────────────────────────────────────

def final_tokens(df, config: dict, base: str) -> float:
    """Consumed tokens at the run's last logged iteration."""
    return float(consumed_tokens(df, config, base)[-1])


def target_iter(config: dict) -> int | None:
    """Intended final iteration: train_samples / global_batch_size."""
    ts, gbs = config.get("train_samples"), config.get("global_batch_size")
    if not ts or not gbs:
        return None
    return int(ts) // int(gbs)


def x_from_tokens(scfg: ScalingCfg, size: Size, config: dict, tok: float, base: str) -> float:
    """Map (consumed tokens, config) -> the x value for the configured x_mode."""
    if scfg.x_mode == "gbs":
        gbs = config.get("global_batch_size")
        if not gbs:
            raise RuntimeError(f"{base}: global_batch_size missing from wandb config")
        return float(gbs)
    if scfg.x_mode == "tokens":
        return tok
    if scfg.x_mode == "flops":
        # N = active params: total (incl. embeddings) or non-embedding, per flag.
        n, fld = ((size.params_ne, "params_ne") if scfg.flops_params == "non-embed"
                  else (size.params, "params"))
        if not n:
            raise RuntimeError(
                f"{base}: x=flops ({scfg.flops_params}) needs `{fld}` for size "
                f"'{size.key}' in the config")
        return 6.0 * float(n) * tok   # standard 6ND training-FLOPs proxy
    raise RuntimeError(f"unknown x mode: {scfg.x_mode}")


def x_value(scfg: ScalingCfg, size: Size, df, config: dict, base: str) -> float:
    tok = 0.0 if scfg.x_mode == "gbs" else final_tokens(df, config, base)
    return x_from_tokens(scfg, size, config, tok, base)


def cap_wandb_retries(retry_max: int, http_timeout: float, retry_wait_max: float) -> None:
    """Make a flaky wandb API fail fast instead of stalling the whole run.

    The public API's data calls (scan_history etc.) go through wandb-core (Go),
    whose default retry policy is ~21 attempts — so one unreachable run can hang
    for >10 min. Those are governed by wandb-core *settings*, which read from env
    vars: internal (``x_*``) settings use the ``WANDB__`` double-underscore
    prefix. Must be set before the Settings singleton is built (i.e. before
    wandb.Api()). We also shrink the pure-Python ``RETRY_TIMEDELTA`` used by the
    listing path, which is imported by value into several modules.
    """
    os.environ.setdefault("WANDB__GRAPHQL_RETRY_MAX", str(retry_max))
    os.environ.setdefault("WANDB__GRAPHQL_RETRY_WAIT_MAX_SECONDS", str(retry_wait_max))
    os.environ.setdefault("WANDB_HTTP_TIMEOUT", str(http_timeout))  # x_graphql_timeout_seconds
    import datetime
    import importlib
    td = datetime.timedelta(seconds=max(http_timeout, retry_wait_max))
    for mod in ("wandb.apis.public.const", "wandb.apis.public.api",
                "wandb.apis.public.runs", "wandb.apis.public.files"):
        try:
            m = importlib.import_module(mod)
        except Exception:
            continue
        if hasattr(m, "RETRY_TIMEDELTA"):
            m.RETRY_TIMEDELTA = td


def fast_facts(api, cfg: Cfg, base: str, frags: list[dict], tail_iters: int) -> dict | None:
    """Tail-only fetch: just the final iter + last `tail_iters` of the metric.

    The scaling-law plot never needs full curves, so re-downloading tens of
    thousands of rows for a running run each refresh is wasted. We instead read
    config + a single short history window from the fragment that got FURTHEST
    (not merely the newest — a relaunch may crash early at a lower step). Returns
    the same fields a (check_run + tail_stat) pass would, or None if no history.
    """
    metric = cfg.metric
    withstep = [f for f in frags if f["last_step"] is not None]
    top = max(withstep, key=lambda f: f["last_step"]) if withstep else frags[-1]
    run = api.run(f"{cfg.entity}/{cfg.project}/{top['id']}")
    config = dict(run.config)
    last = top["last_step"]
    if last is None:   # crashed without a summary _step — scan the whole fragment
        hist = list(run.scan_history(keys=["_step", metric]))
    else:
        lo = max(0, int(last) - tail_iters - 5)
        hist = list(run.scan_history(keys=["_step", metric],
                                     min_step=lo, max_step=int(last) + 1))
    if not hist:
        return None
    steps = np.array([h["_step"] for h in hist], dtype=float)
    vals = np.array([h.get(metric, np.nan) for h in hist], dtype=float)
    final_iter = int(np.nanmax(steps))
    win = vals[steps > final_iter - tail_iters]
    win = win[np.isfinite(win)]
    diffs = np.diff(np.unique(steps))
    return {
        "config": config,
        "final_iter": final_iter,
        "state": top["state"],
        "log_interval": int(diffs.min()) if len(diffs) else 1,
        "n_fragments": len(frags),
        "tail_mean": float(win.mean()) if len(win) else float("nan"),
        "tail_std": float(win.std()) if len(win) else float("nan"),
        "tail_n": int(len(win)),
    }


# ── plot ───────────────────────────────────────────────────────────────────────

def plot(scfg: ScalingCfg, rows: list[dict], out: Path, show_incomplete: bool) -> None:
    cfg = scfg.base
    by_size = {s.key: s for s in scfg.sizes}
    set_plot_style(plt.rcParams)
    fig, ax = plt.subplots(figsize=(5, 4))

    for g in scfg.groups:
        size_key, _ = g.key.split("/", 1)
        size = by_size[size_key]
        pts = sorted((r for r in rows if r["group"] == g.key), key=lambda r: r["x"])
        if not pts:
            continue
        xs = [r["x"] for r in pts]
        ys = [r["tail_mean"] for r in pts]
        # connecting line (no markers); markers drawn per-point so incomplete
        # runs can be hollow ("reached fewer tokens than the budget").
        ax.plot(xs, ys, color=g.color, linestyle=size.linestyle, lw=1.5)
        for r in pts:
            if r["complete"]:
                ax.plot(r["x"], r["tail_mean"], marker=size.marker, ms=6,
                        color=g.color, linestyle="none")
            else:
                ax.plot(r["x"], r["tail_mean"], marker=size.marker, ms=7,
                        mfc="none", mec=g.color, mew=1.6, linestyle="none")

    ax.set_xscale("log")
    if scfg.x_mode == "gbs":
        # few discrete batch sizes — label the actual values, not powers of 10
        xt = sorted({r["x"] for r in rows})
        ax.set_xticks(xt)
        ax.set_xticklabels([f"{int(x)}" for x in xt])
        ax.xaxis.set_minor_locator(plt.NullLocator())
    else:
        # Add minor tick labels such as $5\times10^{20}$ between major (10^N) ticks, using math notation
        from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter

        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=[2,3,4,5,6,7,8,9]))
        def minor_log_label(val, pos=None):
            if val == 0:
                return ""
            exponent = int(np.floor(np.log10(val)))
            coeff = val / (10 ** exponent)
            if np.isclose(coeff, 5):
                return r"$5\cdot10^{{{}}}$".format(exponent)
            return ""
        ax.xaxis.set_minor_formatter(FuncFormatter(minor_log_label))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: r"$10^{{{}}}$".format(int(np.log10(val))) if val != 0 else ""))
 
    if not scfg.linear_y:
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(NullFormatter())

    if scfg.x_mode == "tokens":
        xlabel = "Training Tokens"
    elif scfg.x_mode == "gbs":
        xlabel = "Global Batch Size"
    else:
        nlbl = "non-embed" if scfg.flops_params == "non-embed" else "active"
        xlabel = f"Compute (FLOPs, 6ND, {nlbl} N)"
    if cfg.xlabel:
        ax.set_xlabel(cfg.xlabel)
    else:
        ax.set_xlabel(xlabel)
    if cfg.ylabel:
        ax.set_ylabel(cfg.ylabel)
    else:
        ax.set_ylabel(cfg.metric)
    hint = "  (o = below target budget)" if show_incomplete else ""
    default_title = ("batch-size scaling" if scfg.x_mode == "gbs"
                     else "optimizer scaling laws")
    title = f"{cfg.title or f'{cfg.tag}: {default_title}'}{hint}"
    ax.set_title(title, fontsize=autotune_font_size_title(title))

    # two legends: colour -> optimizer, marker+dash -> size
    opt_handles = [Line2D([], [], color=o.color, lw=2.5, label=o.label)
                   for o in scfg.optimizers]
    size_handles = [Line2D([], [], color="0.3", marker=s.marker, linestyle=s.linestyle,
                           ms=6, label=s.label) for s in scfg.sizes]
    leg1 = ax.legend(handles=opt_handles, frameon=True, fontsize=12,
                     loc="upper right")
    ax.add_artist(leg1)
    ax.legend(handles=size_handles, frameon=True, fontsize=12,
              loc="lower left")

    fig.tight_layout()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    print(f"{OK}wrote{RST} {out.with_suffix('.png')}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path, help="YAML scaling-law config")
    ap.add_argument("--metric", help="history key for y (default: yaml / 'lm loss')")
    ap.add_argument("--x", choices=["tokens", "flops", "gbs"],
                    help="x-axis (default: yaml / tokens). gbs = global batch size "
                         "(for sweep-batchsize.sh runs)")
    ap.add_argument("--flops-params", choices=["active", "non-embed"],
                    help="which active-param count N to use for 6ND flops "
                         "(default: yaml / active = incl. embeddings)")
    ap.add_argument("--tail-iters", type=int, help="tail window in iterations (default 50)")
    ap.add_argument("--linear-y", action="store_true", help="linear loss axis (default log)")
    ap.add_argument("--show-incomplete", action="store_true",
                    help="also plot budgets with no finished run (hollow markers, "
                         "placed at the tokens actually reached)")
    ap.add_argument("--entity", help="wandb entity (default: yaml / API default)")
    ap.add_argument("--out-dir", help="plot output dir (default: yaml)")
    ap.add_argument("--refresh", action="store_true", help="ignore cache, redownload")
    ap.add_argument("--fast", action="store_true",
                    help="tail-only fetch (final iter + last --tail-iters), in parallel; "
                         "skips the full-history download/cache. Much faster, esp. for "
                         "long running runs. Online only; can't make curve plots.")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel wandb fetches in --fast mode (default 8)")
    ap.add_argument("--api-timeout", type=float, default=20,
                    help="per-request wandb HTTP timeout, seconds (default 20). "
                         "Lower = a hung request fails sooner.")
    ap.add_argument("--api-retry-max", type=int, default=3,
                    help="max wandb-core retries per call (default 3; stock is ~21, "
                         "so one unreachable run stalls the whole run for >10 min).")
    ap.add_argument("--api-retry-secs", type=float, default=8,
                    help="cap the wait between wandb retries, seconds (default 8).")
    ap.add_argument("--offline", action="store_true", help="no wandb API calls, cache only")
    ap.add_argument("--list", action="store_true",
                    help="list merged run names matching the groups and exit")
    args = ap.parse_args()

    scfg = load_cfg(args.config, args)
    cfg = scfg.base
    columns = [cfg.metric]

    api = None
    live: dict[str, list[dict]] = {}
    if not args.offline:
        import wandb
        cap_wandb_retries(args.api_retry_max, args.api_timeout, args.api_retry_secs)
        api = wandb.Api(timeout=args.api_timeout)
        cfg.entity = cfg.entity or api.default_entity
        print(f"querying runs in {cfg.entity}/{cfg.project} (filtered by group patterns) ...")
        live = list_fragments(api, cfg, groups=cfg.groups)
        print(f"  {sum(len(v) for v in live.values())} wandb runs -> {len(live)} merged runs")
    elif cfg.entity is None:
        cfg.entity = "offline"

    if args.list:
        for base, frags in sorted(live.items()):
            states = ",".join(f["state"][0] for f in frags)
            print(f"{base}  [{len(frags)} frag {states}] last_step={frags[-1]['last_step']}")
        return 0

    # group key -> base names: regex matches ∪ explicit runs, minus excludes.
    resolved: list[tuple[Group, str]] = []
    for g in cfg.groups:
        pat = re.compile(g.regex)
        excl = [re.compile(e) for e in scfg.excludes.get(g.key, [])]
        if args.offline:
            pool = [p.name[:-len(".meta.json")]
                    for p in (cfg.cache_dir / cfg.project).glob("*.meta.json")]
        else:
            pool = list(live)
        bases = {b for b in pool if pat.fullmatch(b)}
        bases.update(g.runs)               # explicit names (e.g. untagged baselines)
        bases = sorted(b for b in bases
                       if not any(e.fullmatch(b) for e in excl))
        if not bases and not args.offline:
            warn(f"group '{g.key}': no runs matched (regex={g.regex})")
        resolved += [(g, b) for b in bases]

    size_by = {s.key: s for s in scfg.sizes}

    def assemble(g: Group, base: str, config: dict, facts: dict) -> dict | None:
        """Build a plot row from config + facts ({final_iter,state,log_interval,
        n_fragments,tail_mean,tail_std,tail_n}). None if the tail is unusable."""
        if facts["tail_n"] == 0 or not np.isfinite(facts["tail_mean"]):
            warn(f"{base}: no usable '{cfg.metric}' in the last {cfg.tail_iters} "
                 f"iters (n={facts['tail_n']}) — skipping")
            return None
        size = size_by[g.key.split("/", 1)[0]]
        gbs, seq = config.get("global_batch_size"), config.get("seq_length")
        tok = float(facts["final_iter"]) * (gbs or 0) * (seq or 0)
        try:
            x = x_from_tokens(scfg, size, config, tok, base)
        except RuntimeError as e:
            warn(str(e) + " — skipping")
            return None
        tgt = target_iter(config)
        if tgt is None:
            warn(f"{base}: no train_samples/global_batch_size — can't check completion")
        # tolerate being within one log interval of target (off-by-one).
        complete = tgt is None or facts["final_iter"] >= tgt - facts["log_interval"]
        budget = (config.get("global_batch_size") if scfg.x_mode == "gbs"
                  else config.get("train_samples")) or facts["final_iter"]
        return {"group": g.key, "base": base, "x": x,
                "tail_mean": facts["tail_mean"], "tail_std": facts["tail_std"],
                "tail_n": facts["tail_n"], "complete": complete,
                "target_iter": tgt, "budget": budget,
                "final_iter": facts["final_iter"], "state": facts["state"],
                "log_interval": facts["log_interval"], "n_fragments": facts["n_fragments"]}

    def facts_from_cache(base: str) -> tuple[dict, dict] | None:
        """(config, facts) from the local cache, or None if unreadable."""
        df, config, meta = load_run(cfg, base)
        if cfg.metric not in df.columns:
            return None
        facts = check_run(base, df, meta, cfg.metric)
        mean, std, n = tail_stat(df, cfg.metric, cfg.tail_iters)
        facts.update(tail_mean=mean, tail_std=std, tail_n=n)
        return config, facts

    # ── fast path: tail-only, parallel, no full-history download ──
    # Automatic: a run whose cache already matches wandb (cache_is_fresh) hasn't
    # changed — so it's served from the local cache with ZERO API calls. That
    # covers every finished run; only still-changing / uncached runs hit wandb,
    # and those use the cheap tail-only fetch.
    rows = []
    if args.fast and not args.offline:
        from concurrent.futures import ThreadPoolExecutor

        def fetch(item):
            g, base = item
            _, meta_path = cache_paths(cfg, base)
            if (not args.refresh and base in live
                    and cache_is_fresh(meta_path, live[base], columns)):
                try:
                    cached = facts_from_cache(base)
                except Exception as e:                   # noqa: BLE001 (fall through to wandb)
                    warn(f"{base}: cache read failed ({e}) — refetching")
                else:
                    if cached is not None:
                        return (g, base, *cached, "cache")
            if base not in live:
                warn(f"{base}: not found on wandb — skipping")
                return None
            try:
                facts = fast_facts(api, cfg, base, live[base], cfg.tail_iters)
            except Exception as e:                       # noqa: BLE001 (report + skip)
                warn(f"{base}: fast fetch failed ({e}) — skipping")
                return None
            if facts is None:
                warn(f"{base}: no history — skipping")
                return None
            return (g, base, facts["config"], facts, "wandb")

        n_fresh = sum(1 for _, b in resolved if not args.refresh and b in live
                      and cache_is_fresh(*((cache_paths(cfg, b)[1], live[b], columns))))
        print(f"  {len(resolved)} runs: {n_fresh} from cache (unchanged), "
              f"{len(resolved) - n_fresh} via tail-fetch ({args.workers} workers) ...")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for got in ex.map(fetch, resolved):
                if got is None:
                    continue
                g, base, config, facts, _src = got
                row = assemble(g, base, config, facts)
                if row is not None:
                    rows.append(row)
        return _finish(args, scfg, cfg, rows)

    # ── full path: download (cached) full history ──
    for g, base in resolved:
        _, meta_path = cache_paths(cfg, base)
        if not args.offline:
            if base not in live:
                if not meta_path.exists():
                    warn(f"{base}: not found on wandb and not cached — skipping")
                    continue   # explicit run name typo'd, or deleted on wandb
            elif args.refresh or not cache_is_fresh(meta_path, live[base], columns):
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
        mean, std, n = tail_stat(df, cfg.metric, cfg.tail_iters)
        facts.update(tail_mean=mean, tail_std=std, tail_n=n)
        row = assemble(g, base, config, facts)
        if row is not None:
            rows.append(row)

    return _finish(args, scfg, cfg, rows)


def _finish(args: argparse.Namespace, scfg: ScalingCfg, cfg: Cfg, rows: list[dict]) -> int:
    """Shared post-processing: dedupe per (size,opt,budget) -> table -> plot."""
    if not rows:
        print(f"{ERR}no runs to plot{RST}")
        return 1

    # one point per (size, opt, budget): collapse LR/wd/restart variants of the
    # same cell. Prefer a run that reached its budget; tie-break on lower tail.
    def rank(r):  # higher = better pick
        return (r["complete"], -r["tail_mean"])

    best: dict[tuple[str, float], dict] = {}
    for r in rows:
        k = (r["group"], r["budget"])
        cur = best.get(k)
        if cur is None or rank(r) > rank(cur):
            best[k] = r
    rows = list(best.values())

    incomplete = [r for r in rows if not r["complete"]]
    if incomplete:
        verb = "drawn hollow" if args.show_incomplete else "dropped (use --show-incomplete)"
        warn(f"{len(incomplete)}/{len(rows)} plotted budgets have NO run that "
             f"reached the target — {verb}; tail loss is at fewer tokens:")
        for r in sorted(incomplete, key=lambda r: r["base"]):
            print(f"    {r['base']}: iter {r['final_iter']}/{r['target_iter']} "
                  f"({r['state']}, {r['n_fragments']} frag)")
    if not args.show_incomplete:
        rows = [r for r in rows if r["complete"]]
        if not rows:
            print(f"{ERR}no complete runs to plot (try --show-incomplete){RST}")
            return 1

    # ── summary table + csv ──
    import pandas as pd
    table = pd.DataFrame([{k: r[k] for k in
                           ("group", "base", "x", "tail_mean", "tail_std", "tail_n",
                            "final_iter", "target_iter", "complete", "state")}
                          for r in rows]).sort_values(["group", "x"])
    pd.set_option("display.width", 220)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.5g}") + "\n")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = cfg.out_dir / f"{cfg.tag}-scaling-summary.csv"
    table.to_csv(csv_out, index=False)
    print(f"{OK}wrote{RST} {csv_out}")

    # ── plot ── (distinct filename per x-axis so tokens / flops / gbs coexist)
    if scfg.x_mode == "tokens":
        suffix = ""
    elif scfg.x_mode == "gbs":
        suffix = "-gbs"
    else:
        suffix = "-flops" + ("-ne" if scfg.flops_params == "non-embed" else "")
    plot(scfg, rows, cfg.out_dir / f"{cfg.tag}-scaling{suffix}", args.show_incomplete)
    return 0


if __name__ == "__main__":
    sys.exit(main())
