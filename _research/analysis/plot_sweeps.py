#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "wandb>=0.19",
#   "pandas>=2.0",
#   "numpy",
#   "matplotlib>=3.8",
#   "pyyaml",
# ]
# ///
"""plot_sweeps.py — LR-sweep + training-curve plots from wandb, driven by a YAML config.

Run names follow the framework convention ``<exp-name>-<slurm-job-id>``; runs that
share the prefix and differ only in the trailing job id are restarts/continuations
of the same training and get their histories merged (later fragment wins on
overlapping iterations).

Full (unsampled) histories are downloaded via ``scan_history`` and cached as
csv.gz + meta.json per merged run. The cache is invalidated automatically when
the set of wandb fragments changes, a fragment gained steps, or the requested
history columns aren't cached yet.

Outputs:
  1. sweep plot   — x = a config key (default matrix_lr), y = mean(metric) over
                    the last --tail-iters iterations, one line per group
  2. curves plot  — best run per group (lowest tail metric), metric vs consumed
                    tokens (= iteration * global_batch_size * seq_length)
  3. summary csv + stdout table

Usage:
    uv run plot_sweeps.py configs/optimizer-sweep-270m.yaml
    uv run plot_sweeps.py cfg.yaml --metric 'lm loss' --x-key matrix_lr --tail-iters 50
    uv run plot_sweeps.py cfg.yaml --refresh            # force redownload
    uv run plot_sweeps.py cfg.yaml --offline            # cache only, no API calls
    uv run plot_sweeps.py cfg.yaml --list               # list merged run names in the project
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

# Trailing "-<slurm job id>" (4+ digits so LR-ish suffixes like '-p0.5' survive).
JOB_ID_RE = re.compile(r"-(\d{4,})$")

WARN = "\033[33m"
ERR = "\033[31m"
OK = "\033[32m"
RST = "\033[0m"


def warn(msg: str) -> None:
    print(f"{WARN}WARNING:{RST} {msg}")


def split_base(name: str) -> tuple[str, int | None]:
    """'270m-moe-adamw-...-mi300-373983' -> ('270m-moe-adamw-...-mi300', 373983)."""
    m = JOB_ID_RE.search(name)
    return (name[: m.start()], int(m.group(1))) if m else (name, None)


# ── config ───────────────────────────────────────────────────────────────────

@dataclass
class Group:
    key: str
    label: str
    runs: list[str] = field(default_factory=list)   # explicit base names
    regex: str | None = None                        # matched against base names
    x_key: str | None = None                        # per-group override
    color: str | None = None


@dataclass
class Cfg:
    entity: str | None
    project: str
    metric: str
    x_key: str
    tail_iters: int
    target_iter: int | None
    extra_history_keys: list[str]
    cache_dir: Path
    out_dir: Path
    groups: list[Group]
    tag: str  # filename stem for plots/csv


def load_cfg(path: Path, args: argparse.Namespace) -> Cfg:
    raw = yaml.safe_load(path.read_text())

    def respath(key: str, default: str) -> Path:
        p = Path(raw.get(key, default))
        return p if p.is_absolute() else (path.parent / p).resolve()

    groups = []
    for key, g in raw["groups"].items():
        g = g or {}
        groups.append(Group(
            key=key,
            label=g.get("label", key),
            runs=list(g.get("runs", [])),
            regex=g.get("regex"),
            x_key=g.get("x_key"),
            color=g.get("color"),
        ))
    return Cfg(
        entity=args.entity or raw.get("entity"),
        project=raw.get("project", "apertus-v2-optim-baseline"),
        metric=args.metric or raw.get("metric", "lm loss"),
        x_key=args.x_key or raw.get("x_key", "matrix_lr"),
        tail_iters=args.tail_iters or int(raw.get("tail_iters", 50)),
        target_iter=args.target_iter or raw.get("target_iter"),
        extra_history_keys=list(raw.get("extra_history_keys", [])),
        cache_dir=respath("cache_dir", "../results/wandb-cache"),
        out_dir=Path(args.out_dir) if args.out_dir else respath("out_dir", "../results/plots"),
        groups=groups,
        tag=raw.get("tag", path.stem),
    )


# ── wandb listing + cache ────────────────────────────────────────────────────

def list_fragments(api, cfg: Cfg, groups: list[Group] | None = None) -> dict[str, list[dict]]:
    """Paginated listing of the project -> {base_name: [fragment meta, ...]}.

    When `groups` is given, their regexes / explicit run names are pushed down
    as a server-side display_name filter, so wandb only returns matching runs.
    (An unfiltered listing drags every run's full config along and is slow;
    only --list does that.) Server-side matching is a prefix superset — the
    precise fullmatch against base names still happens client-side.
    """
    filters = None
    if groups is not None:
        ors = []
        for g in groups:
            if g.regex:
                ors.append({"display_name": {"$regex": "^" + g.regex.lstrip("^")}})
            for base in g.runs:
                ors.append({"display_name": {"$regex": f"^{re.escape(base)}(-[0-9]+)?$"}})
        if not ors:
            return {}
        filters = {"$or": ors}
    out: dict[str, list[dict]] = {}
    for r in api.runs(f"{cfg.entity}/{cfg.project}", filters=filters, per_page=500):
        base, job_id = split_base(r.name)
        out.setdefault(base, []).append({
            "id": r.id,
            "name": r.name,
            "job_id": job_id,
            "state": r.state,
            "created_at": str(r.created_at),
            "last_step": r.summary.get("_step"),
        })
    for frags in out.values():
        frags.sort(key=lambda f: (f["created_at"], f["job_id"] or 0))
    return out


def cache_paths(cfg: Cfg, base: str) -> tuple[Path, Path]:
    d = cfg.cache_dir / cfg.project
    return d / f"{base}.csv.gz", d / f"{base}.meta.json"


def cache_is_fresh(meta_path: Path, live_frags: list[dict], columns: list[str]) -> bool:
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    if not set(columns) <= set(meta.get("columns", [])):
        return False  # a newly requested metric isn't cached yet
    cached = {f["id"]: (f["state"], f["last_step"]) for f in meta["fragments"]}
    live = {f["id"]: (f["state"], f["last_step"]) for f in live_frags}
    if cached != live:  # new fragment, new steps, or state change all invalidate
        return False
    # caches whose fetched rows fall short of the fragment's known last step
    # were written by the broken resumed-run scan (or lack the bookkeeping
    # entirely, pre-fix caches) -> redownload
    return all(f["last_step"] is None or f.get("fetched_last") == f["last_step"]
               for f in meta["fragments"])


def scan_fragment(run, columns: list[str], last_step, page_size: int = 10_000) -> list[dict]:
    """Full-fidelity history of one wandb run, robust to resumed runs.

    scan_history() stops at the first EMPTY page (wandb<=0.27,
    BetaHistoryScan.__next__), so a continuation whose history starts at
    iter N > page_size silently yields 0 rows when scanned from step 0.
    Scanning explicit page-sized [min_step, max_step) windows up to the
    fragment's summary _step sidesteps that: an empty window is just a
    window with no data, later windows still get fetched.

    keys= restricts to rows where ALL keys are present — fine here because
    Megatron commits one wandb row per iteration.
    """
    keys = ["_step"] + columns
    if last_step is None:
        # no summary -> extent unknown; plain scan (correct for fresh runs,
        # and a resumed run that never logged has no rows to miss anyway)
        return list(run.scan_history(keys=keys, page_size=page_size))
    rows: list[dict] = []
    hi = int(last_step) + 1  # max_step is exclusive
    for lo in range(0, hi, page_size):
        rows.extend(run.scan_history(keys=keys, page_size=page_size,
                                     min_step=lo, max_step=min(lo + page_size, hi)))
    return rows


def download_run(api, cfg: Cfg, base: str, frags: list[dict], columns: list[str]) -> None:
    """Fetch full histories of all fragments, merge, write cache."""
    csv_path, meta_path = cache_paths(cfg, base)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    dfs, config = [], {}
    for i, frag in enumerate(frags):
        run = api.run(f"{cfg.entity}/{cfg.project}/{frag['id']}")
        config = dict(run.config) or config  # latest non-empty fragment's config wins
        rows = scan_fragment(run, columns, frag["last_step"])
        if not rows:
            if frag["last_step"] is not None:
                raise RuntimeError(
                    f"{frag['name']}: 0 history rows despite summary "
                    f"_step={frag['last_step']} — scan failed, not caching")
            print(f"    fragment {frag['name']} ({frag['state']}): no history, skipped")
            continue
        df = pd.DataFrame(rows)
        frag["fetched_first"] = int(df["_step"].min())
        frag["fetched_last"] = int(df["_step"].max())
        if frag["last_step"] is not None and frag["fetched_last"] < int(frag["last_step"]):
            raise RuntimeError(
                f"{frag['name']}: fetched history ends at {frag['fetched_last']} "
                f"< summary _step {frag['last_step']} — incomplete, not caching")
        df["_frag"] = i
        dfs.append(df)
        print(f"    fragment {frag['name']} ({frag['state']}): {len(df)} rows, "
              f"iter {int(df['_step'].min())}..{int(df['_step'].max())}")
    if not dfs:
        raise RuntimeError(f"{base}: no usable history in any fragment")

    merged = pd.concat(dfs, ignore_index=True).sort_values(["_step", "_frag"])
    n_overlap = merged.duplicated("_step").sum()
    if n_overlap:
        print(f"    merged {len(dfs)} fragments, {n_overlap} overlapping iters "
              f"(continuation wins)")
    merged = merged.drop_duplicates("_step", keep="last").drop(columns="_frag")
    merged = merged.sort_values("_step").reset_index(drop=True)

    with gzip.open(csv_path, "wt") as fh:
        merged.to_csv(fh, index=False)
    meta_path.write_text(json.dumps({
        "base": base,
        "project": cfg.project,
        "columns": columns,
        "fragments": frags,
        "config": config,
    }, indent=2, default=str))


def load_run(cfg: Cfg, base: str) -> tuple[pd.DataFrame, dict, dict]:
    csv_path, meta_path = cache_paths(cfg, base)
    meta = json.loads(meta_path.read_text())
    with gzip.open(csv_path, "rt") as fh:
        df = pd.read_csv(fh)
    return df, meta["config"], meta


# ── checks ───────────────────────────────────────────────────────────────────

def check_run(base: str, df: pd.DataFrame, meta: dict, metric: str) -> dict:
    """Sanity checks on one merged run; returns facts for the summary table."""
    steps = df["_step"].to_numpy()
    final_iter = int(steps.max())
    state = meta["fragments"][-1]["state"]

    diffs = np.diff(steps)
    if len(diffs):
        interval = int(np.min(diffs))
        gaps = np.where(diffs != interval)[0]
        if len(gaps):
            spots = ", ".join(f"{int(steps[i])}->{int(steps[i + 1])}" for i in gaps[:5])
            warn(f"{base}: {len(gaps)} gap(s) in iteration sequence "
                 f"(log interval {interval}): {spots}"
                 + (" ..." if len(gaps) > 5 else ""))
    else:
        interval = 1

    n_nan = int(df[metric].isna().sum())
    if n_nan:
        warn(f"{base}: {n_nan} NaN values in '{metric}'")
    if np.isinf(df[metric].to_numpy(dtype=float)).any():
        warn(f"{base}: inf values in '{metric}'")
    if state == "running":
        warn(f"{base}: still RUNNING (final iter so far: {final_iter})")

    return {"final_iter": final_iter, "state": state, "log_interval": interval,
            "n_fragments": len(meta["fragments"])}


def tail_stat(df: pd.DataFrame, metric: str, tail_iters: int) -> tuple[float, float, int]:
    final = df["_step"].max()
    win = df.loc[df["_step"] > final - tail_iters, metric].dropna()
    return float(win.mean()), float(win.std()), len(win)


def consumed_tokens(df: pd.DataFrame, config: dict, base: str) -> np.ndarray:
    gbs, seq = config.get("global_batch_size"), config.get("seq_length")
    if not gbs or not seq:
        raise RuntimeError(f"{base}: global_batch_size/seq_length missing from wandb config")
    if config.get("rampup_batch_size"):
        warn(f"{base}: rampup_batch_size is set — consumed-tokens = iter*GBS*seq "
             f"over-counts the ramp phase")
    return df["_step"].to_numpy() * gbs * seq


def get_x(config: dict, x_key: str, base: str) -> float:
    if x_key not in config:
        near = [k for k in config if x_key.split("_")[0] in k]
        raise RuntimeError(f"{base}: '{x_key}' not in wandb config "
                           f"(similar keys: {near or 'none'})")
    val = config[x_key]
    if val is None and x_key != "lr" and config.get("lr") is not None:
        # Megatron per-class LR semantics: unset class LR falls back to --lr.
        warn(f"{base}: config '{x_key}' is None, falling back to lr="
             f"{config['lr']:g}")
        val = config["lr"]
    if val is None:
        raise RuntimeError(f"{base}: '{x_key}' is None in wandb config")
    return float(val)


# ── plots ────────────────────────────────────────────────────────────────────

def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)


def plot_sweep(cfg: Cfg, rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for g in cfg.groups:
        pts = sorted((r for r in rows if r["group"] == g.key), key=lambda r: r["x"])
        if not pts:
            continue
        xs = [r["x"] for r in pts]
        ys = [r["tail_mean"] for r in pts]
        (line,) = ax.plot(xs, ys, "-o", label=g.label, color=g.color, ms=5)
        # hollow markers for incomplete runs so they're visibly not comparable
        for r in pts:
            if not r["complete"]:
                ax.plot(r["x"], r["tail_mean"], "o", ms=9, mfc="none",
                        mec=line.get_color(), mew=1.5)
        best = min(pts, key=lambda r: r["tail_mean"])
        ax.plot(best["x"], best["tail_mean"], "*", ms=14, color=line.get_color(),
                zorder=5)
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs="auto"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    style(ax, cfg.x_key.replace("_", "-"),
          f"{cfg.metric} (mean over last {cfg.tail_iters} iters)",
          f"{cfg.tag}: LR sweep (★ best, ○ incomplete)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200)
    print(f"{OK}wrote{RST} {out.with_suffix('.png')}")


def plot_curves(cfg: Cfg, best: list[dict], smooth: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    tail_means = []
    for r in best:
        df, g = r["df"], r["group_obj"]
        y = df[cfg.metric]
        if smooth > 1:
            y = y.rolling(smooth, min_periods=1).mean()
        tok = consumed_tokens(df, r["config"], r["base"]) / 1e9
        ax.plot(tok, y, label=f"{g.label} ({cfg.x_key.split('_')[0]}={r['x']:g})",
                color=g.color, lw=1.2)
        tail_means.append(r["tail_mean"])
    # zoom past the initial loss cliff: cap y at ~3x the spread above the best tail
    if tail_means:
        lo, hi = min(tail_means), max(tail_means)
        ax.set_ylim(lo - 0.05, hi + max(3 * (hi - lo), 0.5))
    style(ax, "consumed tokens (B)",
          cfg.metric + (f" (rolling mean, {smooth})" if smooth > 1 else ""),
          f"{cfg.tag}: best run per group")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200)
    print(f"{OK}wrote{RST} {out.with_suffix('.png')}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path, help="YAML sweep config")
    ap.add_argument("--metric", help="history key for y (default: yaml / 'lm loss')")
    ap.add_argument("--x-key", help="wandb config key for sweep x (default: yaml / matrix_lr)")
    ap.add_argument("--tail-iters", type=int, help="tail window in iterations (default 50)")
    ap.add_argument("--target-iter", type=int,
                    help="expected final iteration (default: max observed)")
    ap.add_argument("--entity", help="wandb entity (default: yaml / API default)")
    ap.add_argument("--out-dir", help="plot output dir (default: yaml)")
    ap.add_argument("--refresh", action="store_true", help="ignore cache, redownload")
    ap.add_argument("--offline", action="store_true", help="no wandb API calls, cache only")
    ap.add_argument("--only", action="append", default=[], metavar="GROUP",
                    help="restrict to these group keys (repeatable)")
    ap.add_argument("--smooth", type=int, default=1,
                    help="rolling-mean window (points) for the curves plot")
    ap.add_argument("--require-complete", action="store_true",
                    help="drop runs that did not reach the target iteration")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--no-curves", action="store_true")
    ap.add_argument("--list", action="store_true",
                    help="list merged run names in the project and exit")
    args = ap.parse_args()

    cfg = load_cfg(args.config, args)
    if args.only:
        cfg.groups = [g for g in cfg.groups if g.key in args.only]
    columns = sorted({cfg.metric, *cfg.extra_history_keys})

    api = None
    live: dict[str, list[dict]] = {}
    if not args.offline:
        import wandb
        api = wandb.Api(timeout=120)
        cfg.entity = cfg.entity or api.default_entity
        print(f"querying runs in {cfg.entity}/{cfg.project} "
              f"{'(full listing)' if args.list else '(filtered by group patterns)'} ...")
        live = list_fragments(api, cfg, groups=None if args.list else cfg.groups)
        print(f"  {sum(len(v) for v in live.values())} wandb runs "
              f"-> {len(live)} merged runs")
    elif cfg.entity is None:
        cfg.entity = "offline"

    if args.list:
        for base, frags in sorted(live.items()):
            states = ",".join(f["state"][0] for f in frags)
            print(f"{base}  [{len(frags)} frag {states}] last_step="
                  f"{frags[-1]['last_step']}")
        return 0

    # group key -> base names
    resolved: list[tuple[Group, str]] = []
    for g in cfg.groups:
        bases = list(g.runs)
        if g.regex:
            pat = re.compile(g.regex)
            hits = sorted(b for b in live if pat.fullmatch(b))
            if not hits and not args.offline:
                warn(f"group '{g.key}': regex matched no runs: {g.regex}")
            bases += [b for b in hits if b not in bases]
        if args.offline and g.regex:
            # match against whatever is in the cache
            pat = re.compile(g.regex)
            cached = (cfg.cache_dir / cfg.project).glob("*.meta.json")
            bases += sorted(p.name[:-len(".meta.json")] for p in cached
                            if pat.fullmatch(p.name[:-len(".meta.json")])
                            and p.name[:-len(".meta.json")] not in bases)
        if not bases:
            warn(f"group '{g.key}': no runs resolved")
        resolved += [(g, b) for b in bases]

    # ── fetch / cache ──
    rows = []
    for g, base in resolved:
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
        x_key = g.x_key or cfg.x_key
        try:
            x = get_x(config, x_key, base)
        except RuntimeError as e:
            warn(str(e) + " — skipping")
            continue
        mean, std, n = tail_stat(df, cfg.metric, cfg.tail_iters)
        rows.append({"group": g.key, "group_obj": g, "base": base, "x": x,
                     "tail_mean": mean, "tail_std": std, "tail_n": n,
                     "df": df, "config": config, **facts})

    if not rows:
        print(f"{ERR}no runs to plot{RST}")
        return 1

    # ── completion check: all runs should end at the same iteration ──
    target = cfg.target_iter or max(r["final_iter"] for r in rows)
    for r in rows:
        r["complete"] = r["final_iter"] >= target
    incomplete = [r for r in rows if not r["complete"]]
    if incomplete:
        warn(f"{len(incomplete)}/{len(rows)} runs did NOT reach iter {target} "
             f"(tail losses not at equal tokens!):")
        for r in sorted(incomplete, key=lambda r: r["final_iter"]):
            print(f"    {r['base']}: iter {r['final_iter']} ({r['state']}, "
                  f"{r['n_fragments']} fragments)")
        if args.require_complete:
            rows = [r for r in rows if r["complete"]]
            print(f"  --require-complete: dropped {len(incomplete)} runs")
    else:
        print(f"{OK}all {len(rows)} runs reached iter {target}{RST}")

    # duplicate x within a group (sloppy regex / re-launched config)
    for g in cfg.groups:
        seen: dict[float, str] = {}
        for r in [r for r in rows if r["group"] == g.key]:
            if r["x"] in seen:
                warn(f"group '{g.key}': duplicate x={r['x']:g} "
                     f"({seen[r['x']]} vs {r['base']}) — both plotted")
            seen[r["x"]] = r["base"]

    # ── summary table + csv ──
    table = pd.DataFrame([{k: r[k] for k in
                           ("group", "base", "x", "tail_mean", "tail_std", "tail_n",
                            "final_iter", "complete", "state", "n_fragments")}
                          for r in rows]).sort_values(["group", "x"])
    pd.set_option("display.width", 200)
    print("\n" + table.to_string(index=False,
                                 float_format=lambda v: f"{v:.5g}") + "\n")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = cfg.out_dir / f"{cfg.tag}-summary.csv"
    table.to_csv(csv_out, index=False)
    print(f"{OK}wrote{RST} {csv_out}")

    # ── plots ──
    if not args.no_sweep:
        plot_sweep(cfg, rows, cfg.out_dir / f"{cfg.tag}-sweep")
    if not args.no_curves:
        best = []
        for g in cfg.groups:
            cand = [r for r in rows if r["group"] == g.key]
            if not cand:
                continue
            pool = [r for r in cand if r["complete"]]
            if not pool:
                warn(f"group '{g.key}': no complete run, best-run pick uses incomplete runs")
                pool = cand
            best.append(min(pool, key=lambda r: r["tail_mean"]))
        if best:
            plot_curves(cfg, best, args.smooth, cfg.out_dir / f"{cfg.tag}-curves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
