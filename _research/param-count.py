#!/usr/bin/env python3
"""param-count.py — parameter count + MoE sparsity for a composed config.

Sparsity here = activated params / total params, the number that defines an
MoE point (e.g. the 670B-A40B target is ~6%). We report it two ways:

  full        — includes the (always-active) untied embedding slab.
  non-embed   — excludes embeddings. This is the scale-invariant, clean number:
                with a 131k vocab the untied embeddings are a fixed slab that
                dwarfs everything at small scale, so the *full* ratio is wildly
                scale-dependent (35% at 175m vs 6% at 670B) while non-embed
                stays put. Design iso-sparsity ablations against non-embed.

Two ways to use it:

  # 1. count exactly what the framework composes (DRY — never drifts):
  SIZE=350m-moe RECIPE=master bash launch/framework/lib/dump-args.sh \
      | python3 _research/param-count.py

  # 2. the 670B-A40B reference target the ablations are standing in for:
  python3 _research/param-count.py --target

  # ad-hoc what-if (any unset field falls back to the target's value):
  python3 _research/param-count.py --layers 24 --hidden 1024 --experts 64 --topk 1

Caveats: counts linear weights only (attention qkv/o, MLP gate/up/down, router,
embeddings). Biases, layernorms and router-aux params are <0.1% and ignored.
Vocab defaults to 131072 (the swissai tokenizer, padded /128); override with
--vocab. moe-layer-freq is honoured (int period or an `[0]*3+[1]*58`-style
list expression) to count dense-vs-MoE layers.
"""
import argparse
import sys

DEFAULT_VOCAB = 131072  # alehc/swissai-tokenizer, padded to /128

# The reference target the small-scale ablations stand in for: a DeepSeek-V3
# shaped MoE-670B-A40B (61L, first 3 dense then 58 MoE, 128 experts top-4).
TARGET = dict(layers=61, hidden=7168, heads=128, kv_groups=128, ffn=16384,
              experts=128, topk=4, moe_ffn=4096, shared_ffn=2048,
              moe_layer_freq="[0]*3+[1]*58", swiglu=True, untied=True,
              vocab=DEFAULT_VOCAB, head_dim=None)


def mlp(h, ffn, swiglu):
    # SwiGLU has a gate matrix on top of up+down; plain MLP is just up+down.
    return (3 if swiglu else 2) * h * ffn


def attn(h, heads, kv_groups, head_dim):
    hd = head_dim or h // heads
    q = h * (heads * hd)
    kv = 2 * h * (kv_groups * hd)
    o = (heads * hd) * h
    return q + kv + o


def moe_layer_mask(freq, layers):
    """Return a list[0/1] of length `layers`: 1 = MoE layer, 0 = dense.

    Mirrors Megatron: a string is eval'd to a list, an int N makes every Nth
    layer MoE, absent means every layer is MoE.
    """
    if freq is None:
        return [1] * layers
    if isinstance(freq, int):
        return [1 if (i + 1) % freq == 0 else 0 for i in range(layers)]
    expr = freq.strip().lstrip("(").rstrip(")")          # tolerate bash '\(...\)'
    mask = eval(expr, {"__builtins__": {}}, {})           # e.g. [0]*3+[1]*58
    if len(mask) != layers:
        sys.exit(f"moe-layer-freq expands to {len(mask)} entries but --layers={layers}")
    return [int(x) for x in mask]


def count(cfg):
    h, sw = cfg["hidden"], cfg["swiglu"]
    emb = (2 if cfg["untied"] else 1) * cfg["vocab"] * h
    a = attn(h, cfg["heads"], cfg["kv_groups"], cfg["head_dim"])

    experts = cfg["experts"] or 0
    is_moe = experts > 0
    mask = moe_layer_mask(cfg["moe_layer_freq"], cfg["layers"]) if is_moe else [0] * cfg["layers"]

    dense_mlp = mlp(h, cfg["ffn"], sw)
    if is_moe:
        router = h * experts
        moe_ffn = cfg["moe_ffn"] or cfg["ffn"]
        shared = cfg["shared_ffn"] or 0
        moe_total = router + experts * mlp(h, moe_ffn, sw) + (mlp(h, shared, sw) if shared else 0)
        moe_active = router + cfg["topk"] * mlp(h, moe_ffn, sw) + (mlp(h, shared, sw) if shared else 0)
    else:
        moe_total = moe_active = dense_mlp

    backbone = cfg["layers"] * a
    total = emb + backbone + sum(moe_total if m else dense_mlp for m in mask)
    active = emb + backbone + sum(moe_active if m else dense_mlp for m in mask)
    return dict(emb=emb, total=total, active=active,
                total_ne=total - emb, active_ne=active - emb,
                n_moe=sum(mask), n_dense=cfg["layers"] - sum(mask))


def report(name, cfg):
    r = count(cfg)
    sp = 100 * r["active"] / r["total"]
    sp_ne = 100 * r["active_ne"] / r["total_ne"]
    moe = (f"E={cfg['experts']} top{cfg['topk']} moe_ffn={cfg['moe_ffn']} "
           f"shared={cfg['shared_ffn']} dense_layers={r['n_dense']}") if cfg["experts"] else "dense"
    print(f"=== {name}: L={cfg['layers']} h={cfg['hidden']} {moe}")
    print(f"  embed(untied={cfg['untied']}) = {r['emb']/1e6:9.1f}M   (vocab={cfg['vocab']})")
    print(f"  TOTAL      = {r['total']/1e9:8.3f}B    active = {r['active']/1e9:8.3f}B"
          f"    sparsity = {sp:5.2f}%")
    print(f"  non-embed  = {r['total_ne']/1e9:8.3f}B    active = {r['active_ne']/1e9:8.3f}B"
          f"    sparsity = {sp_ne:5.2f}%   <- the scale-invariant number")
    print()
    return sp_ne


def parse_megatron_args(tokens):
    """Pull the count-relevant flags out of a composed MEGATRON_ARGS list."""
    def val(flag, conv=int, default=None):
        return conv(tokens[tokens.index(flag) + 1]) if flag in tokens else default
    return dict(
        layers=val("--num-layers"), hidden=val("--hidden-size"),
        heads=val("--num-attention-heads"),
        kv_groups=val("--num-query-groups") or val("--num-attention-heads"),
        ffn=val("--ffn-hidden-size"), head_dim=val("--kv-channels"),
        experts=val("--num-experts", default=0), topk=val("--moe-router-topk", default=1),
        moe_ffn=val("--moe-ffn-hidden-size"),
        shared_ffn=val("--moe-shared-expert-intermediate-size", default=0),
        moe_layer_freq=val("--moe-layer-freq", conv=str),
        swiglu="--swiglu" in tokens,
        untied="--untie-embeddings-and-output-weights" in tokens,
        vocab=DEFAULT_VOCAB,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", action="store_true", help="report the 670B-A40B reference target")
    p.add_argument("--vocab", type=int, help=f"override vocab size (default {DEFAULT_VOCAB})")
    for f in ("layers", "hidden", "heads", "kv-groups", "ffn", "experts", "topk",
              "moe-ffn", "shared-ffn", "head-dim"):
        p.add_argument(f"--{f}", type=int)
    p.add_argument("--moe-layer-freq")
    p.add_argument("--name", default="config")
    args = p.parse_args()

    if args.target:
        cfg = dict(TARGET)
        if args.vocab:
            cfg["vocab"] = args.vocab
        report("TARGET 670B-A40B", cfg)
        return

    # Base config: composed args if something was actually piped in, else the
    # target (so ad-hoc `--layers .. --experts ..` is a what-if vs the target).
    # Key on real stdin content, not isatty() — under a script/cron/agent stdin
    # is never a tty even when nothing is piped.
    piped = "" if sys.stdin.isatty() else sys.stdin.read()
    cfg = parse_megatron_args(piped.split()) if piped.strip() else dict(TARGET)

    # Any explicit CLI flag overrides the base.
    over = {k.replace("-", "_"): v for k, v in vars(args).items()
            if v is not None and k not in ("target", "name", "kv_groups")}
    if args.kv_groups is not None:
        over["kv_groups"] = args.kv_groups
    # moe-layer-freq is coupled to layer count: if a what-if changes --layers but
    # doesn't restate --moe-layer-freq, drop the base's freq (else it mismatches).
    if "layers" in over and args.moe_layer_freq is None:
        cfg = dict(cfg, moe_layer_freq=None)
    cfg.update(over)
    cfg.setdefault("head_dim", None)
    cfg.setdefault("swiglu", True)
    cfg.setdefault("untied", True)
    cfg.setdefault("vocab", DEFAULT_VOCAB)

    sp_ne = report(args.name, cfg)
    report("TARGET 670B-A40B", dict(TARGET, vocab=cfg["vocab"]))
    print(f"non-embed sparsity: {args.name}={sp_ne:.2f}%  vs  target=5.47%")


if __name__ == "__main__":
    main()
