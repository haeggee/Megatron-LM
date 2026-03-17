"""Visualize weight matrices from a Megatron-LM checkpoint as heatmaps.

Loads a checkpoint (torch_dist or legacy torch format) and plots Q, K, V,
Out projection, Gate, Up, Down for selected transformer layers.

Usage:
    python scripts/visualize_weight_heatmaps.py \
        --checkpoint-path /path/to/iter_XXXXXX \
        --output weight_heatmaps.png
"""

import argparse
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Checkpoint format detection and loading
# ---------------------------------------------------------------------------

def detect_checkpoint_format(checkpoint_path):
    if os.path.exists(os.path.join(checkpoint_path, "metadata.json")):
        return "torch_dist"
    if any(f.startswith("mp_rank_0") for f in os.listdir(checkpoint_path)):
        return "torch"
    if os.path.exists(os.path.join(checkpoint_path, ".metadata")):
        return "torch_dcp"
    raise ValueError(f"Unknown checkpoint format in {checkpoint_path}")


def load_args_from_common_pt(checkpoint_path):
    common_path = os.path.join(checkpoint_path, "common.pt")
    if not os.path.exists(common_path):
        return None
    common_state = torch.load(common_path, map_location="cpu", weights_only=False)
    return common_state.get("args", None)


def load_state_dict_torch_dist(checkpoint_path):
    from torch.distributed.checkpoint import FileSystemReader, DefaultLoadPlanner
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata, BytesStorageMetadata
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    class _PatchedPlanner(DefaultLoadPlanner):
        """DefaultLoadPlanner crashes when metadata.planner_data is None
        (common with Megatron torch_dist checkpoints). Patch it."""
        def set_up_planner(self, state_dict, metadata, is_coordinator=False):
            if metadata.planner_data is None:
                metadata.planner_data = {}
            return super().set_up_planner(state_dict, metadata, is_coordinator)

    reader = FileSystemReader(checkpoint_path)
    metadata = reader.read_metadata()

    state_dict = {}
    for key, value in metadata.state_dict_metadata.items():
        if isinstance(value, TensorStorageMetadata):
            state_dict[key] = torch.empty(tuple(value.size), dtype=value.properties.dtype)
        elif isinstance(value, BytesStorageMetadata):
            state_dict[key] = "<bytes_io>"

    _load_state_dict(
        state_dict,
        storage_reader=reader,
        planner=_PatchedPlanner(),
        no_dist=True,
    )
    return state_dict


def load_legacy_checkpoint(checkpoint_path):
    for fname in sorted(os.listdir(checkpoint_path)):
        if fname.startswith("mp_rank_00"):
            ckpt_file = os.path.join(checkpoint_path, fname, "model_optim_rng.pt")
            if os.path.exists(ckpt_file):
                return torch.load(ckpt_file, map_location="cpu", weights_only=False)
    raise FileNotFoundError(f"No mp_rank_00 checkpoint found in {checkpoint_path}")


# ---------------------------------------------------------------------------
# Model config extraction
# ---------------------------------------------------------------------------

def extract_model_config(args):
    num_layers = getattr(args, "num_layers", None) or getattr(args, "encoder_num_layers", None)
    num_attention_heads = args.num_attention_heads
    hidden_size = args.hidden_size
    kv_channels = getattr(args, "kv_channels", None) or (hidden_size // num_attention_heads)
    num_query_groups = getattr(args, "num_query_groups", None) or num_attention_heads
    ffn_hidden_size = getattr(args, "ffn_hidden_size", None)
    swiglu = getattr(args, "swiglu", False)

    if ffn_hidden_size is None:
        if swiglu:
            ffn_hidden_size = int((4 * hidden_size * 2 / 3) / 64) * 64
        else:
            ffn_hidden_size = 4 * hidden_size

    return {
        "num_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "hidden_size": hidden_size,
        "kv_channels": kv_channels,
        "num_query_groups": num_query_groups,
        "ffn_hidden_size": ffn_hidden_size,
        "swiglu": swiglu,
    }


# ---------------------------------------------------------------------------
# Key detection and weight access
# ---------------------------------------------------------------------------

# Weight suffixes (relative to decoder.layers or decoder.layers.{i})
WEIGHT_KEYS = {
    "qkv":  "self_attention.linear_qkv.weight",
    "proj": "self_attention.linear_proj.weight",
    "fc1":  "mlp.linear_fc1.weight",
    "fc2":  "mlp.linear_fc2.weight",
}


def detect_key_format(state_dict):
    """Detect whether layers are stacked (no index) or indexed (per-layer keys).

    Returns:
        (prefix, stacked): prefix is the string before 'decoder.layers.',
                           stacked is True if keys lack per-layer indices.
    """
    for key in state_dict:
        idx = key.find("decoder.layers.")
        if idx < 0:
            continue
        prefix = key[:idx]
        after = key[idx + len("decoder.layers."):]
        # Check if next part is a digit (indexed) or not (stacked)
        if re.match(r"\d+\.", after):
            return prefix, False  # indexed: decoder.layers.0.xxx
        else:
            return prefix, True   # stacked: decoder.layers.xxx (tensor[layer_dim])
    raise KeyError("Could not find 'decoder.layers.' in state dict keys")


def get_layer_weight(state_dict, prefix, layer_idx, weight_suffix, stacked):
    """Get a 2D weight matrix for a specific layer.

    For stacked format: key is 'prefix + decoder.layers.{suffix}', tensor is [num_layers, ...]
    For indexed format: key is 'prefix + decoder.layers.{layer_idx}.{suffix}'
    """
    if stacked:
        full_key = f"{prefix}decoder.layers.{weight_suffix}"
        if full_key not in state_dict:
            raise KeyError(f"Key {full_key} not found")
        return state_dict[full_key][layer_idx]
    else:
        full_key = f"{prefix}decoder.layers.{layer_idx}.{weight_suffix}"
        if full_key not in state_dict:
            raise KeyError(f"Key {full_key} not found")
        return state_dict[full_key]


def infer_num_layers(state_dict, prefix, stacked):
    """Infer number of layers from state dict."""
    if stacked:
        qkv_key = f"{prefix}decoder.layers.{WEIGHT_KEYS['qkv']}"
        if qkv_key in state_dict:
            return state_dict[qkv_key].shape[0]
    else:
        indices = set()
        pattern = re.compile(re.escape(prefix) + r"decoder\.layers\.(\d+)\.")
        for key in state_dict:
            m = pattern.search(key)
            if m:
                indices.add(int(m.group(1)))
        if indices:
            return max(indices) + 1
    return 0


# ---------------------------------------------------------------------------
# Weight splitting
# ---------------------------------------------------------------------------

def split_qkv(qkv_weight, config):
    """Split fused QKV weight [qkv_out, hidden] into Q, K, V."""
    num_heads = config["num_attention_heads"]
    num_groups = config["num_query_groups"]
    kv_channels = config["kv_channels"]
    hidden_size = qkv_weight.shape[-1]

    heads_per_group = num_heads // num_groups
    q_per_group = heads_per_group * kv_channels
    group_size = q_per_group + 2 * kv_channels

    reshaped = qkv_weight.view(num_groups, group_size, hidden_size)
    q = reshaped[:, :q_per_group, :].reshape(-1, hidden_size)
    k = reshaped[:, q_per_group:q_per_group + kv_channels, :].reshape(-1, hidden_size)
    v = reshaped[:, q_per_group + kv_channels:, :].reshape(-1, hidden_size)
    return q, k, v


def split_gate_up(fc1_weight):
    """Split fused gate+up weight [2*ffn, hidden] into gate, up."""
    return torch.chunk(fc1_weight, 2, dim=0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def subsample(w, max_size):
    rows, cols = w.shape
    rs = max(1, rows // max_size)
    cs = max(1, cols // max_size)
    return w[::rs, ::cs]


def create_figure(weight_dict, layer_indices, args):
    col_names = ["Q", "K", "V", "Out", "Gate", "Up", "Down"]
    n_rows = len(layer_indices)
    n_cols = len(col_names)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, layer_idx in enumerate(layer_indices):
        for col, name in enumerate(col_names):
            ax = axes[row, col]
            tensor = weight_dict.get((layer_idx, name))

            if tensor is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                w = tensor.float().numpy()
                w = subsample(w, args.subsample)
                vmax = np.percentile(np.abs(w), args.vmax_percentile)
                if vmax == 0:
                    vmax = 1.0
                im = ax.imshow(w, aspect="auto", cmap=args.cmap,
                               vmin=-vmax, vmax=vmax, interpolation="nearest")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if row == 0:
                ax.set_title(name, fontsize=13, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"Layer {layer_idx}", fontsize=12, fontweight="bold")

    ckpt_name = os.path.basename(os.path.normpath(args.checkpoint_path))
    fig.suptitle(f"Weight Heatmaps — {ckpt_name}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    png_name = args.output.split(".")[0] + ".png"
    fig.savefig(png_name, dpi=150, bbox_inches="tight")
    pdf_name = args.output.split(".")[0] + ".pdf"
    fig.savefig(pdf_name, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {args.output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize Megatron-LM weight heatmaps")
    parser.add_argument("--checkpoint-path", type=str, required=True,
                        help="Path to checkpoint directory (e.g. .../iter_0001000)")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (default: 0, n//2, n-1)")
    parser.add_argument("--output", type=str, default="weight_heatmaps.png",
                        help="Output file path")
    parser.add_argument("--subsample", type=int, default=256,
                        help="Max display resolution per axis (default: 256)")
    parser.add_argument("--cmap", type=str, default="RdBu_r",
                        help="Matplotlib colormap (default: RdBu_r)")
    parser.add_argument("--vmax-percentile", type=float, default=99.0,
                        help="Percentile for symmetric color scale (default: 99)")
    args = parser.parse_args()

    # Detect format
    fmt = detect_checkpoint_format(args.checkpoint_path)
    print(f"Detected format: {fmt}")

    # Load checkpoint
    if fmt == "torch_dist":
        ckpt_args = load_args_from_common_pt(args.checkpoint_path)
        print("Loading state dict (may take a moment)...")
        state_dict = load_state_dict_torch_dist(args.checkpoint_path)
    elif fmt == "torch":
        full_state = load_legacy_checkpoint(args.checkpoint_path)
        ckpt_args = full_state.get("args", None)
        state_dict = full_state.get("model", full_state)
    else:
        raise NotImplementedError(f"Format {fmt} not yet supported")

    # Detect key format (stacked vs indexed)
    prefix, stacked = detect_key_format(state_dict)
    print(f"Key prefix: '{prefix}', stacked layers: {stacked}")

    # Extract model config
    if ckpt_args is not None:
        config = extract_model_config(ckpt_args)
    else:
        print("No checkpoint args found, inferring from tensor shapes...")
        # For stacked: qkv shape is [num_layers, qkv_out, hidden]
        qkv_key = f"{prefix}decoder.layers.{WEIGHT_KEYS['qkv']}" if stacked else f"{prefix}decoder.layers.0.{WEIGHT_KEYS['qkv']}"
        qkv_shape = state_dict[qkv_key].shape
        hidden_size = qkv_shape[-1]
        qkv_out = qkv_shape[-2]
        # Rough inference
        kv_channels = 128
        if qkv_out == 3 * hidden_size:
            num_heads = hidden_size // kv_channels
            num_groups = num_heads
        else:
            num_groups = (qkv_out - hidden_size) // (2 * kv_channels)
            num_heads = hidden_size // kv_channels

        fc1_key = f"{prefix}decoder.layers.{WEIGHT_KEYS['fc1']}" if stacked else f"{prefix}decoder.layers.0.{WEIGHT_KEYS['fc1']}"
        fc2_key = f"{prefix}decoder.layers.{WEIGHT_KEYS['fc2']}" if stacked else f"{prefix}decoder.layers.0.{WEIGHT_KEYS['fc2']}"
        fc1_out = state_dict[fc1_key].shape[-2]
        fc2_out = state_dict[fc2_key].shape[-2]
        swiglu = fc1_out == 2 * fc2_out
        ffn_hidden_size = fc2_out if swiglu else fc1_out

        config = {
            "num_layers": infer_num_layers(state_dict, prefix, stacked),
            "num_attention_heads": num_heads,
            "hidden_size": hidden_size,
            "kv_channels": kv_channels,
            "num_query_groups": num_groups,
            "ffn_hidden_size": ffn_hidden_size,
            "swiglu": swiglu,
        }

    # Infer num_layers from state dict if not in args
    if config["num_layers"] is None or config["num_layers"] == 0:
        config["num_layers"] = infer_num_layers(state_dict, prefix, stacked)

    print(f"Model config: {config}")

    # Determine layers to plot
    num_layers = config["num_layers"]
    if args.layers is not None:
        layer_indices = [int(x.strip()) for x in args.layers.split(",")]
    else:
        layer_indices = sorted(set([0, num_layers // 2, num_layers - 1]))
    layer_indices = [i for i in layer_indices if 0 <= i < num_layers]
    print(f"Plotting layers: {layer_indices}")

    # Extract and split weights
    weight_dict = {}
    for li in layer_indices:
        # Attention QKV
        try:
            qkv_w = get_layer_weight(state_dict, prefix, li, WEIGHT_KEYS["qkv"], stacked)
            q, k, v = split_qkv(qkv_w, config)
            weight_dict[(li, "Q")] = q
            weight_dict[(li, "K")] = k
            weight_dict[(li, "V")] = v
        except KeyError as e:
            print(f"Warning: {e}")

        # Attention output projection
        try:
            weight_dict[(li, "Out")] = get_layer_weight(state_dict, prefix, li, WEIGHT_KEYS["proj"], stacked)
        except KeyError as e:
            print(f"Warning: {e}")

        # MLP fc1 (gate + up for SwiGLU, or just fc1)
        try:
            fc1_w = get_layer_weight(state_dict, prefix, li, WEIGHT_KEYS["fc1"], stacked)
            if config["swiglu"]:
                gate, up = split_gate_up(fc1_w)
                weight_dict[(li, "Gate")] = gate
                weight_dict[(li, "Up")] = up
            else:
                weight_dict[(li, "Gate")] = fc1_w
                weight_dict[(li, "Up")] = None
        except KeyError as e:
            print(f"Warning: {e}")

        # MLP fc2 (down projection)
        try:
            weight_dict[(li, "Down")] = get_layer_weight(state_dict, prefix, li, WEIGHT_KEYS["fc2"], stacked)
        except KeyError as e:
            print(f"Warning: {e}")

    # Plot
    create_figure(weight_dict, layer_indices, args)


if __name__ == "__main__":
    main()
