# adapted from megatron/core/optimizer/muon.py and https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/b8365dbdce94a979090af735698fabc6be497f06/emerging_optimizers/orthogonalized_optimizers/orthogonalized_optimizer.py.
import math
import logging
from typing import Callable,Optional, Literal, override

import torch

from . import _get_param_groups, get_megatron_optimizer
from .ademamix import linear_hl_warmup_scheduler, linear_warmup_scheduler
from .layer_wise_optimizer import LayerWiseDistributedOptimizer
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from .optimizer_config import OptimizerConfig, ParamKey, ParamPredicate
from megatron.core.transformer.module import MegatronModule
from megatron.core.optimizer_param_scheduler import ParamGroupOverride
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_size, log_single_rank
from megatron.core.optimizer.muon import get_muon_scale_factor


logger = logging.getLogger(__name__)


try:
    import emerging_optimizers
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz_tp
except ImportError:
    emerging_optimizers = None


class MasterOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        # Common settings.
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        weight_decay_method: Literal["decoupled", "independent"]  = "decoupled",

        # adam & ademamix settings.
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 2.0,
        beta3_warmup: Optional[int] = None,
        alpha_warmup: Optional[int] = None,
        eps: float = 1e-8,

        # Hypersphere optimization.
        hypersphere_mode: Optional[Literal["row", "col", "rowcol", "invrowcol", "equi", "flat", "embed"]] = None,
        hypersphere_embedding_mode: Optional[Literal["row", "col", "rowcol", "invrowcol", "equi", "flat"]] = None,
        hypersphere_kind: Optional[Literal["l2", "standard", "spectral", "orthogonal"]] = None,
        hypersphere_radius: Literal["learnable"] | float = 1.0,
        hypersphere_eps: float = 1e-8,
        hypersphere_update: bool = True,
        hypersphere_update_embeddings: bool = True,
        hypersphere_project: bool = False,
        hypersphere_soft: bool = False,

        # Muon.
        use_orthogonal_updates: bool = False,  # Enable or disable muon entirely.
        poor_mans_ortho: bool = False,  # Use _normalize instead of _orthogonalize in the Muon branch.
        momentum_beta: float = 0.95,
        use_nesterov: bool = True,
        split_qkv: bool = True,  # Also applies to hypersphere optimization.
        split_qkv_heads: bool = True,  # only applies to hypersphere of weights
        split_qkv_heads_update: bool = True,  # only applies to hypersphere of updates
        qkv_split_shapes: Optional[tuple[int, int, int]] = None,
        qkv_dim: Optional[int] = None,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
        preserve_init: bool = False,
    ):

        self.preserve_init = preserve_init
        self.fp32_matmul_prec = fp32_matmul_prec
        self.use_nesterov = use_nesterov
        self.weight_decay_method = weight_decay_method

        self.hypersphere_mode = hypersphere_mode
        self.hypersphere_embedding_mode = hypersphere_embedding_mode
        self.hypersphere_kind = hypersphere_kind
        self.hypersphere_radius = hypersphere_radius
        self.hypersphere_eps = hypersphere_eps
        self.hypersphere_update = hypersphere_update
        self.hypersphere_update_embeddings = hypersphere_update_embeddings
        self.hypersphere_project = hypersphere_project
        self.hypersphere_soft = hypersphere_soft

        self.split_qkv = split_qkv
        self.split_qkv_heads  = split_qkv_heads
        self.split_qkv_heads_update = split_qkv_heads_update
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes
        self.qkv_dim = qkv_dim

        self.poor_mans_ortho = poor_mans_ortho

        self.coefficient_type = coefficient_type
        self.num_ns_steps = num_ns_steps
        self.scale_mode = scale_mode
        self.extra_scale_factor = extra_scale_factor

        self.pg_collection = pg_collection
        self.mode = mode

        self._param_update_hook = None  # set by InternalsLogger; called as hook(p, update, is_qkv)

        default_args_dict = dict(
            lr=lr,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,

            beta1=betas[0],
            beta2=betas[1],
            beta3=betas[2],
            momentum_beta=momentum_beta,
            alpha=alpha,
            step=0,
            beta3_warmup=beta3_warmup,
            alpha_warmup=alpha_warmup,
            eps=eps,

            use_orthogonal_updates=use_orthogonal_updates,
        )
        super().__init__(params, default_args_dict)

        # Normalize parameters at initialization so the first forward pass
        # uses weights that are already on the hypersphere.
        if self.hypersphere_mode is not None and not self.preserve_init:
            with torch.no_grad():
                for group in self.param_groups:
                    for p in group["params"]:
                        is_qkv = self.is_qkv_fn(p)
                        is_out_proj = getattr(p, "is_out_proj", False)
                        is_embedding = getattr(p, "is_embedding_or_output_parameter", False)
                        self._normalize(p, p, is_qkv=is_qkv, is_out_proj=is_out_proj, is_embedding=is_embedding)

    @torch.no_grad()  # type: ignore[misc]
    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        if closure is None:
            loss = None
        else:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step"] += 1
            for p in group["params"]:
                if p.grad is not None:
                    self._param_step(p, group)

        return loss


    def _param_step(self, p, group):
        grad = p.grad
        state = self.state[p]

        # Initialization.
        if "exp_avg" not in state:  # row_gain could be already in state if we are using learnable gains.
            state["exp_avg"] = torch.zeros_like(grad)
            # TODO: Make it such that we can use ademamix-like updates with muon.
            if not group["use_orthogonal_updates"]: 
                if group["beta2"] != 0: # Enables g^2 EMA as in adam & ademamix.
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                if group["alpha"] != 0:  # Enables slow momentum as in ademamix.
                    state["exp_avg_slow"] = torch.zeros_like(grad)

        exp_avg = state["exp_avg"]
        beta1 = group["beta1"]
        momentum_beta = group["momentum_beta"]
        is_qkv = self.is_qkv_fn(p)
        is_out_proj = getattr(p, "is_out_proj", False)
        is_embedding = getattr(p, "is_embedding_or_output_parameter", False)

        # TODO: potentially project gradient to tangent space here.
        if self.hypersphere_project:
            grad = self._project(p, grad, is_qkv=is_qkv, is_out_proj=is_out_proj, is_embedding=is_embedding)

        # Get update direction.
        if group["use_orthogonal_updates"]:  # Muon branch.
            assert emerging_optimizers is not None

            # Weight deacy.
            self._apply_weight_decay_inplace(p, group)

            # Update momentum buffer with EMA of gradient
            exp_avg.lerp_(grad, 1 - momentum_beta)

            # Include nesterov momentum
            if self.use_nesterov:
                grad = grad.lerp(exp_avg, momentum_beta)
            else:
                grad = exp_avg

            # Get update.
            if self.poor_mans_ortho:
                update = grad.clone()
                self._normalize(p, update, is_qkv=is_qkv, is_out_proj=is_out_proj, apply_muon_scaling=True)
            else:
                with emerging_optimizers.utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    group_kwargs = {k: v for k, v in group.items() if k != "params"}
                    update = self.orthogonalize(p, grad, **group_kwargs, is_qkv=is_qkv)

        else: # AdamW & Ademamix branch.
            beta2 = group["beta2"]

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            bias_correction1 = 1.0 - (beta1 ** group["step"])

            if beta2 == 0:
                # No second moment: use first momentum directly.
                if group["alpha"] == 0: # adam
                    update = exp_avg / bias_correction1
                else: # ademamix
                    if group["alpha_warmup"] is None:
                        alpha = group["alpha"]
                    else:
                        alpha = linear_warmup_scheduler(group["step"], group["alpha"], alpha_start=0, warmup=group["alpha_warmup"])

                    if group["beta3_warmup"] is None:
                        beta3 = group["beta3"]
                    else:
                        beta3 = linear_hl_warmup_scheduler(group["step"], group["beta3"], beta_start=beta1, warmup=group["beta3_warmup"])

                    exp_avg_slow = state["exp_avg_slow"]
                    exp_avg_slow.mul_(beta3).add_(grad, alpha=1 - beta3)
                    update = exp_avg / bias_correction1 + alpha * exp_avg_slow
            else:
                exp_avg_sq = state["exp_avg_sq"]
                bias_correction2 = 1.0 - (beta2 ** group["step"])

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])

                if group["alpha"] == 0:  # adam logic.
                    update = exp_avg.div(bias_correction1) / denom  # TODO original equation.
                    #update = exp_avg.div(bias_correction1 * denom)  # TODO is this equivalent?
                else:  # ademamix logic.
                    if group["alpha_warmup"] is None:
                        alpha = group["alpha"]
                    else:
                        alpha = linear_warmup_scheduler(group["step"], group["alpha"], alpha_start=0, warmup=group["alpha_warmup"])

                    if group["beta3_warmup"] is None:
                        beta3 = group["beta3"]
                    else:
                        beta3 = linear_hl_warmup_scheduler(group["step"], group["beta3"], beta_start=beta1, warmup=group["beta3_warmup"])

                    exp_avg_slow = state["exp_avg_slow"]
                    exp_avg_slow.mul_(beta3).add_(grad, alpha=1 - beta3)
                    update = (exp_avg.div(bias_correction1) + alpha * exp_avg_slow) / denom  # TODO Original equation.
                    #update = exp_avg.div(bias_correction1).add_(exp_avg_slow, alpha=alpha).div_(denom)  # TODO is this equivalent?

            self._apply_weight_decay_inplace(p, group)

        # Optionally, normalize update.
        apply_update_norm = self.hypersphere_update and (self.hypersphere_update_embeddings or not is_embedding)
        if self.hypersphere_mode is not None and apply_update_norm:
            self._normalize(p, update, is_qkv=is_qkv, is_out_proj=is_out_proj, is_embedding=is_embedding)

        if self._param_update_hook is not None:
            self._param_update_hook(p, update, is_qkv)

        # Update parameter.
        lr = group["lr"]
        p.add_(update, alpha=-lr)

        # Optionally, normalize parameter.
        if self.hypersphere_mode is not None:
            self._normalize(p, p, is_qkv=is_qkv, is_out_proj=is_out_proj, is_embedding=is_embedding)

    def _apply_weight_decay_inplace(self, p, group):
        weight_decay = group["weight_decay"]
        lr = group["lr"]
        if weight_decay != 0:
            weight_decay_method = group["weight_decay_method"]
            if weight_decay_method == "decoupled":
                p.add_(p, alpha=-weight_decay*lr)
            elif weight_decay_method == "independent":
                p.add_(p, alpha=-weight_decay)
            else:
                raise ValueError(f"Unknown weight decode method {weight_decay_method}")

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, ignore_scale: bool = False,
                      is_qkv: bool = False, **kwargs) -> torch.Tensor:
        """Orthogonalize the momentum.

        Args:
            p: The parameter tensor. i is necessary to pass param tensor in addition to momentum
                because a lot of information is only available in the param tensor,
                attributes for example.
            grad: The momentum tensor.

        Returns:
            The orthogonalized gradient tensor.
        """
        # TODO(deyuf): switch to group
        if self.pg_collection:
            tp_group = (
                self.pg_collection.expt_tp
                if getattr(p, 'expert_tp', False)
                else self.pg_collection.tp
            )
        else:
            tp_group = None
        partition_dim = None if self.mode == "blockwise" else getattr(p, "partition_dim", None)
        if partition_dim == -1:
            # emerging-optimizers use None instead of -1 to indicate no tensor parallel
            partition_dim = None

        if self.split_qkv and is_qkv:  # type: ignore[misc]
            qs, ks, vs = split_qkv(grad, self.qkv_split_shapes)
            if self.split_qkv_heads_update:
                qs = merge_heads([self._orthogonalize(q, tp_group, partition_dim, ignore_scale=ignore_scale)
                                for q in split_heads(qs, self.qkv_dim)])
                ks = merge_heads([self._orthogonalize(k, tp_group, partition_dim, ignore_scale=ignore_scale)
                                for k in split_heads(ks, self.qkv_dim)])
                vs = merge_heads([self._orthogonalize(v, tp_group, partition_dim, ignore_scale=ignore_scale)
                                for v in split_heads(vs, self.qkv_dim)])
            else:
                qs = self._orthogonalize(qs, tp_group, partition_dim, ignore_scale=ignore_scale)
                ks = self._orthogonalize(ks, tp_group, partition_dim, ignore_scale=ignore_scale)
                vs = self._orthogonalize(vs, tp_group, partition_dim, ignore_scale=ignore_scale)
            grad = merge_qkv([qs, ks, vs], grad.shape, self.qkv_split_shapes)
        else:
            grad = self._orthogonalize(grad, tp_group, partition_dim, ignore_scale=ignore_scale)
        return grad


    def _orthogonalize(
        self,
        grad: torch.Tensor,
        tp_group: torch.distributed.ProcessGroup,
        partition_dim: int | None = None,
        ignore_scale: bool = False,
    ) -> torch.Tensor:
        assert grad.ndim == 2
        log_single_rank(
            logger,
            logging.DEBUG,
            f'Orthogonalizing grad with {self.num_ns_steps} steps, {self.coefficient_type} coefficient, '
            f'{self.scale_mode} scale mode, extra_scale_factor={self.extra_scale_factor}',
        )
        size = [grad.size(-2), grad.size(-1)]
        if partition_dim is not None:
            size[partition_dim] *= get_pg_size(tp_group)
        orth_grad = newton_schulz_tp(
            grad,
            steps=self.num_ns_steps,
            coefficient_type=self.coefficient_type,
            tp_group=tp_group,
            partition_dim=partition_dim,
            mode="duplicated" if self.mode == "blockwise" else self.mode,
        )
        if self.scale_mode == "shape_up":
            scale_factor = max(size[0] / size[1], size[1] / size[0]) ** 0.5
        else:
            scale_factor = get_muon_scale_factor(size[0], size[1], mode=self.scale_mode)
        if ignore_scale:
            return orth_grad
        return orth_grad * scale_factor * self.extra_scale_factor


    def _resolve_mode(self, is_embedding: bool):
        """Return the effective hypersphere mode for this parameter."""
        if is_embedding and self.hypersphere_embedding_mode is not None:
            return self.hypersphere_embedding_mode
        return self.hypersphere_mode

    def _normalize(self, p: torch.Tensor, x: torch.Tensor, is_qkv: bool = False, is_out_proj: bool = False,
                   apply_muon_scaling: bool = False, is_embedding: bool = False):
        mode = self._resolve_mode(is_embedding)
        if mode is None:
            return
        if is_qkv and self.split_qkv:
            qs, ks, vs = split_qkv(x, self.qkv_split_shapes)
            if self.split_qkv_heads and mode in {"col", "rowcol", "invrowcol", "flat", "equi"}:
                # When splitting heads using torch.split, we only get views of the
                # original tensor, meaning the qs tensor gets modified in-place,
                # no need to copy the updated q to qs after.
                for q in split_heads(qs, self.qkv_dim):
                    self._normalize(p, q, apply_muon_scaling=apply_muon_scaling)
                for k in split_heads(ks, self.qkv_dim):
                    self._normalize(p, k, apply_muon_scaling=apply_muon_scaling)
                for v in split_heads(vs, self.qkv_dim):
                    self._normalize(p, v, apply_muon_scaling=apply_muon_scaling)
            else:
                # If hypersphere_mode is row, we don't need to split heads manually as before
                # because each head are just contiguous *rows* in qs, splitting is unnecessary.
                self._normalize(p, qs, apply_muon_scaling=apply_muon_scaling)
                self._normalize(p, ks, apply_muon_scaling=apply_muon_scaling)
                self._normalize(p, vs, apply_muon_scaling=apply_muon_scaling)
            x.copy_(merge_qkv((qs, ks, vs), x.size(), self.qkv_split_shapes))
            return


        if self.hypersphere_radius == "learnable":
            raise NotImplementedError(f"Learnable hypersphere NYI")

        if mode == "col":
            dim = 0
        elif mode == "row":
            dim = 1
        elif mode in {"flat", "rowcol", "invrowcol", "equi"}:
            dim = None
        elif mode == "embed":
            if is_out_proj:
                dim = 0
            else:
                dim = 1
        else:
            raise ValueError(f"Unknown normalization {mode}")

        eps = self.hypersphere_radius if self.hypersphere_soft else self.hypersphere_eps

        if mode in {"rowcol", "invrowcol"}:
            assert self.hypersphere_kind == "l2"
            assert self.hypersphere_radius == 1.0
            sinkhorn(x, eps=eps, first_norm_col="inv" not in mode)
        elif mode == "equi":
            assert self.hypersphere_kind == "l2"
            assert self.hypersphere_radius == 1.0
            equilibration(x, eps=eps)
        elif self.hypersphere_kind == "l2":
            norm = torch.norm(x, dim=dim, keepdim=True).clamp_min(eps)
            x.mul_(self.hypersphere_radius / norm)
            if mode == "flat":
                shape_max = max(x.size(-2), x.size(-1))
                x.mul_(shape_max ** 0.5)
        elif self.hypersphere_kind == "spectral":
            assert mode == "flat" or is_embedding, f"Spectral norm only supported for flat mode, got {mode}, is_embedding={is_embedding}"
            norm = spectral_norm(x).clamp_min(eps)
            x.mul_(self.hypersphere_radius / norm)
        elif self.hypersphere_kind == "orthogonal":
            assert mode == "flat" or is_embedding, f"Orthogonal norm only supported for flat mode, got {mode}, is_embedding={is_embedding}"
            original_dtype = x.dtype
            x_f32 = x.float() if original_dtype != torch.float32 else x
            x_normalized = self.hypersphere_radius * self.orthogonalize(p, x_f32, ignore_scale=True, is_qkv=is_qkv)
            x.copy_(x_normalized.to(original_dtype))
        elif self.hypersphere_kind == "standard":
            mu = x.mean(dim=dim, keepdim=True)
            std = x.std(dim=dim, keepdim=True).clamp_min(eps)
            x.add_(mu).mul_(self.hypersphere_radius / std)
        else:
            raise ValueError(f"Unknown hypersphere_kind {self.hypersphere_kind}")

        if apply_muon_scaling:
            if self.scale_mode == "shape_up":
                muon_sf = max(x.size(-2) / x.size(-1), x.size(-1) / x.size(-2)) ** 0.5 * self.extra_scale_factor
            else:
                muon_sf = get_muon_scale_factor(x.size(-2), x.size(-1), mode=self.scale_mode) * self.extra_scale_factor
            x.mul_(muon_sf)

    def _project(self, p, g, is_qkv: bool = False, is_out_proj: bool = False, is_embedding: bool = False):
        mode = self._resolve_mode(is_embedding)
        if mode is None or not self.hypersphere_project:
            return
        if is_qkv and self.split_qkv and mode not in {"row", "embed"}:
            p_qs, p_ks, p_vs = split_qkv(p, self.qkv_split_shapes)
            g_qs, g_ks, g_vs = split_qkv(g.clone(), self.qkv_split_shapes)
            if self.split_qkv_heads:
                for p_q, g_q in zip(split_heads(p_qs, self.qkv_dim), split_heads(g_qs, self.qkv_dim)):
                    g_q.copy_(self._project(p_q, g_q))
                for p_k, g_k in zip(split_heads(p_ks, self.qkv_dim), split_heads(g_ks, self.qkv_dim)):
                    g_k.copy_(self._project(p_k, g_k))
                for p_v, g_v in zip(split_heads(p_vs, self.qkv_dim), split_heads(g_vs, self.qkv_dim)):
                    g_v.copy_(self._project(p_v, g_v))
            else:
                # If hypersphere_mode is row, we don't need to split heads manually as before
                # because each head are just contiguous *rows* in qs, splitting is unnecessary.
                g_qs = self._project(p_qs, g_qs)
                g_ks = self._project(p_ks, g_ks)
                g_vs = self._project(p_vs, g_vs)
            return merge_qkv((g_qs, g_ks, g_vs), p.size(), self.qkv_split_shapes)

        if mode == "col":
            dim = 0
        elif mode == "row":
            dim = 1
        elif mode == "flat":
            dim = None
        elif mode in {"rowcol", "invrowcol", "equi"}:
            raise ValueError(f"Project rowcol nyi")
        elif mode == "embed":
            if is_out_proj:
                dim = 0
            else:
                dim = 1
        else:
            raise ValueError(f"Unknown normalization {mode}")

        if self.hypersphere_kind != "l2":
            raise ValueError(f"Project {self.hypersphere_kind} nyi")

        dots = torch.sum(p * g, dim=dim, keepdim=True)
        return g - (dots / self.hypersphere_radius**2) * p


class GainsMasterOptimizer(MasterOptimizer):
    """MasterOptimizer with learnable per-parameter gains (row, col, or flat scaling).

    The gains are fused into the forward/backward pass: before each param step the
    gains are undone, the base optimizer update is computed on the bare normalized
    weights, and afterwards the (potentially updated) gains are re-applied.

    Gain optimizer state (1st/2nd moments) is stored as plain tensors in
    self.state[p] alongside the regular optimizer state, so it ships through
    torch.optim.Optimizer.{state_dict,load_state_dict} without any override.
    Requires --ckpt-format torch.
    """

    _GAIN_KEYS = ("flat_gain", "row_gain", "col_gain", "q_col_gain", "k_col_gain", "v_col_gain")

    def __init__(
        self,
        params,
        hypersphere_gains_mode: Optional[Literal["flat", "embed", "row", "col", "rowcol"]] = None,
        hypersphere_gains_mode_output: Optional[Literal["flat", "row", "col", "rowcol", "none"]] = None,
        hypersphere_gains_mode_embedding: Optional[Literal["flat", "row", "col", "rowcol", "none"]] = None,
        split_qkv_gains: bool = False,
        gains_lr: Optional[float] = None,
        gains_betas: tuple[float, float] = (0.9, 0.999),
        gains_eps: float = 1e-8,
        gains_weight_decay: float = 0.0,
        gain_parametrization: Literal["direct", "offset", "softplus"] = "direct",
        **kwargs,
    ):
        assert hypersphere_gains_mode is not None
        self.hypersphere_gains_mode = hypersphere_gains_mode
        self.hypersphere_gains_mode_output = hypersphere_gains_mode_output
        self.hypersphere_gains_mode_embedding = hypersphere_gains_mode_embedding
        self.split_qkv_gains = split_qkv_gains
        self.gains_lr = gains_lr
        self.gains_betas = gains_betas
        self.gains_eps = gains_eps
        self.gains_weight_decay = gains_weight_decay
        self.gain_parametrization = gain_parametrization
        super().__init__(params, **kwargs)
        self._setup_gains()

    def _phi(self, g: torch.Tensor) -> torch.Tensor:
        """Forward map from raw gain g to effective multiplier phi(g)."""
        mode = self.gain_parametrization
        if mode == "direct":
            return g
        if mode == "offset":
            return 1.0 + g
        if mode == "softplus":
            return torch.nn.functional.softplus(g)
        raise ValueError(f"Unknown gain_parametrization {mode}")

    def _phi_prime(self, g: torch.Tensor) -> torch.Tensor | float:
        """Derivative phi'(g). Returns a scalar 1.0 for the linear modes to skip a multiply."""
        mode = self.gain_parametrization
        if mode in ("direct", "offset"):
            return 1.0
        if mode == "softplus":
            return torch.sigmoid(g)
        raise ValueError(f"Unknown gain_parametrization {mode}")

    def _phi_inv(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse map: given a target effective multiplier x>0, return g s.t. phi(g) = x."""
        mode = self.gain_parametrization
        if mode == "direct":
            return x
        if mode == "offset":
            return x - 1.0
        if mode == "softplus":
            # Stable softplus_inv for x > 0: g = x + log1p(-exp(-x)).
            # As x -> inf, g -> x. As x -> 0+, g -> -inf.
            return x + torch.log1p(-torch.exp(-x))
        raise ValueError(f"Unknown gain_parametrization {mode}")

    @torch.no_grad()  # type: ignore[misc]
    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        if closure is None:
            loss = None
        else:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step"] += 1
            for p in group["params"]:
                if p.grad is not None:
                    gain_grads = self._preprocess_gains(p)
                    self._param_step(p, group)
                    self._gains_step(p, group, gain_grads)
                    self._apply_gains(p)

        return loss

    @torch.no_grad()
    def _preprocess_gains(self, p: torch.nn.Parameter) -> dict:
        """Undo gains on p, compute ∂L/∂gain, rescale p.grad. Returns gain gradient dict."""
        state = self.state[p]
        eps = 1e-8

        # Dispatch to split-QKV variant if applicable.
        if self.split_qkv_gains and self.is_qkv_fn(p) and "q_col_gain" in state:
            return self._preprocess_split_qkv_gains(p)

        flat = state.get("flat_gain")
        row = state.get("row_gain")
        col = state.get("col_gain")

        if flat is None and row is None and col is None:
            return {}

        # Effective multipliers phi(g). Stored values are raw g; multiplication
        # onto p uses phi(g). Computed once and reused for undo/rescale.
        flat_phi = self._phi(flat) if flat is not None else None
        row_phi = self._phi(row) if row is not None else None
        col_phi = self._phi(col) if col is not None else None

        # Undo gains to recover bare normalized weight.
        if flat_phi is not None:
            p.div_(flat_phi.clamp_min(eps))
        if row_phi is not None:
            p.div_(row_phi[:, None].clamp_min(eps))
        if col_phi is not None:
            p.div_(col_phi[None, :].clamp_min(eps))

        # ∂L/∂phi(g) (must be computed before p.grad is rescaled).
        p_times_pgrad = p * p.grad
        gain_grads = {}
        if flat_phi is not None:
            assert row_phi is None and col_phi is None
            gain_grads["flat_gain"] = torch.sum(p_times_pgrad)
        elif row_phi is not None and col_phi is not None:
            gain_grads["row_gain"] = torch.sum(p_times_pgrad * col_phi[None, :], dim=1)
            gain_grads["col_gain"] = torch.sum(p_times_pgrad * row_phi[:, None], dim=0)
        elif row_phi is not None:
            gain_grads["row_gain"] = torch.sum(p_times_pgrad, dim=1)
        else:
            gain_grads["col_gain"] = torch.sum(p_times_pgrad, dim=0)

        # Chain rule: ∂L/∂g = phi'(g) · ∂L/∂phi(g). For direct/offset phi' = 1.
        if flat is not None:
            gain_grads["flat_gain"] = gain_grads["flat_gain"] * self._phi_prime(flat)
        if row is not None:
            gain_grads["row_gain"] = gain_grads["row_gain"] * self._phi_prime(row)
        if col is not None:
            gain_grads["col_gain"] = gain_grads["col_gain"] * self._phi_prime(col)

        # Rescale p.grad so MasterOptimizer sees ∂L/∂(bare p).
        if flat_phi is not None:
            p.grad.mul_(flat_phi)
        if row_phi is not None:
            p.grad.mul_(row_phi[:, None])
        if col_phi is not None:
            p.grad.mul_(col_phi[None, :])

        return gain_grads

    @torch.no_grad()
    def _preprocess_split_qkv_gains(self, p: torch.nn.Parameter) -> dict:
        """_preprocess_gains variant for QKV parameters with separate Q/K/V column gains."""
        state = self.state[p]
        eps = 1e-8
        row = state.get("row_gain")
        q_col = state["q_col_gain"]
        k_col = state["k_col_gain"]
        v_col = state["v_col_gain"]
        shapes = self.qkv_split_shapes

        # Effective multipliers phi(g) for each stored raw g.
        row_phi = self._phi(row) if row is not None else None
        q_col_phi = self._phi(q_col)
        k_col_phi = self._phi(k_col)
        v_col_phi = self._phi(v_col)

        # 1. Undo gains on p to recover the bare normalized weight.
        if row_phi is not None:
            p.div_(row_phi[:, None].clamp_min(eps))
        qs, ks, vs = split_qkv(p, shapes)
        qs.div_(q_col_phi[None, :].clamp_min(eps))
        ks.div_(k_col_phi[None, :].clamp_min(eps))
        vs.div_(v_col_phi[None, :].clamp_min(eps))
        p.copy_(merge_qkv((qs, ks, vs), p.size(), shapes))

        # 2. Gain gradients w.r.t. phi(g); chain rule by phi'(g) applied below.
        p_times_pgrad = p * p.grad
        ptp_q, ptp_k, ptp_v = split_qkv(p_times_pgrad, shapes)
        gain_grads = {}
        if row_phi is not None:
            rq_phi, rk_phi, rv_phi = split_qkv_1d(row_phi, shapes)
            gain_grads["q_col_gain"] = torch.sum(ptp_q * rq_phi[:, None], dim=0)
            gain_grads["k_col_gain"] = torch.sum(ptp_k * rk_phi[:, None], dim=0)
            gain_grads["v_col_gain"] = torch.sum(ptp_v * rv_phi[:, None], dim=0)
            rg_q = torch.sum(ptp_q * q_col_phi[None, :], dim=1)
            rg_k = torch.sum(ptp_k * k_col_phi[None, :], dim=1)
            rg_v = torch.sum(ptp_v * v_col_phi[None, :], dim=1)
            gain_grads["row_gain"] = merge_qkv_1d((rg_q, rg_k, rg_v), row.shape[0], shapes)
        else:
            gain_grads["q_col_gain"] = torch.sum(ptp_q, dim=0)
            gain_grads["k_col_gain"] = torch.sum(ptp_k, dim=0)
            gain_grads["v_col_gain"] = torch.sum(ptp_v, dim=0)

        # Chain rule: ∂L/∂g = phi'(g) · ∂L/∂phi(g).
        if row is not None:
            gain_grads["row_gain"] = gain_grads["row_gain"] * self._phi_prime(row)
        gain_grads["q_col_gain"] = gain_grads["q_col_gain"] * self._phi_prime(q_col)
        gain_grads["k_col_gain"] = gain_grads["k_col_gain"] * self._phi_prime(k_col)
        gain_grads["v_col_gain"] = gain_grads["v_col_gain"] * self._phi_prime(v_col)

        # 3. Rescale p.grad by gains.
        if row_phi is not None:
            p.grad.mul_(row_phi[:, None])
        gq, gk, gv = split_qkv(p.grad, shapes)
        gq.mul_(q_col_phi[None, :])
        gk.mul_(k_col_phi[None, :])
        gv.mul_(v_col_phi[None, :])
        p.grad.copy_(merge_qkv((gq, gk, gv), p.grad.size(), shapes))

        return gain_grads

    @torch.no_grad()
    def _gains_step(self, p, group, gain_grads: dict):
        """Inline Adam update for all gain tensors belonging to p."""
        if not gain_grads:
            return
        state = self.state[p]
        step = group["step"]
        beta1, beta2 = self.gains_betas
        eps = self.gains_eps
        wd = self.gains_weight_decay

        if self.gains_lr is None:
            lr = group["lr"]
        else:
            max_lr = group.get("max_lr", group["lr"])
            schedule_mult = (group["lr"] / max_lr) if max_lr > 0 else 1.0
            lr = self.gains_lr * schedule_mult

        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step

        for name, grad in gain_grads.items():
            gain = state[name]
            m = state[f"{name}_m"]
            v = state[f"{name}_v"]
            if wd != 0.0:
                gain.mul_(1.0 - lr * wd)
            m.mul_(beta1).add_(grad, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            gain.addcdiv_(m, denom, value=-lr / bias_correction1)

    @torch.no_grad()
    def _apply_gains(self, p):
        state = self.state[p]
        q_col = state.get("q_col_gain")
        if q_col is not None:
            row = state.get("row_gain")
            k_col = state["k_col_gain"]
            v_col = state["v_col_gain"]
            if row is not None:
                p.mul_(self._phi(row)[:, None])
            qs, ks, vs = split_qkv(p, self.qkv_split_shapes)
            qs.mul_(self._phi(q_col)[None, :])
            ks.mul_(self._phi(k_col)[None, :])
            vs.mul_(self._phi(v_col)[None, :])
            p.copy_(merge_qkv((qs, ks, vs), p.size(), self.qkv_split_shapes))
            return
        flat = state.get("flat_gain")
        row = state.get("row_gain")
        col = state.get("col_gain")
        if flat is not None:
            p.mul_(self._phi(flat))
        if row is not None:
            p.mul_(self._phi(row)[:, None])
        if col is not None:
            p.mul_(self._phi(col)[None, :])

    def _resolve_gains_mode(self, p):
        """Return the effective gains mode for a parameter, respecting per-layer overrides."""
        is_output = getattr(p, "is_output_parameter", False)
        is_embedding = getattr(p, "is_embedding_parameter", False)
        if is_output and self.hypersphere_gains_mode_output is not None:
            return self.hypersphere_gains_mode_output
        if is_embedding and self.hypersphere_gains_mode_embedding is not None:
            return self.hypersphere_gains_mode_embedding
        return self.hypersphere_gains_mode

    @torch.no_grad()
    def _setup_gains(self):
        """Initialize gain tensors and their Adam m/v buffers in self.state[p].

        Only creates missing entries (idempotent).

        When self.preserve_init is True, new gain tensors are initialised to the
        current parameter norms instead of ones. p is NOT modified — the forward
        pass continues to see the original init weights. At the first optimizer
        step _preprocess_gains divides p by the gains, yielding a bare weight
        that is already approximately on the hypersphere.

        For rowcol: row norms are absorbed first; col norms are then computed from
        the (virtual) row-normalised p, matching the sequential undo in _preprocess_gains.
        """
        eps = self.hypersphere_eps
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim < 2:
                    continue
                gains_mode = self._resolve_gains_mode(p)
                if gains_mode is None or gains_mode == "none":
                    continue
                is_out_proj = getattr(p, "is_out_proj", False)
                is_qkv = self.is_qkv_fn(p)
                state = self.state[p]

                wants_row = ("row" in gains_mode) or (gains_mode == "embed" and not is_out_proj)
                wants_col = ("col" in gains_mode) or (gains_mode == "embed" and is_out_proj)
                wants_flat = (gains_mode == "flat")

                # All gain state stores raw `g`; the effective multiplier is phi(g).
                # Initialisation maps the desired phi(g_init) (== 1.0, or the param norm
                # for preserve_init) through phi^-1 to get the raw value to store.
                if wants_row:
                    if "row_gain" not in state:
                        if self.preserve_init:
                            target = p.detach().float().norm(dim=1).clamp_min(eps)
                        else:
                            target = torch.ones(p.size(0), dtype=torch.float32, device=p.device)
                        state["row_gain"] = self._phi_inv(target)
                    if "row_gain_m" not in state:
                        state["row_gain_m"] = torch.zeros_like(state["row_gain"])
                    if "row_gain_v" not in state:
                        state["row_gain_v"] = torch.zeros_like(state["row_gain"])

                if wants_col:
                    if is_qkv and self.split_qkv_gains:
                        if self.preserve_init and not wants_row:
                            p_f32 = p.detach().float()
                            qs, ks, vs = split_qkv(p_f32, self.qkv_split_shapes)
                            slice_norms = {
                                "q": qs.norm(dim=0).clamp_min(eps),
                                "k": ks.norm(dim=0).clamp_min(eps),
                                "v": vs.norm(dim=0).clamp_min(eps),
                            }
                        for prefix in ("q", "k", "v"):
                            key = f"{prefix}_col_gain"
                            if key not in state:
                                if self.preserve_init and not wants_row:
                                    target = slice_norms[prefix]
                                else:
                                    target = torch.ones(p.size(1), dtype=torch.float32, device=p.device)
                                state[key] = self._phi_inv(target)
                            if f"{key}_m" not in state:
                                state[f"{key}_m"] = torch.zeros_like(state[key])
                            if f"{key}_v" not in state:
                                state[f"{key}_v"] = torch.zeros_like(state[key])
                    else:
                        if "col_gain" not in state:
                            if self.preserve_init and not wants_row:
                                target = p.detach().float().norm(dim=0).clamp_min(eps)
                            else:
                                target = torch.ones(p.size(1), dtype=torch.float32, device=p.device)
                            state["col_gain"] = self._phi_inv(target)
                        if "col_gain_m" not in state:
                            state["col_gain_m"] = torch.zeros_like(state["col_gain"])
                        if "col_gain_v" not in state:
                            state["col_gain_v"] = torch.zeros_like(state["col_gain"])

                if wants_flat:
                    if "flat_gain" not in state:
                        if self.preserve_init:
                            shape_max = max(p.size(-2), p.size(-1))
                            target = (p.detach().float().norm().clamp_min(eps) / shape_max ** 0.5).reshape(())
                        else:
                            target = torch.ones((), dtype=torch.float32, device=p.device)
                        state["flat_gain"] = self._phi_inv(target)
                    if "flat_gain_m" not in state:
                        state["flat_gain_m"] = torch.zeros_like(state["flat_gain"])
                    if "flat_gain_v" not in state:
                        state["flat_gain_v"] = torch.zeros_like(state["flat_gain"])

def split_qkv(x, shapes: tuple[int, int, int]) -> list[torch.Tensor]:
    # split grouped attention parameters (e.g., QKV, GQA, etc.)
    shape = x.shape
    num_query_groups = shape[0] // sum(shapes)
    qkv = torch.split(
        x.view(num_query_groups, sum(shapes), -1),
        shapes,
        dim=1,
    )
    qkv = [g.reshape(-1, shape[-1]) for g in qkv]
    return qkv


def split_heads(x, head_dim: int) -> tuple[torch.Tensor]:
    return torch.split(x, head_dim, dim=0)


def merge_qkv(qkv, xshape: tuple[int, int], shapes: tuple[int, int, int]) -> torch.Tensor:
    num_query_groups = xshape[0] // sum(shapes)
    qkv = [g.view(num_query_groups, -1, xshape[-1]) for g in qkv]
    return torch.cat(qkv, dim=1).view(xshape)


def split_qkv_1d(x: torch.Tensor, shapes: tuple[int, int, int]) -> list[torch.Tensor]:
    """Split a 1D tensor (e.g. row_gain) along the same QKV boundaries as split_qkv."""
    return [g.squeeze(-1) for g in split_qkv(x.unsqueeze(-1), shapes)]


def merge_qkv_1d(parts: tuple[torch.Tensor, ...], full_len: int, shapes: tuple[int, int, int]) -> torch.Tensor:
    """Inverse of split_qkv_1d."""
    parts_2d = [g.unsqueeze(-1) for g in parts]
    return merge_qkv(parts_2d, (full_len, 1), shapes).squeeze(-1)


def merge_heads(xs) -> torch.Tensor:
    return torch.cat(xs, dim=0)


@torch.no_grad()
def spectral_norm(x, n_iters: int = 10):
    v = torch.randn(x.size(-1), device=x.device, dtype=x.dtype)
    for _ in range(n_iters):
        u = x @ v
        u = u / torch.linalg.vector_norm(u)
        v = x.T @ u
        v = v / torch.linalg.vector_norm(v)
    return (u @ x @ v).abs()


def sinkhorn(x, n_iters: int = 10, eps: float = 1e-8, first_norm_col: bool = True):
    for _ in range(n_iters):
        norm_col = torch.norm(x, dim=0, keepdim=True).clamp_min(eps)
        norm_row = torch.norm(x, dim=1, keepdim=True).clamp_min(eps)
        if first_norm_col:
            x.div_(norm_col).div_(norm_row)
        else:
            x.div_(norm_row).div_(norm_col)

def equilibration(x, n_iters: int = 1, eps: float = 1e-8):
    for _ in range(n_iters):
        norm_col = torch.norm(x, dim=0, keepdim=True).clamp_min(eps)
        x.div_(norm_col)
        norm_row = torch.norm(x, dim=1, keepdim=True).clamp_min(eps)
        x.div_(norm_row)



def get_megatron_master_optimizer(
    config: OptimizerConfig,
    model_chunks: list[MegatronModule],
    config_overrides: Optional[dict[ParamKey, ParamGroupOverride]] = None,
    use_gloo_process_groups: bool = True,
    layer_wise_distributed_optimizer: bool = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """This function is used to get the muon optimizer for the model chunks.
    It is used to get the muon optimizer for the model chunks.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (List[MegatronModule]): model chunks to get optimizer for.
        use_gloo_process_groups (bool): if false, disable use of Gloo process groups
            in underlying Megatron optimizers.
        layer_wise_distributed_optimizer (bool): if true, use layer-wise distributed optimizer.
            Defaults to False.
    """
    # Muon currently use adam config. setting str here to call regular get for adam creation
    original_optimizer_name = config.optimizer
    config.optimizer = 'adam'

    # Dist-opt is not supported due to strong coupling with how DDP init grad buffer
    # In theory we can change DDP to enable use muon and dist-opt-adam together
    if config.use_distributed_optimizer:
        raise Exception('master with dist optimizer is not supported.')
    # only support bf16 w/o loss scale now
    if config.fp16:
        raise Exception('master with fp16 is not supported.')

    # before this function receive properly created collection
    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    log_single_rank(logger, logging.INFO, f'Setting up emerging master with config {config}')

    # Needed for torch_dist ckpt_format, unlike torch ckpt_format
    # For other emerging optimizers, need to implement init_state_fn as well
    # TODO(boxiangw): Improve usability after optimizer refactor
    # TODO(boxiangw): support precision aware optimizer
    # TODO: do we need this anymore?
    def master_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if "exp_avg" not in opt.state[p]:
                    opt.state[p]["exp_avg"] = torch.zeros_like(p.data)
                    if not group["use_orthogonal_updates"]:  # Enables g^2 EMA as in adam & ademamix.
                        if group["beta2"] != 0:
                            opt.state[p]["exp_avg_sq"] = torch.zeros_like(p.data)
                        if group["alpha"] != 0:  # Enables slow momentum as in ademamix.
                            opt.state[p]["exp_avg_slow"] = torch.zeros_like(p.data)

    def adam_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if "exp_avg" not in opt.state[p]:
                    if config is None or not config.use_precision_aware_optimizer:
                        opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                        opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                    else:
                        opt.initialize_state(p)

    optimizers = []
    # record list of non/linear params
    linear_params = []
    nonlinear_params = []
    for model_chunk in model_chunks:
        # use config to determine qkv split shapes.
        # no need to check tp since tp splits by head and this is per head(group) dimension
        num_attention_heads = model_chunk.config.num_attention_heads
        num_query_groups = model_chunk.config.num_query_groups
        kv_channels = model_chunk.config.kv_channels
        qkv_split_shapes = [
            num_attention_heads // num_query_groups * kv_channels,
            kv_channels,
            kv_channels,
        ]
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            # add flag for expert weight so optimizer can figure which tp group it uses
            # alternatively, create new param group and save tp_group. this require more
            # change in optimizer
            if 'experts' in name and 'shared' not in name:
                param.expert_tp = True
            # add flag for qkv parameter
            # TODO(deyuf): support MLA
            if 'linear_qkv.weight' in name and len(param.shape) == 2:
                param.is_qkv = True
            if ("linear_fc2" in name or "linear_proj" in name) and len(param.shape) == 2:
                param.is_out_proj = True
            include_embeddings = config.hypersphere_embeddings or config.hypersphere_embedding_mode is not None
            if (
                (include_embeddings or not getattr(param, 'is_embedding_or_output_parameter', False))
                and len(param.shape) == 2
            ):
                linear_params.append(param)
            else:
                nonlinear_params.append(param)

    matrix_lr = config.matrix_lr if config.matrix_lr is not None else config.muon_lr_factor * config.lr
    lr_ratio = config.min_lr / config.lr if config.scale_min_lr and config.lr > 0 else None

    master_kwargs = {
        # Common.
        "lr": matrix_lr,
        "weight_decay": config.weight_decay,
        "weight_decay_method": config.weight_decay_method,

        # Adam & Ademamix settings.
        "betas": (config.adam_beta1, config.adam_beta2, config.ademamix_beta3),
        "alpha": config.ademamix_alpha,
        "beta3_warmup": config.ademamix_beta3_warmup,
        "alpha_warmup": config.ademamix_alpha_warmup,
        "eps": config.adam_eps,

        # Hypersphere optimization.
        "hypersphere_mode": config.hypersphere_mode,
        "hypersphere_embedding_mode": config.hypersphere_embedding_mode,
        "hypersphere_kind": config.hypersphere_kind,
        "hypersphere_radius": config.hypersphere_radius,
        "hypersphere_update": config.hypersphere_update,
        "hypersphere_update_embeddings": config.hypersphere_update_embeddings,
        "hypersphere_project": config.hypersphere_project,
        "hypersphere_soft": config.hypersphere_soft,

        # Muon.
        "use_orthogonal_updates": config.use_orthogonal_updates,
        "poor_mans_ortho": config.poor_mans_ortho,
        "momentum_beta": config.muon_momentum,
        "use_nesterov": config.muon_use_nesterov,
        "split_qkv": config.muon_split_qkv,
        "split_qkv_heads": config.hypersphere_split_heads,
        "split_qkv_heads_update": config.hypersphere_split_heads_update,
        "is_qkv_fn": lambda p: getattr(p, "is_qkv", False),
        "fp32_matmul_prec": config.muon_fp32_matmul_prec,
        "num_ns_steps": config.muon_num_ns_steps,
        "scale_mode": config.muon_scale_mode,
        "extra_scale_factor": config.muon_extra_scale_factor,
        "pg_collection": pg_collection,
        "mode": config.muon_tp_mode,

        "qkv_split_shapes": qkv_split_shapes,
        "qkv_dim": kv_channels,  # head dim for split_heads when split_qkv_heads=True
        "preserve_init": config.hypersphere_preserve_init,
    }

    # freezing nonlinear params and get param groups for muon
    for param in nonlinear_params:
        param.requires_grad = False

    config_overrides_master = {**config_overrides}

    # Resolve effective embedding / output LRs: absolute (--embedding-lr /
    # --output-lr) wins, else fall back to the multiplier form.
    if config.embedding_lr is not None:
        effective_embedding_lr = config.embedding_lr
    elif config.embedding_lr_multiplier is not None:
        effective_embedding_lr = config.embedding_lr_multiplier * config.lr
    else:
        effective_embedding_lr = None

    if config.output_lr is not None:
        effective_output_lr = config.output_lr
    elif config.embedding_lr_multiplier is not None or config.embedding_lr is not None:
        # When an embedding-specific LR is set, the LM head decouples from the
        # embedding and defaults to the base lr (historical behaviour).
        effective_output_lr = config.lr
    else:
        effective_output_lr = None  # no override; stays on the matrix group

    embedding_override = {}
    if effective_embedding_lr is not None:
        embedding_override["max_lr"] = effective_embedding_lr
    no_ortho_emb = config.use_orthogonal_updates and not config.use_orthogonal_embeddings
    if no_ortho_emb:
        embedding_override["use_orthogonal_updates"] = False

    matrix_override = ParamGroupOverride(max_lr=matrix_lr)
    if lr_ratio is not None:
        matrix_override["min_lr"] = matrix_lr * lr_ratio

    if "max_lr" in embedding_override:
        if lr_ratio is not None:
            embedding_override["min_lr"] = embedding_override["max_lr"] * lr_ratio
        # Exclude embedding/output params from the wildcard to avoid conflicting max_lr overrides.
        non_emb = ParamPredicate(
            name="non_embedding_or_output",
            fn=lambda p: not getattr(p, "is_embedding_or_output_parameter", False),
        )
        config_overrides_master[ParamKey(predicate=non_emb)] = matrix_override
    else:
        config_overrides_master[ParamKey(name="*")] = matrix_override

    if embedding_override:
        # Embedding weights get the embedding LR override.
        config_overrides_master[ParamKey(attr="is_embedding_parameter")] = ParamGroupOverride(**embedding_override)

    if effective_output_lr is not None or no_ortho_emb:
        # LM head (output layer) gets its own override when untied; when tied it
        # shares the embedding tensor so is_embedding_parameter is also set and
        # the embedding override already applies — we use a predicate to avoid
        # a conflict.
        output_override = {}
        if effective_output_lr is not None:
            output_override["max_lr"] = effective_output_lr
            if lr_ratio is not None:
                output_override["min_lr"] = effective_output_lr * lr_ratio
        if no_ortho_emb:
            output_override["use_orthogonal_updates"] = False
        only_output = ParamPredicate(
            name="output_not_embedding",
            fn=lambda p: (getattr(p, "is_output_parameter", False)
                          and not getattr(p, "is_embedding_parameter", False)),
        )
        config_overrides_master[ParamKey(predicate=only_output)] = ParamGroupOverride(**output_override)

    linear_param_groups = _get_param_groups(model_chunks, config, config_overrides_master)
    # if layerwise distributed optimizer is not used, need to handle ep params separately
    expert_param_groups = []
    if not layer_wise_distributed_optimizer:
        expert_param_groups = [g for g in linear_param_groups if g['is_expert_parallel']]
        linear_param_groups = [g for g in linear_param_groups if not g['is_expert_parallel']]

    gains_kwargs = dict(
        hypersphere_gains_mode=config.hypersphere_gains_mode,
        hypersphere_gains_mode_output=config.hypersphere_gains_mode_output,
        hypersphere_gains_mode_embedding=config.hypersphere_gains_mode_embedding,
        split_qkv_gains=config.split_qkv_gains,
        gains_lr=config.gains_lr,
        gains_betas=(config.adam_beta1, config.adam_beta2),
        gains_eps=config.adam_eps,
        gains_weight_decay=config.weight_decay,
        gain_parametrization=config.gain_parametrization,
    )
    if config.hypersphere_gains_mode:
        optimizer = GainsMasterOptimizer(
            linear_param_groups, **gains_kwargs, **master_kwargs
        )
    else:
        optimizer = MasterOptimizer(linear_param_groups, **master_kwargs)

    reset_config_bf16 = False
    if config.bf16:
        if layer_wise_distributed_optimizer:
            # creating master weight before layerwise sharding will lead to unnecessary master
            # weight so here we delay master weight creation into layer_wise unset config.bf16
            # will also result in all optimizers below(adam) to also not be wrapped
            config.bf16 = False
            reset_config_bf16 = True
        else:
            # if not using layer_wise wrapper, just create master weight here is fine
            optimizer = Float16OptimizerWithFloat16Params(
                optimizer, config, None, master_init_state_fn
            )
    else:
        optimizer = FP32Optimizer(optimizer, config, master_init_state_fn)

    optimizers.append(optimizer)

    # expert optimizer exists meaning layerwise distributed optimizer is not used
    if len(expert_param_groups) > 0:
        if config.hypersphere_gains_mode:
            expert_optimizer = GainsMasterOptimizer(
                expert_param_groups, **gains_kwargs, **master_kwargs
            )
        else:
            expert_optimizer = MasterOptimizer(expert_param_groups, **master_kwargs)
        if config.bf16:
            expert_optimizer = Float16OptimizerWithFloat16Params(
                expert_optimizer, config, None, master_init_state_fn
            )
        else:
            expert_optimizer = FP32Optimizer(expert_optimizer, config, master_init_state_fn)
        setattr(expert_optimizer, 'grad_stats_parallel_group', pg_collection.tp_ep_pp)
        optimizers.append(expert_optimizer)

    # done with muon, unfreeze nonlinear and freeze linear
    for param in nonlinear_params:
        param.requires_grad = True
    for param in linear_params:
        param.requires_grad = False

    config_overrides_adam = {**config_overrides}
    if config.qk_layernorm_frozen:
        print("Freezing qknorm")
        config_overrides_adam[ParamKey(name="*q_layernorm*")] = ParamGroupOverride(max_lr=0)
        config_overrides_adam[ParamKey(name="*k_layernorm*")] = ParamGroupOverride(max_lr=0)

    # Propagate the embedding / output LR overrides into chained_adam. Without
    # this, when --hs-embed is not set the embedding (and untied output) live
    # in chained_adam and silently ignore --embedding-lr / --output-lr / --elr.
    if effective_embedding_lr is not None:
        adam_emb_override = {"max_lr": effective_embedding_lr}
        if lr_ratio is not None:
            adam_emb_override["min_lr"] = effective_embedding_lr * lr_ratio
        config_overrides_adam[ParamKey(attr="is_embedding_parameter")] = ParamGroupOverride(**adam_emb_override)
    if effective_output_lr is not None:
        only_output_adam = ParamPredicate(
            name="output_not_embedding",
            fn=lambda p: (getattr(p, "is_output_parameter", False)
                          and not getattr(p, "is_embedding_parameter", False)),
        )
        adam_out_override = {"max_lr": effective_output_lr}
        if lr_ratio is not None:
            adam_out_override["min_lr"] = effective_output_lr * lr_ratio
        config_overrides_adam[ParamKey(predicate=only_output_adam)] = ParamGroupOverride(**adam_out_override)

    # call original get. linear params will be skipped since they're freezed
    chained_adam = get_megatron_optimizer(
        config,
        model_chunks,
        config_overrides=config_overrides_adam,
        use_gloo_process_groups=use_gloo_process_groups,
    )

    # unfreeze everything
    for param in linear_params:
        param.requires_grad = True

    # chain everything together
    init_fns = [master_init_state_fn] + len(chained_adam.chained_optimizers) * [adam_init_state_fn]
    optimizers += chained_adam.chained_optimizers

    config.optimizer = original_optimizer_name
    if layer_wise_distributed_optimizer:
        log_single_rank(logger, logging.INFO, 'Using LayerWiseDistributedOptimizer for Muon')
        if reset_config_bf16:
            config.bf16 = True
        return LayerWiseDistributedOptimizer(
            optimizers, config, pg_collection, init_state_fn_list=init_fns
        )
    return ChainedOptimizer(optimizers)
