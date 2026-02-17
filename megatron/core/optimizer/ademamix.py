# TODO split qkv normalization.
import math
import torch
from typing import List, Dict, Optional, Callable, Tuple, Any, Literal

def linear_warmup_scheduler(step: int, alpha_end: float, alpha_start: float = 0, warmup: int = 1) -> float:
    """Linear warmup scheduler for alpha parameter."""
    if step < warmup:
        a = step / float(warmup)
        return (1.0 - a) * alpha_start + a * alpha_end
    return alpha_end

def linear_hl_warmup_scheduler(step: int, beta_end: float, beta_start: float = 0, warmup: int = 1) -> float:
    """Half-life warmup scheduler for beta3 parameter."""
    def f(beta: float, eps: float = 1e-8) -> float:
        return math.log(0.5) / math.log(beta + eps) - 1

    def f_inv(t: float) -> float:
        return math.pow(0.5, 1 / (t + 1))

    if step < warmup:
        a = step / float(warmup)
        return f_inv((1.0 - a) * f(beta_start) + a * f(beta_end))
    return beta_end


class AdEMAMix(torch.optim.Optimizer):
    """
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 2.0,
        beta3_warmup: Optional[int] = None,
        alpha_warmup: Optional[int] = None,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        hyperball_mode: Optional[Literal["row", "col", "rowcol", "flat"]] = None,
        hyperball_kind: Optional[Literal["l2", "standard", "spectral"]] = None,
        hyperball_radius: Literal["learnable"] | float = 1.0,
        hyperball_eps: float = 1e-8,
        hyperball_update: float = True,
    ):

        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta3 parameter: {betas[2]}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")

        defaults = dict(
            lr=lr,
            betas=betas,
            alpha=alpha,
            beta3_warmup=beta3_warmup,
            alpha_warmup=alpha_warmup,
            eps=eps,
            weight_decay=weight_decay,
            step=0,
            hyperball_mode=hyperball_mode,
            hyperball_kind=hyperball_kind,
            hyperball_radius=hyperball_radius,
            hyperball_eps=hyperball_eps,
            hyperball_update=hyperball_update,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step"] += 1
            lr = group["lr"]
            lmbda = group["weight_decay"]
            eps = group["eps"]
            beta1, beta2, beta3_final = group["betas"]
            beta3_warmup = group["beta3_warmup"]
            alpha_final = group["alpha"]
            alpha_warmup = group["alpha_warmup"]
            step = group["step"]

            hyperball_kwargs = {
                "hyperball_mode": group["hyperball_mode"],
                "hyperball_kind": group["hyperball_kind"],
                "hyperball_radius": group["hyperball_radius"],
                "hyperball_eps": group["hyperball_eps"],
                "hyperball_update": group["hyperball_update"],
            }

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdEMAMix does not support sparse gradients.")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # state["step"] = torch.zeros((1,), dtype=torch.float32, device=p.device)
                    if beta1 != 0.0:  # save memory in case beta1 is 0.0
                        state["exp_avg_fast"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if alpha_final != 0.0:
                        state["exp_avg_slow"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"]   = torch.zeros_like(p, memory_format=torch.preserve_format)

                if beta1 != 0.0:
                    exp_avg_fast = state["exp_avg_fast"]
                if alpha_final != 0.0:
                    exp_avg_slow = state["exp_avg_slow"]
                exp_avg_sq = state["exp_avg_sq"]

                bias_correction1 = 1.0 - (beta1 ** step)
                bias_correction2 = 1.0 - (beta2 ** step)

                # Compute the effective alpha and beta3 in case warmup is used
                if alpha_warmup is not None and alpha_final != 0.0:
                    alpha = linear_warmup_scheduler(step, alpha_end=alpha_final, alpha_start=0, warmup=alpha_warmup)
                else:
                    alpha = alpha_final

                if beta3_warmup is not None:
                    beta3 = linear_hl_warmup_scheduler(step, beta_end=beta3_final, beta_start=beta1, warmup=beta3_warmup)
                else:
                    beta3 = beta3_final

                if beta1 != 0.0:
                    exp_avg_fast.mul_(beta1).add_(grad, alpha=1 - beta1)
                else:
                    exp_avg_fast = grad
                if alpha_final != 0.0:
                    exp_avg_slow.mul_(beta3).add_(grad, alpha=1 - beta3)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                if beta1 != 0.0 and alpha_final != 0.0:
                    update = (exp_avg_fast.div(bias_correction1) + alpha * exp_avg_slow) / denom
                elif beta1 != 0.0:  # alpha=0
                    update = exp_avg_fast.div(bias_correction1) / denom
                elif alpha_final != 0.0:  # beta1=0
                    update = (alpha * exp_avg_slow) / denom
                else:  # alpha=0 and beta1=0 lol.
                    update = 1 / denom

                # decay
                if lmbda != 0.0:
                    update.add_(p, alpha=lmbda)
                
                self.pre_weight_update_fn_inplace(p, update, **hyperball_kwargs)
                p.add_(update, alpha=-lr)
                self.post_weight_update_fn_inplace(p, update, **hyperball_kwargs)

        return loss

    def _norm_mode_to_dim(self, mode: str) -> Optional[int]:
        if mode == "col":
            return 0
        if mode == "row":
            return 1
        if mode == "flat":
            return None
        raise ValueError(f"Unknown mode {mode}")

    def pre_weight_update_fn_inplace(self, p: torch.Tensor, update: torch.Tensor,
                                     hyperball_mode, hyperball_kind, hyperball_radius, hyperball_eps, hyperball_update) -> None:
        """Store the original weight norm and normalize the update."""

        if hyperball_mode is None:  # No normalization constraint requested.
            return
        if hyperball_radius == "learnable":
            raise NotImplementedError(f"Learnable hyperball NYI")
        if hyperball_mode == "rowcol":
            raise NotImplementedError(f"Rowcol hyperball NYI")
        if hyperball_kind == "spectral":
            raise NotImplementedError(f"hyperball spectral NYI")  # because param might be sharded.


        # Normalize the update in-place and scale by R
        # This modifies update to be: R * normalize(update)
        R = hyperball_radius
        self.state[p]["hyperball_R"] = R
        dim = self._norm_mode_to_dim(hyperball_mode)
        if not hyperball_update:
            return
        if hyperball_kind == "l2":
            update_norm = update.norm(dim=dim, keepdim=True).clamp_min(hyperball_eps)
            update.mul_(R / update_norm)
        else:  # standardization.
            mu = update.mean(dim=dim, keepdim=True)
            std = update.std(dim=dim, keepdim=True).clamp_min(hyperball_eps)
            update.add_(mu).mul_(R / std)
    
    def post_weight_update_fn_inplace(self, p: torch.Tensor, update: torch.Tensor,
                                      hyperball_mode, hyperball_kind, hyperball_radius, hyperball_eps, hyperball_update) -> None:
        """Normalize the updated weights and scale back to original norm."""

        if hyperball_mode is None:  # No normalization constraint requested.
            return
        if hyperball_radius == "learnable":
            raise NotImplementedError(f"Learnable hyperball error NYI")
        if hyperball_mode == "rowcol":
            raise NotImplementedError(f"Rowcol hyperball NYI")
        if hyperball_kind == "spectral":
            raise NotImplementedError(f"hyperball spectral NYI")  # because param might be sharded.

        # Normalize the result and scale back by R: p = R * (p / ||p||).
        R = self.state[p]["hyperball_R"]
        dim = self._norm_mode_to_dim(hyperball_mode)
        if hyperball_kind == "l2":
            p_norm = p.norm(dim=dim, keepdim=True).clamp_min(hyperball_eps)
            p.mul_(R / p_norm)
        else:
            mu = p.mean(dim=dim, keepdim=True)
            std = p.std(dim=dim, keepdim=True).clamp_min(hyperball_eps)
            p.add_(mu).mul_(R / std)



