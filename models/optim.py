"""Muon optimizer (MomentUm Orthogonalized by Newton-schulz).

Geometry-aware optimizer for 2D weight matrices: it takes the momentum, orthogonalizes it via a
quintic Newton-Schulz iteration (so the update is (approximately) the nearest semi-orthogonal
matrix), and applies it. This makes each update an "angular" rotation of the weight matrix rather
than an axis-aligned Adam step, which is the sense in which it is tailored to normalized/hyperspherical
weights. 1D params (embeddings, norms, scalars, biases) are NOT suitable for Muon and are optimized
with AdamW instead (see create_model._build_dense_optimizer).

Reference: Keller Jordan et al., "Muon: An optimizer for hidden layers in neural networks."
"""

import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Orthogonalize G (2D) via a quintic Newton-Schulz iteration; returns U V^T-ish (semi-orthogonal)."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + eps)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.t()
        transposed = True
    for _ in range(steps):
        A = X @ X.t()
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.t()
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True,
                 ns_steps: int = 5, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    raise ValueError("Muon only supports 2D params; route others to AdamW.")
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf = st["momentum_buffer"]
                buf.mul_(mom).add_(g)
                u = g.add(buf, alpha=mom) if group["nesterov"] else buf
                u = zeropower_via_newtonschulz5(u, steps=group["ns_steps"])
                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)                       # decoupled weight decay
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5  # shape-aware update scaling
                p.add_(u.to(p.dtype), alpha=-lr * scale)
        return loss


class AdamATan2(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.95), weight_decay: float = 0.0,
                 a: float = 1.2732395447351628, b: float = 1.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, a=a, b=b)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            a, b = group["a"], group["b"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    st["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                st["step"] += 1
                t = st["step"]
                exp_avg, exp_avg_sq = st["exp_avg"], st["exp_avg_sq"]

                g32 = g.float()
                exp_avg.lerp_(g32, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g32, g32, value=1.0 - beta2)

                m_hat = exp_avg / (1.0 - beta1 ** t)
                v_hat = exp_avg_sq / (1.0 - beta2 ** t)

                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)  # decoupled weight decay
                update = torch.atan2(m_hat, b * v_hat.sqrt()).mul_(a)
                p.add_(update.to(p.dtype), alpha=-lr)
        return loss