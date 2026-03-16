import torch


def psi_collision(x: torch.Tensor) -> torch.Tensor:
    """Collision-sensitive distance transform from the paper: Ψ(x) = exp(x) - x."""
    return torch.exp(x) - x


def collision_avoidance_energy_from_signed_distance(
    signed_distance: torch.Tensor,
    omega_c: float = 10.0,
    r: float = 3.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute Eq.(9)-style collision energy from signed distance."""
    if r <= 0:
        raise ValueError(f"r must be positive, got {r}")
    if omega_c <= 0:
        raise ValueError(f"omega_c must be positive, got {omega_c}")

    d = signed_distance
    d_type = d.dtype

    mask_pos = (d > 0).to(d_type)
    mask_neg = (d < 0).to(d_type)

    scaled = torch.clamp(1.0 - d / r, min=0.0)
    transformed = psi_collision(omega_c * scaled)

    pos_term = (mask_pos * transformed).sum() / (mask_pos.sum() + eps)
    neg_term = (mask_neg * transformed).sum() / (mask_neg.sum() + eps)

    return (pos_term + neg_term) / omega_c


def gradient_guidance_step(
    x: torch.Tensor,
    signed_distance_fn,
    step_size: float = 0.1,
    omega_c: float = 10.0,
    r: float = 3.0,
    eps: float = 1e-6,
    grad_clip: float | None = 5.0,
):
    """One-step gradient guidance update for collision avoidance."""
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")

    x_var = x.detach().clone().requires_grad_(True)
    signed_distance = signed_distance_fn(x_var)

    energy = collision_avoidance_energy_from_signed_distance(
        signed_distance=signed_distance,
        omega_c=omega_c,
        r=r,
        eps=eps,
    )

    grad = torch.autograd.grad(energy, x_var, create_graph=False, retain_graph=False)[0]

    if grad_clip is not None:
        grad_norm = grad.norm()
        if grad_norm > grad_clip:
            grad = grad * (grad_clip / (grad_norm + 1e-12))

    x_next = (x_var - step_size * grad).detach()
    return x_next, energy.detach(), grad.detach()


def sample_cost_map(
    cost_map: torch.Tensor,
    query_xy: torch.Tensor,
    origin_xy: torch.Tensor,
    resolution: float,
    align_corners: bool = True,
) -> torch.Tensor:
    """Sample differentiable cost map M(x) at world-coordinate query points."""
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")

    if cost_map.ndim == 3:
        cost_map = cost_map.unsqueeze(1)
    elif cost_map.ndim != 4:
        raise ValueError(f"cost_map must be [B,H,W] or [B,1,H,W], got {tuple(cost_map.shape)}")

    if query_xy.ndim != 3 or query_xy.shape[-1] != 2:
        raise ValueError(f"query_xy must be [B,T,2], got {tuple(query_xy.shape)}")

    batch, _, height, width = cost_map.shape
    if query_xy.shape[0] != batch or origin_xy.shape != (batch, 2):
        raise ValueError(
            f"shape mismatch: cost_map batch={batch}, query_xy={tuple(query_xy.shape)}, origin_xy={tuple(origin_xy.shape)}"
        )

    pix_xy = (query_xy - origin_xy[:, None, :]) / resolution

    if align_corners:
        gx = 2.0 * pix_xy[..., 0] / max(width - 1, 1) - 1.0
        gy = 2.0 * pix_xy[..., 1] / max(height - 1, 1) - 1.0
    else:
        gx = (2.0 * (pix_xy[..., 0] + 0.5) / width) - 1.0
        gy = (2.0 * (pix_xy[..., 1] + 0.5) / height) - 1.0

    grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)  # [B,T,1,2]

    sampled = torch.nn.functional.grid_sample(
        cost_map,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=align_corners,
    )
    return sampled[:, 0, :, 0]


def drivable_area_energy(
    map_values: torch.Tensor,
    omega_d: float = 10.0,
    eps: float = 1e-6,
    only_positive: bool = True,
) -> torch.Tensor:
    """Eq.(12)-style staying-in-drivable-area energy."""
    if omega_d <= 0:
        raise ValueError(f"omega_d must be positive, got {omega_d}")

    m = map_values
    pos_mask = (m > 0).to(m.dtype)

    transformed = psi_collision(omega_d * m)
    if only_positive:
        transformed = transformed * pos_mask

    denom = pos_mask.sum() + eps
    return transformed.sum() / denom / omega_d


def drivable_area_guidance_step(
    ego_xy: torch.Tensor,
    cost_map: torch.Tensor,
    origin_xy: torch.Tensor,
    resolution: float,
    step_size: float = 0.1,
    omega_d: float = 10.0,
    eps: float = 1e-6,
    grad_clip: float | None = 5.0,
    only_positive: bool = True,
    align_corners: bool = True,
):
    """One gradient-guidance step for drivable-area constraint."""
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")

    x_var = ego_xy.detach().clone().requires_grad_(True)
    sampled_m = sample_cost_map(
        cost_map=cost_map,
        query_xy=x_var,
        origin_xy=origin_xy,
        resolution=resolution,
        align_corners=align_corners,
    )
    energy = drivable_area_energy(
        map_values=sampled_m,
        omega_d=omega_d,
        eps=eps,
        only_positive=only_positive,
    )

    grad = torch.autograd.grad(energy, x_var, create_graph=False, retain_graph=False)[0]

    if grad_clip is not None:
        grad_norm = grad.norm()
        if grad_norm > grad_clip:
            grad = grad * (grad_clip / (grad_norm + 1e-12))

    x_next = (x_var - step_size * grad).detach()
    return x_next, energy.detach(), grad.detach()
