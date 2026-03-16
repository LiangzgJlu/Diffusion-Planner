import torch

from diffusion_planner.model.guidance.gradient_guidance import drivable_area_energy


def _compute_route_signed_distance(ego_xy: torch.Tensor, route_lanes: torch.Tensor) -> torch.Tensor:
    """Approximate signed distance to route-lane drivable corridor.

    Args:
        ego_xy: [B, T, 2]
        route_lanes: [B, R, V, 12], where [:, :, :, :2] is centerline point,
            [:, :, :, 4:6] is vector to left boundary,
            [:, :, :, 6:8] is vector to right boundary.

    Returns:
        signed_distance: [B, T], positive means outside drivable corridor.
    """
    center_xy = route_lanes[..., :2]  # [B,R,V,2]
    to_left = route_lanes[..., 4:6]   # [B,R,V,2]
    to_right = route_lanes[..., 6:8]  # [B,R,V,2]

    # lane half-width estimate at each sampled center point
    half_width = torch.minimum(torch.norm(to_left, dim=-1), torch.norm(to_right, dim=-1))  # [B,R,V]

    # distance from query points to all route lane sampled points
    diff = ego_xy[:, :, None, None, :] - center_xy[:, None, :, :, :]  # [B,T,R,V,2]
    dist = torch.norm(diff, dim=-1)  # [B,T,R,V]

    signed = dist - half_width[:, None, :, :]  # [B,T,R,V]
    signed_min = signed.amin(dim=(-2, -1))  # nearest route-lane support point
    return signed_min


def drivable_guidance_fn(x, t, cond, inputs, *args, **kwargs) -> torch.Tensor:
    """Drivable-area guidance using route-lane geometry and Eq.(12)-style energy."""
    if "route_lanes" not in inputs:
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

    B = x.shape[0]
    x = x.reshape(B, x.shape[1], -1, 4)

    # Only guide during middle diffusion times.
    mask_diffusion_time = ((t < 0.1) & (t > 0.005)).view(B, 1, 1, 1)
    x = torch.where(mask_diffusion_time, x, x.detach())

    ego_xy = x[:, 0, 1:, :2]  # [B,T,2]
    route_lanes = inputs["route_lanes"]

    # Keep non-empty route lanes only.
    route_valid = torch.norm(route_lanes[..., :2], dim=-1).sum(dim=-1) > 0  # [B,R]
    if not route_valid.any():
        return torch.zeros(B, device=x.device, dtype=x.dtype)

    safe_route = route_lanes.clone()
    safe_route[~route_valid] = 1e6

    signed_distance = _compute_route_signed_distance(ego_xy, safe_route)

    batch_energy = []
    for b in range(B):
        batch_energy.append(
            drivable_area_energy(
                map_values=signed_distance[b],
                omega_d=10.0,
                eps=1e-6,
                only_positive=True,
            )
        )

    return -torch.stack(batch_energy, dim=0)
