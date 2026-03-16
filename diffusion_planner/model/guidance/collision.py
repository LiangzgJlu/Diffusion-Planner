import torch
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

from diffusion_planner.model.guidance.gradient_guidance import (
    collision_avoidance_energy_from_signed_distance,
)

ego_size = [get_pacifica_parameters().length, get_pacifica_parameters().width]

COG_TO_REAR = 1.67
INFLATION = 1.0


def batch_signed_distance_rect(rect1, rect2):
    """
    rect1: [B, 4, 2]
    rect2: [B, 4, 2]

    return [B] (signed distance between two rectangles)
    """
    norm_vec = torch.stack(
        [
            rect1[:, 0] - rect1[:, 1],
            rect1[:, 1] - rect1[:, 2],
            rect2[:, 0] - rect2[:, 1],
            rect2[:, 1] - rect2[:, 2],
        ],
        dim=1,
    )
    norm_vec = norm_vec / torch.norm(norm_vec, dim=2, keepdim=True)

    proj1 = torch.einsum("bij,bkj->bik", norm_vec, rect1)
    proj1_min, proj1_max = proj1.min(dim=2)[0], proj1.max(dim=2)[0]

    proj2 = torch.einsum("bij,bkj->bik", norm_vec, rect2)
    proj2_min, proj2_max = proj2.min(dim=2)[0], proj2.max(dim=2)[0]

    overlap = torch.cat([proj1_min - proj2_max, proj2_min - proj1_max], dim=1)
    positive_distance = torch.where(overlap < 0, 1e5, overlap)

    is_overlap = (overlap < 0).all(dim=1)
    distance = torch.where(
        is_overlap,
        overlap.max(dim=1).values,
        positive_distance.min(dim=1).values,
    )

    return distance


def center_rect_to_points(rect):
    """
    rect: [B, 6] (x, y, cos_h, sin_h, l, w)

    return [B, 4, 2] (4 points of the rectangle)
    """
    xy, cos_h, sin_h, lw = rect[:, :2], rect[:, 2], rect[:, 3], rect[:, 4:]

    rot = torch.stack([cos_h, -sin_h, sin_h, cos_h], dim=1).reshape(-1, 2, 2)
    lw = torch.einsum(
        "bj,ij->bij",
        lw,
        torch.tensor([[1.0, 1], [-1, 1], [-1, -1], [1, -1]], device=lw.device) / 2,
    )
    lw = torch.einsum("bij,bkj->bik", lw, rot)

    return xy[:, None, :] + lw


def collision_guidance_fn(x, t, cond, inputs, *args, **kwargs) -> torch.Tensor:
    """Collision guidance integrated with Eq.(9)-style collision energy."""
    B, P, _, _ = x.shape
    neighbor_current_mask = inputs["neighbor_current_mask"]

    x = x.reshape(B, P, -1, 4)
    mask_diffusion_time = ((t < 0.1) & (t > 0.005)).view(B, 1, 1, 1)
    x = torch.where(mask_diffusion_time, x, x.detach())

    heading = x[:, :, :, 2:].detach()
    heading = heading / torch.norm(heading, dim=-1, keepdim=True).clamp_min(1e-6)
    x = torch.cat([x[:, :, :, :2], heading], dim=-1)

    ego_pred = x[:, :1, 1:, :]
    cos_h, sin_h = ego_pred[..., 2:3], ego_pred[..., 3:4]
    ego_pred = torch.cat(
        [
            ego_pred[..., 0:1] + cos_h * COG_TO_REAR,
            ego_pred[..., 1:2] + sin_h * COG_TO_REAR,
            ego_pred[..., 2:],
        ],
        dim=-1,
    )

    neighbors_pred = x[:, 1:, 1:, :]
    _, pn, horizon, _ = neighbors_pred.shape

    predictions = torch.cat([ego_pred, neighbors_pred.detach()], dim=1)

    lw = torch.cat(
        [
            torch.tensor(ego_size, device=predictions.device)[None, None, :].repeat(B, 1, 1),
            inputs["neighbor_agents_past"][:, :pn, -1, [7, 6]],
        ],
        dim=1,
    )

    bbox = torch.cat(
        [predictions, lw.unsqueeze(2).expand(-1, -1, horizon, -1) + INFLATION],
        dim=-1,
    )
    bbox = center_rect_to_points(bbox.reshape(-1, 6)).reshape(B, pn + 1, horizon, 4, 2)

    ego_bbox = bbox[:, :1].expand(-1, pn, -1, -1, -1)
    nbr_bbox = bbox[:, 1:]

    distances = batch_signed_distance_rect(
        ego_bbox.reshape(-1, 4, 2),
        nbr_bbox.reshape(-1, 4, 2),
    ).reshape(B, pn, horizon)

    valid_mask = (~neighbor_current_mask).unsqueeze(-1).expand(-1, -1, horizon)

    # invalid neighbor slots are treated as safely far away
    safe_distance = torch.full_like(distances, 10.0)
    distances = torch.where(valid_mask, distances, safe_distance)

    batch_energy = []
    for b in range(B):
        energy = collision_avoidance_energy_from_signed_distance(
            signed_distance=distances[b],
            omega_c=10.0,
            r=3.0,
            eps=1e-6,
        )
        batch_energy.append(energy)

    batch_energy = torch.stack(batch_energy, dim=0)
    return -batch_energy
