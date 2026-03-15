"""Run a minimal demo for the newly integrated guidance energy functions."""

import torch

from diffusion_planner.model.guidance.gradient_guidance import (
    collision_avoidance_energy_from_signed_distance,
    drivable_area_guidance_step,
    gradient_guidance_step,
)


def run_collision_demo() -> None:
    print("=== Collision guidance demo ===")

    # x: [B, T, 2]
    x = torch.tensor(
        [[[0.0, 0.0], [0.5, 0.2], [1.0, 0.4], [1.5, 0.6]]],
        dtype=torch.float32,
    )

    obstacle = torch.tensor([1.0, 0.5], dtype=torch.float32)

    def signed_distance_fn(x_state: torch.Tensor) -> torch.Tensor:
        # Signed distance to a circular obstacle (radius=0.6)
        return torch.linalg.norm(x_state - obstacle, dim=-1) - 0.6

    energy_before = collision_avoidance_energy_from_signed_distance(signed_distance_fn(x))
    x_next, energy_step, grad = gradient_guidance_step(
        x=x,
        signed_distance_fn=signed_distance_fn,
        step_size=0.05,
    )
    energy_after = collision_avoidance_energy_from_signed_distance(signed_distance_fn(x_next))

    print(f"energy_before: {energy_before.item():.6f}")
    print(f"energy_step:   {energy_step.item():.6f}")
    print(f"energy_after:  {energy_after.item():.6f}")
    print(f"grad_norm:     {grad.norm().item():.6f}")


def run_drivable_demo() -> None:
    print("\n=== Drivable area guidance demo ===")

    # Simple 32x32 map, positive values represent off-road cost.
    h, w = 32, 32
    cost_map = torch.zeros((1, h, w), dtype=torch.float32)
    cost_map[:, :5, :] = 1.0
    cost_map[:, -5:, :] = 1.0
    cost_map[:, :, :5] = 1.0
    cost_map[:, :, -5:] = 1.0

    ego_xy = torch.tensor(
        [[[2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [8.0, 8.0]]],
        dtype=torch.float32,
    )
    origin_xy = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    ego_next, energy_before, grad = drivable_area_guidance_step(
        ego_xy=ego_xy,
        cost_map=cost_map,
        origin_xy=origin_xy,
        resolution=1.0,
        step_size=0.05,
    )

    _, energy_after, _ = drivable_area_guidance_step(
        ego_xy=ego_next,
        cost_map=cost_map,
        origin_xy=origin_xy,
        resolution=1.0,
        step_size=0.0 + 1e-3,
    )

    print(f"energy_before: {energy_before.item():.6f}")
    print(f"energy_after:  {energy_after.item():.6f}")
    print(f"grad_norm:     {grad.norm().item():.6f}")


def main() -> None:
    torch.manual_seed(0)
    run_collision_demo()
    run_drivable_demo()


if __name__ == "__main__":
    main()
