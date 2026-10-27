"""Pure tensor FORGE reward/control equations (metres, newtons, radians)."""
import torch


def logistic_kernel(distance, a, b):
    # Stable equivalent of 1 / (exp(-a*d) + b + exp(a*d)).
    z = torch.exp(-a * distance.abs())
    return z / (1.0 + b * z + z.square())


def force_excess(force, threshold):
    return (torch.linalg.vector_norm(force, dim=-1) - threshold).clamp_min(0.0)


def dead_zone(force, width):
    return force.sign() * (force.abs() - width).clamp_min(0.0)


def bounded_reference(position, velocity, goal, dt, speed, acceleration, bandwidth=15.0):
    """Critically damped reference; vector speed and acceleration bounds, no snapping."""
    desired_acc = bandwidth**2 * (goal-position) - 2*bandwidth*velocity
    desired_acc *= (acceleration / desired_acc.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.)
    new_velocity = velocity + dt*desired_acc
    new_velocity *= (speed / new_velocity.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.)
    return position + dt*new_velocity, new_velocity


def pitch_cost(angle, tolerance, scale):
    return ((angle.abs()-tolerance).clamp_min(0.) / scale).square().clamp(max=8.)
