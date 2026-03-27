from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ArmGeometry:
    sh_half: float = 0.05
    upper_len: float = 0.14
    fore_len: float = 0.18
    fore_angle: float = 0.281
    shoulder_min: float = -2.30
    shoulder_max: float = 2.30


@dataclass(frozen=True)
class IKConfig:
    max_iterations: int = 12
    damping: float = 1.0e-3
    finite_diff_eps: float = 1.0e-4
    convergence_tol: float = 1.0e-4
    final_error_tol: float = 0.002
    shoulder_limit_margin: float = 0.12
    max_step: float = 0.25


@dataclass(frozen=True)
class IKResult:
    theta_l: float
    theta_r: float
    error_norm: float


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _inside_margin(theta_l: float, theta_r: float, geom: ArmGeometry, margin: float) -> bool:
    return (
        geom.shoulder_min + margin <= theta_l <= geom.shoulder_max - margin
        and geom.shoulder_min + margin <= theta_r <= geom.shoulder_max - margin
    )


def forward_point(theta_l: float, theta_r: float, geom: ArmGeometry = ArmGeometry()) -> tuple[float, float] | None:
    """Return pen-mount XY in arm-local coordinates or None if unreachable."""
    s_lx, s_ly = -geom.sh_half, 0.0
    s_rx, s_ry = geom.sh_half, 0.0

    e_lx = s_lx + geom.upper_len * (-math.sin(theta_l))
    e_ly = s_ly + geom.upper_len * math.cos(theta_l)
    e_rx = s_rx + geom.upper_len * (-math.sin(theta_r))
    e_ry = s_ry + geom.upper_len * math.cos(theta_r)

    dx = e_rx - e_lx
    dy = e_ry - e_ly
    d = math.hypot(dx, dy)
    if d < 1e-9 or d > 2.0 * geom.fore_len:
        return None

    a = d / 2.0
    h2 = geom.fore_len * geom.fore_len - a * a
    if h2 < 0.0:
        return None

    h = math.sqrt(h2)
    mx = (e_lx + e_rx) / 2.0
    my = (e_ly + e_ry) / 2.0
    ux, uy = dx / d, dy / d
    nx, ny = -uy, ux

    p1x, p1y = mx + h * nx, my + h * ny
    p2x, p2y = mx - h * nx, my - h * ny
    if p1y >= p2y:
        return p1x, p1y
    return p2x, p2y


def _jacobian(theta_l: float, theta_r: float, geom: ArmGeometry, eps: float) -> tuple[tuple[float, float], tuple[float, float]] | None:
    base = forward_point(theta_l, theta_r, geom)
    if base is None:
        return None

    l_plus = forward_point(theta_l + eps, theta_r, geom)
    l_minus = forward_point(theta_l - eps, theta_r, geom)
    r_plus = forward_point(theta_l, theta_r + eps, geom)
    r_minus = forward_point(theta_l, theta_r - eps, geom)
    if None in (l_plus, l_minus, r_plus, r_minus):
        return None

    j11 = (l_plus[0] - l_minus[0]) / (2.0 * eps)
    j21 = (l_plus[1] - l_minus[1]) / (2.0 * eps)
    j12 = (r_plus[0] - r_minus[0]) / (2.0 * eps)
    j22 = (r_plus[1] - r_minus[1]) / (2.0 * eps)
    return ((j11, j12), (j21, j22))


def solve_ik(
    target_xy: tuple[float, float],
    seed: tuple[float, float],
    geom: ArmGeometry = ArmGeometry(),
    config: IKConfig = IKConfig(),
) -> IKResult | None:
    theta_l = _clamp(float(seed[0]), geom.shoulder_min, geom.shoulder_max)
    theta_r = _clamp(float(seed[1]), geom.shoulder_min, geom.shoulder_max)
    last_error_norm = float('inf')
    stagnation_count = 0

    for _ in range(max(1, config.max_iterations)):
        current = forward_point(theta_l, theta_r, geom)
        if current is None:
            return None

        err_x = float(target_xy[0]) - current[0]
        err_y = float(target_xy[1]) - current[1]
        error_norm = math.hypot(err_x, err_y)
        if error_norm <= config.convergence_tol and _inside_margin(
            theta_l, theta_r, geom, config.shoulder_limit_margin
        ):
            return IKResult(theta_l=theta_l, theta_r=theta_r, error_norm=error_norm)

        if error_norm >= last_error_norm - 1.0e-6:
            stagnation_count += 1
            if stagnation_count >= 2:
                break
        else:
            stagnation_count = 0
        last_error_norm = error_norm

        jacobian = _jacobian(theta_l, theta_r, geom, config.finite_diff_eps)
        if jacobian is None:
            return None
        (j11, j12), (j21, j22) = jacobian

        a = j11 * j11 + j21 * j21 + config.damping
        b = j11 * j12 + j21 * j22
        c = j12 * j12 + j22 * j22 + config.damping
        det = a * c - b * b
        if abs(det) < 1.0e-12:
            return None

        rhs1 = j11 * err_x + j21 * err_y
        rhs2 = j12 * err_x + j22 * err_y
        delta_l = (c * rhs1 - b * rhs2) / det
        delta_r = (a * rhs2 - b * rhs1) / det

        max_delta = max(abs(delta_l), abs(delta_r))
        if max_delta > config.max_step:
            scale = config.max_step / max_delta
            delta_l *= scale
            delta_r *= scale

        theta_l = _clamp(theta_l + delta_l, geom.shoulder_min, geom.shoulder_max)
        theta_r = _clamp(theta_r + delta_r, geom.shoulder_min, geom.shoulder_max)

    final_point = forward_point(theta_l, theta_r, geom)
    if final_point is None:
        return None
    final_error = math.hypot(float(target_xy[0]) - final_point[0], float(target_xy[1]) - final_point[1])
    if final_error > config.final_error_tol:
        return None
    if not _inside_margin(theta_l, theta_r, geom, config.shoulder_limit_margin):
        return None
    return IKResult(theta_l=theta_l, theta_r=theta_r, error_norm=final_error)


def solve_best_ik(
    target_xy: tuple[float, float],
    seeds: list[tuple[float, float]],
    geom: ArmGeometry = ArmGeometry(),
    config: IKConfig = IKConfig(),
) -> IKResult | None:
    best = None
    seen = set()
    for seed in seeds:
        key = (round(float(seed[0]), 6), round(float(seed[1]), 6))
        if key in seen:
            continue
        seen.add(key)
        result = solve_ik(target_xy, seed, geom, config)
        if result is None:
            continue
        if best is None or result.error_norm < best.error_norm:
            best = result
    return best
