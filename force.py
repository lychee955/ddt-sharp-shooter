"""The original fitted motion model, with a checked positive-time solution."""

from dataclasses import dataclass
import math

from scipy.optimize import brentq


DRAG = 1.05235296
WIND_ACCELERATION = 5.50186622
GRAVITY = -163.56591668


@dataclass(frozen=True)
class ShotSolution:
    force: float
    flight_time: float
    residual: float


def trajectory(force, degree, wind, flight_time):
    """Return the position in the fitted model's distance units."""
    angle = math.radians(min(degree, 180 - degree))
    a = -math.expm1(-DRAG * flight_time) / DRAG
    b = (flight_time - a) / DRAG
    return (force * math.cos(angle) * a + WIND_ACCELERATION * wind * b,
            force * math.sin(angle) * a + GRAVITY * b)


def solve_force(deg: float, wind: float, dx: float, dy: float) -> ShotSolution:
    if not all(math.isfinite(value) for value in (deg, wind, dx, dy)):
        raise ValueError("角度、风力和距离必须是有限数值")
    if not 0 <= deg <= 180 or dx < 0:
        raise ValueError("角度应在 0 到 180 度之间，水平距离不能为负")
    angle = math.radians(min(deg, 180 - deg))
    cosine, sine = math.cos(angle), math.sin(angle)
    acceleration = WIND_ACCELERATION * wind
    # x = force*cos(angle)*A(t) + acceleration*B(t)
    # y = force*sin(angle)*A(t) + gravity*B(t)
    # Eliminating force gives a monotonic scalar equation for B(t).
    if abs(cosine) < 1e-8:
        if abs(acceleration) < 1e-8:
            raise ValueError("90 度下无法确定唯一力度，请调整角度")
        target_b = dx / acceleration
    else:
        tangent = sine / cosine
        denominator = GRAVITY - acceleration * tangent
        if abs(denominator) < 1e-10:
            raise ValueError("当前角度和风力没有唯一轨迹解")
        target_b = (dy - dx * tangent) / denominator
    if not math.isfinite(target_b) or target_b <= 0:
        raise ValueError("当前角度无法到达该目标，请调整角度")

    def equation(t):
        a = -math.expm1(-DRAG * t) / DRAG
        return (t - a) / DRAG - target_b

    upper = max(1.0, DRAG * target_b + 1.0 / DRAG + 1.0)
    flight_time, result = brentq(equation, 0.0, upper, full_output=True, xtol=1e-12)
    if not result.converged or flight_time <= 0:
        raise ValueError("力度求解未收敛")
    a = -math.expm1(-DRAG * flight_time) / DRAG
    if abs(cosine) < 1e-8:
        force = (dy - GRAVITY * target_b) / (sine * a)
    else:
        force = (dx - acceleration * target_b) / (cosine * a)
    if not math.isfinite(force) or force <= 0:
        raise ValueError("当前角度没有有效的正力度解")
    actual_x, actual_y = trajectory(force, deg, wind, flight_time)
    residual = math.hypot(actual_x - dx, actual_y - dy)
    if residual > 1e-6 * max(1.0, abs(dx), abs(dy)):
        raise ValueError("轨迹误差过大，请调整角度")
    return ShotSolution(force, flight_time, residual)


def calc_force(deg: float, wind: float, dx: float, dy: float, *, clamp=True):
    """Keep the legacy numeric API; analysis uses solve_force without a cap."""
    force = solve_force(deg, wind, dx, dy).force
    return min(force, 100.0) if clamp else force
