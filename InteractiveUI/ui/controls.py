import math
from dataclasses import dataclass
KEYS = frozenset("wasdijkl")

@dataclass
class Camera:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0

    def step(self, keys: set[str], dt: float, speed: float, look_speed: float) -> None:

        turn = (int("l" in keys) - int("j" in keys)) * look_speed * dt
        midpoint = self.yaw + turn / 2
        self.yaw += turn
        self.pitch = max(-math.radians(80), min(math.radians(80),
            self.pitch + (int("i" in keys) - int("k" in keys)) * look_speed * dt))
        right = int("d" in keys) - int("a" in keys)
        forward = int("w" in keys) - int("s" in keys)
        length = max(1.0, math.hypot(right, forward))


        arc = math.sin(turn / 2) / (turn / 2) if abs(turn) > 1e-8 else 1.0
        distance = speed * dt * arc / length
        self.x += (right * math.cos(midpoint) + forward * math.sin(midpoint)) * distance
        self.z += (forward * math.cos(midpoint) - right * math.sin(midpoint)) * distance

    def sample(self) -> dict[str, float]:
        return {key: getattr(self, key) for key in ("x", "y", "z", "yaw", "pitch")}


def parse_input(message: dict) -> tuple[set[str], float, float]:
    keys = message.get("keys", [])
    if not isinstance(keys, list) or len(keys) > 8 or any(not isinstance(k, str) or k not in KEYS for k in keys):
        raise ValueError("keys must contain only w/a/s/d/i/j/k/l")
    speed, look = float(message.get("speed", 3)), float(message.get("lookSpeed", 60))
    if not math.isfinite(speed) or not math.isfinite(look) or not 0.1 <= speed <= 12 or not 5 <= look <= 180:
        raise ValueError("speed must be 0.1–12; lookSpeed must be 5–180 degrees/s")
    return set(keys), speed, math.radians(look)
