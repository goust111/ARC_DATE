from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


_DIRS = ("up", "down", "left", "right")
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

_GEN_MIN_SIDE = 12
_GEN_MAX_SIDE = 30
_GEN_MAX_FILL = 6
_GEN_CONTAINER_TRIES = 120
_GEN_MAX_ATTEMPTS = 400
_GEN_PADDING = 1


@dataclass(frozen=True)
class ContainerSpec:
    direction: str
    wall_color: int
    fill_color: int
    fill_r0: int
    fill_r1: int
    fill_c0: int
    fill_c1: int

    @property
    def fill_h(self) -> int:
        return self.fill_r1 - self.fill_r0

    @property
    def fill_w(self) -> int:
        return self.fill_c1 - self.fill_c0

    @property
    def fill_area(self) -> int:
        return self.fill_h * self.fill_w


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                pts.append((cr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(pts)
    return components


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), max(rows), min(cols), max(cols)


def _fill_bounds_from_bbox(
    direction: str, r0: int, r1: int, c0: int, c1: int
) -> tuple[int, int, int, int] | None:
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    if direction == "up" and h >= 2 and w >= 3:
        return r0, r1, c0 + 1, c1
    if direction == "down" and h >= 2 and w >= 3:
        return r0 + 1, r1 + 1, c0 + 1, c1
    if direction == "left" and h >= 3 and w >= 2:
        return r0 + 1, r1, c0, c1
    if direction == "right" and h >= 3 and w >= 2:
        return r0 + 1, r1, c0 + 1, c1 + 1
    return None


def _expected_wall_cells(direction: str, r0: int, r1: int, c0: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if direction in {"up", "down"}:
        wall_row = r1 if direction == "up" else r0
        for r in range(r0, r1 + 1):
            cells.add((r, c0))
            cells.add((r, c1))
        for c in range(c0, c1 + 1):
            cells.add((wall_row, c))
    else:
        wall_col = c1 if direction == "left" else c0
        for c in range(c0, c1 + 1):
            cells.add((r0, c))
            cells.add((r1, c))
        for r in range(r0, r1 + 1):
            cells.add((r, wall_col))
    return cells


def _path_clear(
    grid: np.ndarray, direction: str, fill_r0: int, fill_r1: int, fill_c0: int, fill_c1: int, bg: int
) -> bool:
    if direction == "up":
        return bool(np.all(grid[:fill_r0, fill_c0:fill_c1] == bg))
    if direction == "down":
        return bool(np.all(grid[fill_r1:, fill_c0:fill_c1] == bg))
    if direction == "left":
        return bool(np.all(grid[fill_r0:fill_r1, :fill_c0] == bg))
    return bool(np.all(grid[fill_r0:fill_r1, fill_c1:] == bg))


def _container_from_component(
    grid: np.ndarray, points: list[tuple[int, int]], wall_color: int, bg: int
) -> ContainerSpec | None:
    r0, r1, c0, c1 = _bbox(points)
    point_set = set(points)
    matches: list[ContainerSpec] = []

    for direction in _DIRS:
        bounds = _fill_bounds_from_bbox(direction, r0, r1, c0, c1)
        if bounds is None:
            continue
        if point_set != _expected_wall_cells(direction, r0, r1, c0, c1):
            continue

        fill_r0, fill_r1, fill_c0, fill_c1 = bounds
        patch = grid[fill_r0:fill_r1, fill_c0:fill_c1]
        if patch.size == 0:
            continue
        fill_color = int(patch[0, 0])
        if fill_color in {bg, wall_color}:
            continue
        if not np.all(patch == fill_color):
            continue
        if not _path_clear(grid, direction, fill_r0, fill_r1, fill_c0, fill_c1, bg):
            continue

        matches.append(
            ContainerSpec(
                direction=direction,
                wall_color=wall_color,
                fill_color=fill_color,
                fill_r0=fill_r0,
                fill_r1=fill_r1,
                fill_c0=fill_c0,
                fill_c1=fill_c1,
            )
        )

    if len(matches) != 1:
        return None
    return matches[0]


def _find_containers(grid: np.ndarray, bg: int) -> list[ContainerSpec]:
    candidates: list[tuple[int, list[ContainerSpec]]] = []
    for color in (int(v) for v in np.unique(grid) if int(v) != bg):
        specs: list[ContainerSpec] = []
        ok = True
        for comp in _components_mask(grid == color):
            spec = _container_from_component(grid, comp, color, bg)
            if spec is None:
                ok = False
                break
            specs.append(spec)
        if ok and specs:
            candidates.append((color, specs))

    assert candidates, "expected one wall color whose components are all U-shaped containers"
    candidates.sort(key=lambda item: (-len(item[1]), -sum(spec.fill_area for spec in item[1]), item[0]))
    specs = sorted(candidates[0][1], key=lambda spec: (spec.fill_r0, spec.fill_c0, spec.direction))
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pour each container's solid fill out through its opening and let it slide to the grid edge.

    Input structure:
    - The background is the dominant color.
    - Several disconnected grey wall components form axis-aligned U-shaped containers:
      each container is a three-sided rectangular frame with exactly one open side.
    - Every container cavity is completely filled by one solid non-background, non-grey color,
      and the path from the opening to the matching grid edge is unobstructed.

    Rule:
    - Detect the wall color by finding the color whose connected components are all valid U-shaped containers.
    - For each container, infer the opening direction from the missing side of its rectangular frame.
    - Remove the solid fill from the cavity, then translate that rectangular fill rigidly along the opening direction
      until its leading edge reaches the grid boundary.
    - Leave all grey wall pixels fixed in place.

    Key insight:
    - The motion direction is encoded geometrically: the only side not closed by grey wall pixels is the side
      through which the interior color must exit and slide to the edge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _bg_color(g)
    containers = _find_containers(g, bg)

    out = g.copy()
    for spec in containers:
        out[spec.fill_r0:spec.fill_r1, spec.fill_c0:spec.fill_c1] = bg

    h, w = g.shape
    for spec in containers:
        if spec.direction == "up":
            dest = (slice(0, spec.fill_h), slice(spec.fill_c0, spec.fill_c1))
        elif spec.direction == "down":
            dest = (slice(h - spec.fill_h, h), slice(spec.fill_c0, spec.fill_c1))
        elif spec.direction == "left":
            dest = (slice(spec.fill_r0, spec.fill_r1), slice(0, spec.fill_w))
        else:
            dest = (slice(spec.fill_r0, spec.fill_r1), slice(w - spec.fill_w, w))
        out[dest] = spec.fill_color

    return out


def _reserve_bounds(spec: ContainerSpec, h: int, w: int) -> tuple[int, int, int, int]:
    if spec.direction == "up":
        return 0, spec.fill_r1 + 1, spec.fill_c0 - 1, spec.fill_c1 + 1
    if spec.direction == "down":
        return spec.fill_r0 - 1, h, spec.fill_c0 - 1, spec.fill_c1 + 1
    if spec.direction == "left":
        return spec.fill_r0 - 1, spec.fill_r1 + 1, 0, spec.fill_c1 + 1
    return spec.fill_r0 - 1, spec.fill_r1 + 1, spec.fill_c0 - 1, w


def _expanded_bounds(
    bounds: tuple[int, int, int, int], h: int, w: int, pad: int
) -> tuple[int, int, int, int]:
    r0, r1, c0, c1 = bounds
    return max(0, r0 - pad), min(h, r1 + pad), max(0, c0 - pad), min(w, c1 + pad)


def _reserve_is_free(blocked: np.ndarray, bounds: tuple[int, int, int, int], pad: int) -> bool:
    h, w = blocked.shape
    r0, r1, c0, c1 = _expanded_bounds(bounds, h, w, pad)
    return not bool(blocked[r0:r1, c0:c1].any())


def _mark_reserved(blocked: np.ndarray, bounds: tuple[int, int, int, int], pad: int) -> None:
    h, w = blocked.shape
    r0, r1, c0, c1 = _expanded_bounds(bounds, h, w, pad)
    blocked[r0:r1, c0:c1] = True


def _paint_container(grid: np.ndarray, spec: ContainerSpec) -> None:
    fr0, fr1, fc0, fc1 = spec.fill_r0, spec.fill_r1, spec.fill_c0, spec.fill_c1
    grid[fr0:fr1, fc0:fc1] = spec.fill_color

    wall = spec.wall_color
    if spec.direction == "up":
        grid[fr0 : fr1 + 1, fc0 - 1] = wall
        grid[fr0 : fr1 + 1, fc1] = wall
        grid[fr1, fc0 - 1 : fc1 + 1] = wall
    elif spec.direction == "down":
        grid[fr0 - 1 : fr1, fc0 - 1] = wall
        grid[fr0 - 1 : fr1, fc1] = wall
        grid[fr0 - 1, fc0 - 1 : fc1 + 1] = wall
    elif spec.direction == "left":
        grid[fr0 - 1, fc0 : fc1 + 1] = wall
        grid[fr1, fc0 : fc1 + 1] = wall
        grid[fr0 - 1 : fr1 + 1, fc1] = wall
    else:
        grid[fr0 - 1, fc0 - 1 : fc1] = wall
        grid[fr1, fc0 - 1 : fc1] = wall
        grid[fr0 - 1 : fr1 + 1, fc0 - 1] = wall


def _background_is_unique_mode(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    bg_count = int(counts[values == bg][0])
    return all(bg_count > int(count) for value, count in zip(values, counts) if int(value) != bg)


def _sample_container(
    rng: np.random.Generator,
    h: int,
    w: int,
    direction: str,
    wall_color: int,
    fill_color: int,
    blocked: np.ndarray,
) -> ContainerSpec | None:
    max_fill_h = min(_GEN_MAX_FILL, h - 2)
    max_fill_w = min(_GEN_MAX_FILL, w - 2)
    if max_fill_h < 1 or max_fill_w < 1:
        return None

    for _ in range(_GEN_CONTAINER_TRIES):
        fill_h = int(rng.integers(1, max_fill_h + 1))
        fill_w = int(rng.integers(1, max_fill_w + 1))
        fill_r0 = int(rng.integers(1, h - fill_h))
        fill_c0 = int(rng.integers(1, w - fill_w))
        spec = ContainerSpec(
            direction=direction,
            wall_color=wall_color,
            fill_color=fill_color,
            fill_r0=fill_r0,
            fill_r1=fill_r0 + fill_h,
            fill_c0=fill_c0,
            fill_c1=fill_c0 + fill_w,
        )
        if _reserve_is_free(blocked, _reserve_bounds(spec, h, w), _GEN_PADDING):
            return spec
    return None


def _target_container_count(rng: np.random.Generator, h: int, w: int) -> int:
    small_side = min(h, w)
    if small_side < 16:
        choices = np.array([1, 2])
    elif small_side < 22:
        choices = np.array([1, 2, 3])
    else:
        choices = np.array([2, 3, 4])
    return int(rng.choice(choices))


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(_GEN_MIN_SIDE, _GEN_MAX_SIDE + 1))
    w = int(rng.integers(_GEN_MIN_SIDE, _GEN_MAX_SIDE + 1))
    count = _target_container_count(rng, h, w)

    palette = [int(v) for v in rng.permutation(10)]
    bg = palette[0]
    wall = palette[1]
    fill_colors = palette[2 : 2 + count]
    directions = [str(v) for v in rng.permutation(np.array(_DIRS, dtype=object))[:count]]

    blocked = np.zeros((h, w), dtype=bool)
    specs: list[ContainerSpec] = []
    for direction, fill_color in zip(directions, fill_colors):
        spec = _sample_container(rng, h, w, direction, wall, fill_color, blocked)
        if spec is None:
            return None
        specs.append(spec)
        _mark_reserved(blocked, _reserve_bounds(spec, h, w), _GEN_PADDING)

    grid = np.full((h, w), bg, dtype=int)
    for spec in specs:
        _paint_container(grid, spec)

    if not _background_is_unique_mode(grid, bg):
        return None

    try:
        detected = _find_containers(grid, bg)
        out = main(grid)
    except AssertionError:
        return None

    if len(detected) != count:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid container-pouring instance")
