from __future__ import annotations

from pathlib import Path
import itertools
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask, get_bg


def _normalize_coords(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    r0 = min(rs)
    c0 = min(cs)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _is_connected(coords: tuple[tuple[int, int], ...]) -> bool:
    cells = set(coords)
    stack = [coords[0]]
    seen = {coords[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(coords)


def _enumerate_shapes() -> dict[int, list[tuple[tuple[int, int], ...]]]:
    cells = [(r, c) for r in range(3) for c in range(3)]
    by_area: dict[int, list[tuple[tuple[int, int], ...]]] = {n: [] for n in range(3, 7)}
    for area in range(3, 7):
        seen: set[tuple[tuple[int, int], ...]] = set()
        for combo in itertools.combinations(cells, area):
            norm = _normalize_coords(combo)
            if norm in seen or not _is_connected(norm):
                continue
            seen.add(norm)
            by_area[area].append(norm)
    return by_area


SHAPES_BY_AREA = _enumerate_shapes()
GRID_SIDE_MIN = 11
GRID_SIDE_MAX = 17
CENTER_JITTER = 2
GENERATION_ATTEMPTS = 800
SCENARIOS = (
    {"name": "corner_shape", "corner_mode": "corner", "need_shape": True, "need_ray": False},
    {"name": "edge_shape", "corner_mode": "edge", "need_shape": True, "need_ray": False},
    {"name": "mixed", "corner_mode": "either", "need_shape": True, "need_ray": True},
)
SCENARIO_WEIGHTS = (2, 2, 3)


def _shape_dims(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return int(max(rs) + 1), int(max(cs) + 1)


def _border_cycle(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    cycle: list[tuple[int, int]] = []
    cycle.extend((0, c) for c in range(1, w))
    cycle.extend((r, w - 1) for r in range(1, h))
    cycle.extend((h - 1, c) for c in range(w - 2, -1, -1))
    cycle.extend((r, 0) for r in range(h - 2, -1, -1))
    assert len(cycle) == 2 * h + 2 * w - 4
    return cycle


def _inward_direction(coord: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r, c = coord
    h, w = map(int, shape)
    if r == 0 and c > 0:
        return 1, 0
    if c == w - 1 and r > 0:
        return 0, -1
    if r == h - 1 and c < w - 1:
        return -1, 0
    assert c == 0 and r < h - 1
    return 0, 1


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 7 and w >= 7
    assert int(get_bg(g)) == int(Color.BLACK)

    colors = set(map(int, np.unique(g)))
    assert colors <= {int(Color.BLACK), int(Color.BLUE), int(Color.ORANGE)}

    blue = np.argwhere(g == Color.BLUE)
    assert blue.shape == (1, 2)
    start = tuple(map(int, blue[0]))
    assert start[0] in (0, h - 1) or start[1] in (0, w - 1)

    shape_mask = g == Color.ORANGE
    comps = components_mask(shape_mask, connectivity=4)
    assert len(comps) == 1
    shape_coords = comps[0]
    n = int(shape_coords.shape[0])
    assert 3 <= n <= 6
    assert np.all((shape_coords[:, 0] >= 1) & (shape_coords[:, 0] <= h - 2))
    assert np.all((shape_coords[:, 1] >= 1) & (shape_coords[:, 1] <= w - 2))
    return shape_mask, start, n


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    shape_mask, start, n = _parse_input(g)
    h, w = g.shape

    cycle = _border_cycle(g.shape)
    start_idx = {coord: idx for idx, coord in enumerate(cycle)}[start]
    ordered = cycle[start_idx:] + cycle[:start_idx]
    activated = ordered[::n]

    out = g.copy()
    for r, c in activated:
        out[r, c] = Color.ORANGE

    ray_mask = np.zeros_like(shape_mask, dtype=bool)
    stop_reasons: list[str] = []
    ray_lengths: list[int] = []

    for source in activated:
        dr, dc = _inward_direction(source, g.shape)
        r, c = source
        length = 0
        while True:
            nr = r + dr
            nc = c + dc
            if not (0 < nr < h - 1 and 0 < nc < w - 1):
                stop_reasons.append("opposite_border")
                ray_lengths.append(length)
                break
            if shape_mask[nr, nc]:
                stop_reasons.append("shape")
                ray_lengths.append(length)
                break
            if ray_mask[nr, nc]:
                stop_reasons.append("ray")
                ray_lengths.append(length)
                break
            out[nr, nc] = Color.ORANGE
            ray_mask[nr, nc] = True
            length += 1
            r, c = nr, nc

    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    meta: dict[str, object] = {
        "start": start,
        "start_is_corner": start in corners,
        "n": n,
        "activated": activated,
        "activation_count": len(activated),
        "corner_count": sum(coord in corners for coord in activated),
        "stop_reasons": tuple(stop_reasons),
        "ray_lengths": tuple(ray_lengths),
        "orange_ratio": float(np.count_nonzero(out == Color.ORANGE) / out.size),
        "shape_pixels": int(np.count_nonzero(shape_mask)),
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the central orange contour, repeat orange activations around the border at that period, and shoot inward orthogonal rays.

    Input structure:
    - The grid has a black background with exactly one blue marker on the outer border.
    - Near the center there is one small 4-connected orange contour shape with 3 to 6 pixels and bbox at most 3x3.
    - All other border and interior cells are black.

    Rule:
    - Count the orange shape pixels to get N.
    - Recolor the blue border marker to orange, then walk clockwise around the border as a ring and activate every N-th border position starting from that marker.
    - For each activated border cell, cast an orange ray inward along the orthogonal normal of its owning side; corners belong to the previous side in clockwise order.
    - A ray stops before the first orange shape pixel, before any already drawn ray pixel, or before the opposite border; the original central shape stays unchanged.

    Key insight:
    - The same central orange object serves two roles at once: its pixel count sets the border period, while its geometry blocks the inward rays at task-defining places.
    """
    out, _meta = _simulate(np.asarray(grid, dtype=int))
    return out


def _center_positions(span: int, size: int) -> list[int]:
    lo = 3
    hi = span - size - 3
    if lo > hi:
        return []
    center = (span - size) // 2
    window_lo = max(lo, center - CENTER_JITTER)
    window_hi = min(hi, center + CENTER_JITTER)
    positions = list(range(window_lo, window_hi + 1))
    if not positions:
        positions = list(range(lo, hi + 1))
    random.shuffle(positions)
    return positions


def _random_start(h: int, w: int, corner_mode: str) -> tuple[int, int]:
    corners = [(0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0)]
    if corner_mode == "either":
        corner_mode = random.choices(["corner", "edge"], weights=[2, 3], k=1)[0]
    if corner_mode == "corner":
        return random.choice(corners)

    side = random.choice(("top", "right", "bottom", "left"))
    if side == "top":
        return 0, random.randint(1, w - 2)
    if side == "right":
        return random.randint(1, h - 2), w - 1
    if side == "bottom":
        return h - 1, random.randint(1, w - 2)
    return random.randint(1, h - 2), 0


def _build_grid(h: int, w: int, start: tuple[int, int], shape_coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for r, c in shape_coords:
        grid[r, c] = Color.ORANGE
    grid[start] = Color.BLUE
    return grid


def _input_semantics_ok(grid: np.ndarray) -> bool:
    try:
        shape_mask, start, n = _parse_input(grid)
    except Exception:
        return False

    h, w = grid.shape
    if int(np.count_nonzero(grid == Color.BLUE)) != 1:
        return False
    if start[0] not in (0, h - 1) and start[1] not in (0, w - 1):
        return False
    if int(np.count_nonzero(shape_mask)) != n:
        return False
    shape_coords = np.argwhere(shape_mask)
    r0 = int(shape_coords[:, 0].min())
    c0 = int(shape_coords[:, 1].min())
    r1 = int(shape_coords[:, 0].max())
    c1 = int(shape_coords[:, 1].max())
    if (r1 - r0 + 1) > 3 or (c1 - c0 + 1) > 3:
        return False
    if r0 < 3 or c0 < 3 or r1 > h - 4 or c1 > w - 4:
        return False
    return True


def _scenario_ok(meta: dict[str, object], scenario: dict[str, object]) -> bool:
    stop_reasons = set(map(str, meta["stop_reasons"]))
    ray_lengths = [int(x) for x in meta["ray_lengths"]]
    start_is_corner = bool(meta["start_is_corner"])

    if scenario["corner_mode"] == "corner" and not start_is_corner:
        return False
    if scenario["corner_mode"] == "edge" and start_is_corner:
        return False

    if scenario["need_shape"] and "shape" not in stop_reasons:
        return False
    if scenario["need_ray"] and "ray" not in stop_reasons:
        return False
    if "opposite_border" not in stop_reasons:
        return False
    if int(meta["activation_count"]) < 3:
        return False
    if max(ray_lengths, default=0) < 2:
        return False
    if float(meta["orange_ratio"]) > 0.42:
        return False
    return True


def _candidate_shapes(area: int) -> list[tuple[tuple[int, int], ...]]:
    shapes = SHAPES_BY_AREA[area][:]
    random.shuffle(shapes)
    return shapes


def _choose_area(scenario: dict[str, object]) -> int:
    if scenario["need_ray"]:
        return random.choices([3, 4, 5, 6], weights=[4, 4, 2, 1], k=1)[0]
    return random.choice([3, 4, 5, 6])


def _gen_attempt() -> np.ndarray | None:
    scenario = random.choices(SCENARIOS, weights=SCENARIO_WEIGHTS, k=1)[0]
    h = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    w = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    area = _choose_area(scenario)
    start = _random_start(h, w, str(scenario["corner_mode"]))

    shape_templates = _candidate_shapes(area)
    for template in shape_templates:
        sh, sw = _shape_dims(template)
        rows = _center_positions(h, sh)
        cols = _center_positions(w, sw)
        if not rows or not cols:
            continue
        for r0 in rows:
            for c0 in cols:
                shape_coords = tuple((r0 + r, c0 + c) for r, c in template)
                grid = _build_grid(h, w, start, shape_coords)
                if not _input_semantics_ok(grid):
                    continue
                try:
                    out, meta = _simulate(grid)
                except Exception:
                    continue
                if not _scenario_ok(meta, scenario):
                    continue
                if np.array_equal(out, grid):
                    continue
                return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
