from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
LEGEND_BORDER = 5
USABLE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (12, 20)
GRID_W_RANGE = (12, 20)
TYPE_COUNT_RANGE = (2, 4)
DOORS_PER_TYPE_RANGE = (1, 3)
LEGEND_OFFSET_CHOICES = (0, 1)
MAX_GENERATION_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _expand_bbox(
    bbox: tuple[int, int, int, int], shape: tuple[int, int], pad: int = 1
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _find_legend(
    grid: np.ndarray,
) -> tuple[dict[int, int], np.ndarray, tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    candidates: list[tuple[tuple[int, int], dict[int, int], np.ndarray, tuple[int, int, int, int]]] = []
    for comp in _components(g == LEGEND_BORDER):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 4:
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(sub, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(sub[border] == LEGEND_BORDER):
            continue
        if np.any(sub[~border] == LEGEND_BORDER):
            continue

        mapping: dict[int, int] = {}
        ok = True
        for r in range(r0 + 1, r1):
            row = g[r, c0 + 1 : c1]
            colored = [int(v) for v in row if int(v) not in (BG, LEGEND_BORDER)]
            if len(colored) != 2:
                ok = False
                break
            pillar, lock = colored
            if pillar == lock or pillar in (BG, LEGEND_BORDER) or lock in (BG, LEGEND_BORDER):
                ok = False
                break
            if pillar in mapping and mapping[pillar] != lock:
                ok = False
                break
            mapping[pillar] = lock
        if not ok or not mapping:
            continue

        legend_mask = np.zeros_like(g, dtype=bool)
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        candidates.append(((r0, c0), mapping, legend_mask, (r0, c0, r1, c1)))

    assert candidates, "legend not found"
    candidates.sort(key=lambda item: item[0])
    _, mapping, legend_mask, bbox_ = candidates[0]
    return mapping, legend_mask, bbox_


def _scan_gates(
    grid: np.ndarray, legend_mask: np.ndarray, pillar_colors: set[int]
) -> list[tuple[int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    gates: list[tuple[int, int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color not in pillar_colors or legend_mask[r, c]:
                continue
            if c + 3 < w and not np.any(legend_mask[r, c : c + 4]):
                if int(g[r, c + 1]) == BG and int(g[r, c + 2]) == BG and int(g[r, c + 3]) == color:
                    gates.append((r, c, r, c + 3, color))
            if r + 3 < h and not np.any(legend_mask[r : r + 4, c]):
                if int(g[r + 1, c]) == BG and int(g[r + 2, c]) == BG and int(g[r + 3, c]) == color:
                    gates.append((r, c, r + 3, c, color))
    return gates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify doors built from same-colored pillars and fill each black channel
    with the lock color defined by the top-left gray-bordered legend.
    """
    g = np.asarray(grid, dtype=int)
    mapping, legend_mask, _ = _find_legend(g)

    out = g.copy()
    for r0, c0, r1, c1, pillar in _scan_gates(g, legend_mask, set(mapping)):
        lock = mapping[pillar]
        if r0 == r1:
            out[r0, c0 + 1 : c1] = lock
        else:
            out[r0 + 1 : r1, c0] = lock
    return out


def _draw_legend(
    grid: np.ndarray, top: int, left: int, mapping_rows: list[tuple[int, int]]
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    g = grid.copy()
    height = len(mapping_rows) + 2
    width = 4
    r0, c0, r1, c1 = top, left, top + height - 1, left + width - 1
    g[r0, c0 : c1 + 1] = LEGEND_BORDER
    g[r1, c0 : c1 + 1] = LEGEND_BORDER
    g[r0 : r1 + 1, c0] = LEGEND_BORDER
    g[r0 : r1 + 1, c1] = LEGEND_BORDER
    for idx, (pillar, lock) in enumerate(mapping_rows):
        rr = r0 + 1 + idx
        g[rr, c0 + 1] = pillar
        g[rr, c0 + 2] = lock
    return g, (r0, c0, r1, c1)


def _place_gate(
    grid: np.ndarray,
    reserved: np.ndarray,
    rng: np.random.Generator,
    pillar: int,
    orientation: str,
) -> tuple[int, int, int, int, int] | None:
    h, w = grid.shape
    for _ in range(PLACEMENT_ATTEMPTS):
        if orientation == "h":
            r = int(rng.integers(0, h))
            c = int(rng.integers(0, w - 3))
            bbox = (r, c, r, c + 3)
            cells = [(r, c + i) for i in range(4)]
        else:
            r = int(rng.integers(0, h - 3))
            c = int(rng.integers(0, w))
            bbox = (r, c, r + 3, c)
            cells = [(r + i, c) for i in range(4)]

        er0, ec0, er1, ec1 = _expand_bbox(bbox, grid.shape, pad=1)
        if np.any(reserved[er0 : er1 + 1, ec0 : ec1 + 1]):
            continue
        if any(int(grid[rr, cc]) != BG for rr, cc in cells):
            continue

        if orientation == "h":
            grid[r, c] = pillar
            grid[r, c + 3] = pillar
            reserved[er0 : er1 + 1, ec0 : ec1 + 1] = True
            return (r, c, r, c + 3, pillar)
        grid[r, c] = pillar
        grid[r + 3, c] = pillar
        reserved[er0 : er1 + 1, ec0 : ec1 + 1] = True
        return (r, c, r + 3, c, pillar)
    return None


def _expected_output(
    grid: np.ndarray, mapping: dict[int, int], gates: list[tuple[int, int, int, int, int]]
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r0, c0, r1, c1, pillar in gates:
        lock = mapping[pillar]
        if r0 == r1:
            out[r0, c0 + 1 : c1] = lock
        else:
            out[r0 + 1 : r1, c0] = lock
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    type_count = int(rng.integers(TYPE_COUNT_RANGE[0], TYPE_COUNT_RANGE[1] + 1))
    colors = list(rng.choice(USABLE_COLORS, size=2 * type_count, replace=False))
    pillars = colors[:type_count]
    locks = colors[type_count:]
    mapping_rows = list(zip(pillars, locks))
    rng.shuffle(mapping_rows)
    mapping = {pillar: lock for pillar, lock in mapping_rows}

    legend_top = int(rng.choice(LEGEND_OFFSET_CHOICES))
    legend_left = int(rng.choice(LEGEND_OFFSET_CHOICES))
    legend_h = type_count + 2
    legend_w = 4

    min_h = max(GRID_H_RANGE[0], legend_top + legend_h + 4)
    min_w = max(GRID_W_RANGE[0], legend_left + legend_w + 4)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None
    h = int(rng.integers(min_h, GRID_H_RANGE[1] + 1))
    w = int(rng.integers(min_w, GRID_W_RANGE[1] + 1))

    grid = np.full((h, w), BG, dtype=int)
    grid, legend_bbox = _draw_legend(grid, legend_top, legend_left, mapping_rows)

    reserved = np.zeros_like(grid, dtype=bool)
    er0, ec0, er1, ec1 = _expand_bbox(legend_bbox, grid.shape, pad=1)
    reserved[er0 : er1 + 1, ec0 : ec1 + 1] = True

    per_type = [int(rng.integers(DOORS_PER_TYPE_RANGE[0], DOORS_PER_TYPE_RANGE[1] + 1)) for _ in range(type_count)]
    door_colors: list[int] = []
    for pillar, count in zip(pillars, per_type):
        door_colors.extend([pillar] * count)
    rng.shuffle(door_colors)

    if len(door_colors) >= 2:
        orientations = ["h", "v"]
        orientations.extend("h" if int(x) == 0 else "v" for x in rng.integers(0, 2, size=len(door_colors) - 2))
        rng.shuffle(orientations)
    else:
        orientations = ["h" if int(rng.integers(0, 2)) == 0 else "v"]

    gates: list[tuple[int, int, int, int, int]] = []
    for pillar, orientation in zip(door_colors, orientations):
        gate = _place_gate(grid, reserved, rng, pillar, orientation)
        if gate is None:
            return None
        gates.append(gate)

    legend_mapping, legend_mask, _ = _find_legend(grid)
    found = sorted(_scan_gates(grid, legend_mask, set(legend_mapping)))
    expected = sorted(gates)
    if found != expected:
        return None

    expected_out = _expected_output(grid, mapping, gates)
    if not np.array_equal(main(grid), expected_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid door-legend puzzle")
