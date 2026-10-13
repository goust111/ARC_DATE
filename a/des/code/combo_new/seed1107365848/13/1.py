import random
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_VEC = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
SIDES = tuple(SIDE_TO_VEC)
BG = 0
GRID_SIDE_RANGE = (12, 30)
OUTER_SIZE_RANGE = (5, 11)
GEN_RETRIES = 400
PLACEMENT_RETRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _contiguous_run(indices: list[int]) -> bool:
    return bool(indices) and indices == list(range(indices[0], indices[-1] + 1))


def _ray_cells(side: str, center_row: int, center_col: int, length: int) -> list[tuple[int, int]]:
    dr, dc = SIDE_TO_VEC[side]
    return [(center_row + dr * step, center_col + dc * step) for step in range(1, length + 1)]


def _analyze_frame(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    outline_color: int,
    bg: int,
) -> dict[str, object] | None:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    for r, c in cells:
        if r not in (r0, r1) and c not in (c0, c1):
            return None

    if (
        int(grid[r0, c0]) != outline_color
        or int(grid[r0, c1]) != outline_color
        or int(grid[r1, c0]) != outline_color
        or int(grid[r1, c1]) != outline_color
    ):
        return None

    side_data = {
        "top": [int(v) for v in grid[r0, c0 + 1 : c1]],
        "bottom": [int(v) for v in grid[r1, c0 + 1 : c1]],
        "left": [int(v) for v in grid[r0 + 1 : r1, c0]],
        "right": [int(v) for v in grid[r0 + 1 : r1, c1]],
    }

    gap_info: dict[str, tuple[int, int]] = {}
    for side, values in side_data.items():
        if any(v not in (outline_color, bg) for v in values):
            return None
        missing = [idx for idx, value in enumerate(values, start=1) if value == bg]
        if not missing:
            continue
        if len(missing) > 3 or not _contiguous_run(missing):
            return None
        gap_info[side] = (missing[0], missing[-1])

    if len(gap_info) != 1:
        return None

    for side, values in side_data.items():
        if side in gap_info:
            start, end = gap_info[side]
            for idx, value in enumerate(values, start=1):
                if start <= idx <= end:
                    if value != bg:
                        return None
                elif value != outline_color:
                    return None
        elif any(value != outline_color for value in values):
            return None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    seed_coords = np.argwhere((inner != bg) & (inner != outline_color))
    if seed_coords.shape[0] != 1:
        return None
    seed_r = int(seed_coords[0, 0] + r0 + 1)
    seed_c = int(seed_coords[0, 1] + c0 + 1)
    seed_color = int(grid[seed_r, seed_c])

    side = next(iter(gap_info))
    start, end = gap_info[side]
    length = end - start + 1
    if side in ("top", "bottom"):
        gap_center_row = r0 if side == "top" else r1
        gap_center_col = c0 + (start + end) // 2
    else:
        gap_center_row = r0 + (start + end) // 2
        gap_center_col = c0 if side == "left" else c1

    return {
        "bbox": (r0, c0, r1, c1),
        "gap_side": side,
        "gap_width": length,
        "seed_color": seed_color,
        "ray_cells": _ray_cells(side, gap_center_row, gap_center_col, length),
        "area": len(cells),
    }


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    bg = _get_bg(grid)
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    best: list[dict[str, object]] = []
    best_score = (-1, -1)
    for color in colors:
        infos = []
        for comp in _components(grid == color):
            info = _analyze_frame(grid, comp, color, bg)
            if info is not None:
                infos.append(info)
        if not infos:
            continue
        score = (sum(int(info["area"]) for info in infos), len(infos))
        if score > best_score:
            best_score = score
            best = infos
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a short outward ray from each rectangular frame gap using the interior seed color.

    Input structure:
    - A black background surrounds one or more thin rectangular outline objects in a
      shared frame color.
    - Each rectangle has exactly one contiguous 1-3 cell gap on one side of its border.
    - Each rectangle interior contains exactly one single-pixel seed color that differs
      from the outline and background.

    Rule:
    - Detect each rectangular outline, locate which side contains the gap, and measure
      the gap width.
    - Use the interior seed color to draw a ray from the gap center outward along the
      gap's normal direction.
    - The ray length equals the gap width, while the original outline, gap, and seed
      pixels remain unchanged.

    Key insight:
    - The seed color, gap direction, and gap width encode three independent pieces of
      information: color, orientation, and ray length.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for info in _find_frames(g):
        seed_color = int(info["seed_color"])
        for r, c in info["ray_cells"]:
            if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
                out[r, c] = seed_color

    return out


def _spec_ray_cells(spec: dict[str, object]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = spec["bbox"]
    side = str(spec["gap_side"])
    start = int(spec["gap_start"])
    width = int(spec["gap_width"])
    end = start + width - 1
    if side == "top":
        return _ray_cells(side, r0, c0 + (start + end) // 2, width)
    if side == "bottom":
        return _ray_cells(side, r1, c0 + (start + end) // 2, width)
    if side == "left":
        return _ray_cells(side, r0 + (start + end) // 2, c0, width)
    return _ray_cells(side, r0 + (start + end) // 2, c1, width)


def _reserve_cells(h: int, w: int, spec: dict[str, object]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = spec["bbox"]
    ray = _spec_ray_cells(spec)
    footprint = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    footprint.update(ray)
    reserved: set[tuple[int, int]] = set()
    for r, c in footprint:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    reserved.add((nr, nc))
    return reserved


def _random_spec(rng: random.Random, h: int, w: int, seed_color: int) -> dict[str, object] | None:
    side = rng.choice(SIDES)
    gap_width = rng.randint(1, 3)

    max_outer_h = min(OUTER_SIZE_RANGE[1], h - 2)
    max_outer_w = min(OUTER_SIZE_RANGE[1], w - 2)
    if max_outer_h < OUTER_SIZE_RANGE[0] or max_outer_w < OUTER_SIZE_RANGE[0]:
        return None

    outer_h = rng.randint(OUTER_SIZE_RANGE[0], max_outer_h)
    outer_w = rng.randint(OUTER_SIZE_RANGE[0], max_outer_w)
    if side in ("top", "bottom") and outer_w < gap_width + 2:
        return None
    if side in ("left", "right") and outer_h < gap_width + 2:
        return None

    top_margin = gap_width if side == "top" else 1
    bottom_margin = gap_width if side == "bottom" else 1
    left_margin = gap_width if side == "left" else 1
    right_margin = gap_width if side == "right" else 1

    max_r0 = h - bottom_margin - outer_h
    max_c0 = w - right_margin - outer_w
    if top_margin > max_r0 or left_margin > max_c0:
        return None

    r0 = rng.randint(top_margin, max_r0)
    c0 = rng.randint(left_margin, max_c0)
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1

    if side in ("top", "bottom"):
        gap_start = rng.randint(1, outer_w - gap_width - 1)
    else:
        gap_start = rng.randint(1, outer_h - gap_width - 1)

    seed_r = rng.randint(r0 + 1, r1 - 1)
    seed_c = rng.randint(c0 + 1, c1 - 1)

    return {
        "bbox": (r0, c0, r1, c1),
        "gap_side": side,
        "gap_start": gap_start,
        "gap_width": gap_width,
        "seed_pos": (seed_r, seed_c),
        "seed_color": seed_color,
    }


def _draw_spec(grid: np.ndarray, outline_color: int, spec: dict[str, object]) -> None:
    r0, c0, r1, c1 = spec["bbox"]
    side = str(spec["gap_side"])
    gap_start = int(spec["gap_start"])
    gap_width = int(spec["gap_width"])
    gap_end = gap_start + gap_width - 1

    grid[r0, c0 : c1 + 1] = outline_color
    grid[r1, c0 : c1 + 1] = outline_color
    grid[r0 : r1 + 1, c0] = outline_color
    grid[r0 : r1 + 1, c1] = outline_color

    if side == "top":
        grid[r0, c0 + gap_start : c0 + gap_end + 1] = BG
    elif side == "bottom":
        grid[r1, c0 + gap_start : c0 + gap_end + 1] = BG
    elif side == "left":
        grid[r0 + gap_start : r0 + gap_end + 1, c0] = BG
    else:
        grid[r0 + gap_start : r0 + gap_end + 1, c1] = BG

    seed_r, seed_c = spec["seed_pos"]
    grid[seed_r, seed_c] = int(spec["seed_color"])


def _expected_from_specs(grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        for r, c in _spec_ray_cells(spec):
            out[r, c] = int(spec["seed_color"])
    return out


def _attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_SIDE_RANGE)
    w = rng.randint(*GRID_SIDE_RANGE)
    area = h * w
    max_rects = 1 + int(area >= 220) + int(area >= 380) + int(area >= 560)
    rect_count = rng.randint(1, max_rects)

    palette = [c for c in range(1, 10)]
    outline_color = rng.choice(palette)
    seed_palette = [c for c in palette if c != outline_color]
    rng.shuffle(seed_palette)
    if rect_count > len(seed_palette):
        return None

    specs: list[dict[str, object]] = []
    reserved = np.zeros((h, w), dtype=bool)
    for seed_color in seed_palette[:rect_count]:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            spec = _random_spec(rng, h, w, seed_color)
            if spec is None:
                continue
            reserve = _reserve_cells(h, w, spec)
            if any(reserved[r, c] for r, c in reserve):
                continue
            for r, c in reserve:
                reserved[r, c] = True
            specs.append(spec)
            placed = True
            break
        if not placed:
            return None

    grid = np.full((h, w), BG, dtype=int)
    for spec in specs:
        _draw_spec(grid, outline_color, spec)

    expected = _expected_from_specs(grid, specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None

    found = _find_frames(grid)
    if len(found) != rect_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
