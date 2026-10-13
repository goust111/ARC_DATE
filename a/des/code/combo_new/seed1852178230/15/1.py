import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ("up", "down", "left", "right")
GEN_RETRIES = 400
PLACEMENT_TRIES = 240
GRID_SIZE_RANGE = (16, 30)
RECT_COUNT_CHOICES = (1, 2, 3)
SEED_COUNT_CHOICES = (1, 2, 3)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _contiguous(indices: list[int]) -> bool:
    if not indices:
        return False
    return indices == list(range(indices[0], indices[-1] + 1))


def _centered_span(values: list[int], width: int, limit: int) -> list[int]:
    uniq = sorted(set(values))
    if len(uniq) == width and _contiguous(uniq):
        return uniq
    center = sum(values) / len(values)
    start = int(np.floor(center - (width - 1) / 2.0 + 1e-9))
    start = max(0, min(start, limit - width))
    return list(range(start, start + width))


def _analyze_frame(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a non-empty interior"

    top_missing = [c for c in range(c0 + 1, c1) if int(grid[r0, c]) != GREY]
    bottom_missing = [c for c in range(c0 + 1, c1) if int(grid[r1, c]) != GREY]
    left_missing = [r for r in range(r0 + 1, r1) if int(grid[r, c0]) != GREY]
    right_missing = [r for r in range(r0 + 1, r1) if int(grid[r, c1]) != GREY]

    missing_sides = []
    if top_missing:
        missing_sides.append(("up", top_missing))
    if bottom_missing:
        missing_sides.append(("down", bottom_missing))
    if left_missing:
        missing_sides.append(("left", left_missing))
    if right_missing:
        missing_sides.append(("right", right_missing))
    assert len(missing_sides) == 1, "each frame must have exactly one gap side"

    direction, missing = missing_sides[0]
    assert _contiguous(missing), "gap must be a single contiguous segment"

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    seed_mask = (interior != BLACK) & (interior != GREY)
    seed_coords = np.argwhere(seed_mask)
    assert 1 <= len(seed_coords) <= 3, "expected 1 to 3 seed pixels"
    seed_colors = np.unique(interior[seed_mask])
    assert seed_colors.size == 1, "all seed pixels inside a frame must share one color"

    seed_color = int(seed_colors[0])
    seed_rows = [int(r0 + 1 + rr) for rr, _ in seed_coords]
    seed_cols = [int(c0 + 1 + cc) for _, cc in seed_coords]
    width = len(seed_coords)

    if direction in ("up", "down"):
        ray_cols = _centered_span(seed_cols, width, grid.shape[1])
        start_row = min(seed_rows) if direction == "up" else max(seed_rows)
        ray_rows = range(start_row, -1, -1) if direction == "up" else range(start_row, grid.shape[0])
        ray_cells = [(rr, cc) for rr in ray_rows for cc in ray_cols]
    else:
        ray_rows = _centered_span(seed_rows, width, grid.shape[0])
        start_col = min(seed_cols) if direction == "left" else max(seed_cols)
        ray_cols = range(start_col, -1, -1) if direction == "left" else range(start_col, grid.shape[1])
        ray_cells = [(rr, cc) for rr in ray_rows for cc in ray_cols]

    return {
        "bbox": (r0, c0, r1, c1),
        "direction": direction,
        "seed_color": seed_color,
        "seed_count": width,
        "ray_cells": ray_cells,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire a seed-colored strip ray out of each gapped grey frame, with strip width set by the number of seeds inside.

    Input structure:
    - Black is the background.
    - Several disjoint Grey rectangular frame components appear, and each frame has exactly one contiguous gap on one border side.
    - Inside each frame there are 1 to 3 seed pixels of one non-Black non-Grey color; different frames use different seed colors.

    Rule:
    - For each Grey frame, determine the gap side to get the ray direction: up, down, left, or right.
    - Count the seed pixels inside that frame; this count N is the ray width.
    - Center an N-cell-wide strip on the seed group's center row or center column, then extend it from the seed position through the gap to the grid edge.
    - Paint only Black cells in that strip with the seed color, leaving Grey frame pixels and the original seed pixels unchanged.

    Key insight:
    - The missing border segment contributes direction, while the number of interior seeds contributes geometry, so solving the task requires combining both cues.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for comp in _components(g == GREY):
        info = _analyze_frame(g, comp)
        color = info["seed_color"]
        for r, c in info["ray_cells"]:
            if out[r, c] == BLACK:
                out[r, c] = color
    return out


def _choose_gap_lengths(span: int, seed_count: int) -> list[int]:
    return [length for length in range(seed_count, span + 1) if length % 2 == seed_count % 2]


def _make_candidate(rng: random.Random, h: int, w: int, color: int) -> dict | None:
    direction = rng.choice(DIRECTIONS)
    seed_count = rng.choice(SEED_COUNT_CHOICES)

    if direction in ("up", "down"):
        inner_h = rng.randint(4, 8)
        inner_w = rng.randint(max(seed_count + 1, 3), 9)
        seed_row = rng.randint(1, inner_h - 2)
        seed_start_col = rng.randint(0, inner_w - seed_count)
        gap_lengths = []
        for length in _choose_gap_lengths(inner_w, seed_count):
            gap_start = seed_start_col - (length - seed_count) // 2
            if 0 <= gap_start and gap_start + length <= inner_w:
                gap_lengths.append(length)
        if not gap_lengths:
            return None
        gap_length = rng.choice(gap_lengths)
        gap_start = seed_start_col - (gap_length - seed_count) // 2
        seed_cells = [(1 + seed_row, 1 + seed_start_col + offset) for offset in range(seed_count)]
        if direction == "up":
            gap_cells = [(0, 1 + gap_start + offset) for offset in range(gap_length)]
        else:
            gap_cells = [(inner_h + 1, 1 + gap_start + offset) for offset in range(gap_length)]
    else:
        inner_h = rng.randint(max(seed_count + 1, 3), 9)
        inner_w = rng.randint(4, 8)
        seed_col = rng.randint(1, inner_w - 2)
        seed_start_row = rng.randint(0, inner_h - seed_count)
        gap_lengths = []
        for length in _choose_gap_lengths(inner_h, seed_count):
            gap_start = seed_start_row - (length - seed_count) // 2
            if 0 <= gap_start and gap_start + length <= inner_h:
                gap_lengths.append(length)
        if not gap_lengths:
            return None
        gap_length = rng.choice(gap_lengths)
        gap_start = seed_start_row - (gap_length - seed_count) // 2
        seed_cells = [(1 + seed_start_row + offset, 1 + seed_col) for offset in range(seed_count)]
        if direction == "left":
            gap_cells = [(1 + gap_start + offset, 0) for offset in range(gap_length)]
        else:
            gap_cells = [(1 + gap_start + offset, inner_w + 1) for offset in range(gap_length)]

    rect_h = inner_h + 2
    rect_w = inner_w + 2
    max_r0 = h - rect_h
    max_c0 = w - rect_w
    if max_r0 < 0 or max_c0 < 0:
        return None

    bbox_cells = [(rr, cc) for rr in range(rect_h) for cc in range(rect_w)]
    border_cells = (
        [(0, cc) for cc in range(rect_w)]
        + [(rect_h - 1, cc) for cc in range(rect_w)]
        + [(rr, 0) for rr in range(1, rect_h - 1)]
        + [(rr, rect_w - 1) for rr in range(1, rect_h - 1)]
    )
    border_set = set(border_cells) - set(gap_cells)

    local_seed_rows = [rr for rr, _ in seed_cells]
    local_seed_cols = [cc for _, cc in seed_cells]
    if direction in ("up", "down"):
        ray_cols = _centered_span(local_seed_cols, seed_count, rect_w)
        start_row = min(local_seed_rows) if direction == "up" else max(local_seed_rows)
        ray_rows = range(start_row, -1, -1) if direction == "up" else range(start_row, rect_h)
        ray_local = {(rr, cc) for rr in ray_rows for cc in ray_cols}
    else:
        ray_rows = _centered_span(local_seed_rows, seed_count, rect_h)
        start_col = min(local_seed_cols) if direction == "left" else max(local_seed_cols)
        ray_cols = range(start_col, -1, -1) if direction == "left" else range(start_col, rect_w)
        ray_local = {(rr, cc) for rr in ray_rows for cc in ray_cols}

    return {
        "color": color,
        "direction": direction,
        "seed_count": seed_count,
        "rect_h": rect_h,
        "rect_w": rect_w,
        "border_cells": border_set,
        "gap_cells": set(gap_cells),
        "seed_cells": list(seed_cells),
        "ray_local": ray_local,
        "bbox_cells": bbox_cells,
    }


def _expanded_bbox(r0: int, c0: int, rect_h: int, rect_w: int, h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    cells = set()
    rr0 = max(0, r0 - pad)
    rr1 = min(h - 1, r0 + rect_h - 1 + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(w - 1, c0 + rect_w - 1 + pad)
    for rr in range(rr0, rr1 + 1):
        for cc in range(cc0, cc1 + 1):
            cells.add((rr, cc))
    return cells


def _try_place(
    rng: random.Random,
    grid: np.ndarray,
    candidate: dict,
    blocked_rect: set[tuple[int, int]],
    blocked_ray: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    rect_h = candidate["rect_h"]
    rect_w = candidate["rect_w"]

    for _ in range(PLACEMENT_TRIES):
        r0 = rng.randint(0, h - rect_h)
        c0 = rng.randint(0, w - rect_w)

        bbox_with_pad = _expanded_bbox(r0, c0, rect_h, rect_w, h, w, pad=1)
        if bbox_with_pad & blocked_rect:
            continue
        if bbox_with_pad & blocked_ray:
            continue

        ray_global = {(r0 + rr, c0 + cc) for rr, cc in candidate["ray_local"]}
        if ray_global & blocked_rect:
            continue
        if ray_global & blocked_ray:
            continue

        for rr, cc in candidate["border_cells"]:
            grid[r0 + rr, c0 + cc] = GREY
        for rr, cc in candidate["seed_cells"]:
            grid[r0 + rr, c0 + cc] = candidate["color"]

        blocked_rect.update(bbox_with_pad)
        blocked_ray.update(ray_global)
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_SIZE_RANGE)
    w = rng.randint(*GRID_SIZE_RANGE)
    grid = np.full((h, w), BLACK, dtype=int)

    rect_count = rng.choice(RECT_COUNT_CHOICES)
    colors = rng.sample(SEED_COLORS, rect_count)
    blocked_rect: set[tuple[int, int]] = set()
    blocked_ray: set[tuple[int, int]] = set()

    for color in colors:
        placed = False
        for _ in range(PLACEMENT_TRIES):
            candidate = _make_candidate(rng, h, w, color)
            if candidate is None:
                continue
            if _try_place(rng, grid, candidate, blocked_rect, blocked_ray):
                placed = True
                break
        if not placed:
            return None

    infos = [_analyze_frame(grid, comp) for comp in _components(grid == GREY)]
    if len(infos) != rect_count:
        return None
    if len({info["seed_color"] for info in infos}) != rect_count:
        return None
    if len({info["direction"] for info in infos}) == 1 and rect_count > 1 and rng.random() < 0.5:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
