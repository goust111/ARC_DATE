from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREY = 5
ORANGE = 7

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
REGION_KEYS = ("tl", "tr", "bl", "br")
SCENARIOS = ("plain", "boundary_clip", "divider_clip", "cross_divider")
GEN_ATTEMPTS = 600

BASE_SHAPES = [
    {(0, 0), (0, 1), (1, 0)},
    {(0, 0), (1, 0), (2, 0)},
    {(0, 0), (0, 1), (1, 1)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
    {(0, 0), (0, 1), (1, 1), (2, 1)},
    {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)},
    {(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)},
    {(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)},
]


def _normalize(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _random_shape() -> tuple[tuple[int, int], ...]:
    cells = set(random.choice(BASE_SHAPES))
    if random.random() < 0.5:
        cells = {(c, r) for r, c in cells}
    if random.random() < 0.5:
        mr = max(r for r, _ in cells)
        cells = {(mr - r, c) for r, c in cells}
    if random.random() < 0.5:
        mc = max(c for _, c in cells)
        cells = {(r, mc - c) for r, c in cells}
    return _normalize(cells)


def _connected(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...] | set[tuple[int, int]]) -> bool:
    pts = {tuple(cell) for cell in cells}
    if not pts:
        return False
    seen = set()
    queue = deque([next(iter(pts))])
    while queue:
        r, c = queue.popleft()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in pts and (nr, nc) not in seen:
                queue.append((nr, nc))
    return seen == pts


def _find_dividers(grid: np.ndarray) -> tuple[int, int]:
    sep_rows = np.where(np.all(grid == GREY, axis=1))[0]
    sep_cols = np.where(np.all(grid == GREY, axis=0))[0]
    assert len(sep_rows) == 1 and len(sep_cols) == 1, "expected one full grey row and one full grey column"
    return int(sep_rows[0]), int(sep_cols[0])


def _region_bounds(h: int, w: int, sep_r: int, sep_c: int) -> dict[str, tuple[int, int, int, int]]:
    return {
        "tl": (0, sep_r - 1, 0, sep_c - 1),
        "tr": (0, sep_r - 1, sep_c + 1, w - 1),
        "bl": (sep_r + 1, h - 1, 0, sep_c - 1),
        "br": (sep_r + 1, h - 1, sep_c + 1, w - 1),
    }


def _translate_blue(
    cells: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    direction: str,
    distance: int,
    shape: tuple[int, int],
    sep_r: int,
    sep_c: int,
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    h, w = map(int, shape)
    dr, dc = DIRS[direction]
    kept = []
    out_count = 0
    divider_count = 0
    for r, c in cells:
        nr = int(r + dr * distance)
        nc = int(c + dc * distance)
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            out_count += 1
            continue
        if nr == sep_r or nc == sep_c:
            divider_count += 1
            continue
        kept.append((nr, nc))
    return tuple(sorted(set(kept))), int(out_count), int(divider_count)


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    sep_r, sep_c = _find_dividers(g)
    bounds = _region_bounds(h, w, sep_r, sep_c)

    red_cells = np.argwhere(g == RED)
    assert red_cells.shape[0] == 1, "expected exactly one red marker"
    marker = tuple(map(int, red_cells[0]))
    mr, mc = marker
    assert mr in (0, h - 1) or mc in (0, w - 1), "red marker must be on the boundary"
    assert marker not in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)), "red marker must not be a corner"
    if mr == 0:
        direction = "up"
    elif mr == h - 1:
        direction = "down"
    elif mc == 0:
        direction = "left"
    else:
        direction = "right"

    blue_cells = [tuple(map(int, rc)) for rc in np.argwhere(g == BLUE)]
    assert 3 <= len(blue_cells) <= 5, "blue object must have 3-5 pixels"
    assert _connected(blue_cells), "blue object must be 4-connected"

    counts = {}
    orange_cells = {}
    for key, (r0, r1, c0, c1) in bounds.items():
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        counts[key] = int(np.sum(region == ORANGE))
        local = np.argwhere(region == ORANGE)
        orange_cells[key] = {(int(r0 + rr), int(c0 + cc)) for rr, cc in local}

    count_values = list(counts.values())
    assert len(set(count_values)) == 4, "region orange counts must be distinct"
    assert all(1 <= v <= 5 for v in count_values), "region orange counts must be within 1..5"
    winner = max(counts, key=counts.get)
    distance = int(counts[winner])
    return {
        "shape": (h, w),
        "sep_r": sep_r,
        "sep_c": sep_c,
        "bounds": bounds,
        "marker": marker,
        "direction": direction,
        "blue_cells": tuple(sorted(blue_cells)),
        "counts": counts,
        "orange_cells": orange_cells,
        "winner": winner,
        "distance": distance,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the orange votes in the densest partition, then slide the blue object by that winning vote count.

    Input structure:
    - Black background contains one full grey horizontal divider row and one full grey vertical divider column.
    - The dividers split the grid into four rectangular regions containing scattered orange vote pixels.
    - A single 4-connected blue object lies inside one region, and a single red boundary marker sits on one non-corner edge cell.

    Rule:
    - Detect the grey divider row and column to recover the four regions.
    - Count orange pixels in each region, keep only the unique densest region's orange pixels, and erase orange pixels in the other three regions.
    - Use that maximum orange count as distance D, read the red marker's edge as the slide direction, and translate the blue object by D cells.
    - Drop translated blue pixels that land outside the grid or on the grey divider row/column; keep the grey dividers and red marker unchanged.

    Key insight:
    The orange votes do two jobs at once: they identify the winning region visually and they encode the blue object's slide distance numerically.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_grid(g)
    out = g.copy()
    out[out == ORANGE] = BG
    out[out == BLUE] = BG

    for r, c in info["orange_cells"][info["winner"]]:
        out[r, c] = ORANGE

    moved, _oob, _divider = _translate_blue(
        info["blue_cells"],
        info["direction"],
        info["distance"],
        info["shape"],
        info["sep_r"],
        info["sep_c"],
    )
    for r, c in moved:
        if out[r, c] not in (GREY, RED):
            out[r, c] = BLUE
    return out


def _choose_dimensions() -> tuple[int, int, int, int, int, int]:
    top_h = random.randint(5, 8)
    bottom_h = random.randint(5, 8)
    left_w = random.randint(5, 8)
    right_w = random.randint(5, 8)
    h = top_h + 1 + bottom_h
    w = left_w + 1 + right_w
    return h, w, top_h, bottom_h, left_w, right_w


def _marker_candidates(length: int, blocked_index: int) -> list[int]:
    lo = max(1, length // 2 - 2)
    hi = min(length - 2, length // 2 + 2)
    preferred = [i for i in range(lo, hi + 1) if i != blocked_index]
    if preferred:
        return preferred
    return [i for i in range(1, length - 1) if i != blocked_index]


def _place_marker(h: int, w: int, sep_r: int, sep_c: int, direction: str) -> tuple[int, int]:
    if direction == "up":
        return 0, random.choice(_marker_candidates(w, sep_c))
    if direction == "down":
        return h - 1, random.choice(_marker_candidates(w, sep_c))
    if direction == "left":
        return random.choice(_marker_candidates(h, sep_r)), 0
    return random.choice(_marker_candidates(h, sep_r)), w - 1


def _relevant_side(cells: tuple[tuple[int, int], ...], direction: str, sep_r: int, sep_c: int) -> str:
    if direction in ("left", "right"):
        return "left" if all(c < sep_c for _, c in cells) else "right"
    return "top" if all(r < sep_r for r, _ in cells) else "bottom"


def _crosses_relevant_divider(
    src: tuple[tuple[int, int], ...],
    dst: tuple[tuple[int, int], ...],
    direction: str,
    sep_r: int,
    sep_c: int,
) -> bool:
    return _relevant_side(src, direction, sep_r, sep_c) != _relevant_side(dst, direction, sep_r, sep_c)


def _blue_candidates(
    shape_cells: tuple[tuple[int, int], ...],
    h: int,
    w: int,
    sep_r: int,
    sep_c: int,
    marker: tuple[int, int],
    direction: str,
    distance: int,
    scenario: str,
) -> list[tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], str]]:
    bounds = _region_bounds(h, w, sep_r, sep_c)
    shape_h = 1 + max(r for r, _ in shape_cells)
    shape_w = 1 + max(c for _, c in shape_cells)
    out = []

    for region_key, (r0, r1, c0, c1) in bounds.items():
        if r1 - r0 + 1 < shape_h or c1 - c0 + 1 < shape_w:
            continue
        for base_r in range(r0, r1 - shape_h + 2):
            for base_c in range(c0, c1 - shape_w + 2):
                src = tuple(sorted((base_r + dr, base_c + dc) for dr, dc in shape_cells))
                moved, oob, divider = _translate_blue(src, direction, distance, (h, w), sep_r, sep_c)
                if len(moved) < 2 or not _connected(moved):
                    continue
                if marker in moved:
                    continue
                if moved == src:
                    continue
                crossed = _crosses_relevant_divider(src, moved, direction, sep_r, sep_c)
                ok = False
                if scenario == "plain":
                    ok = oob == 0 and divider == 0 and not crossed
                elif scenario == "boundary_clip":
                    ok = oob > 0 and divider == 0
                elif scenario == "divider_clip":
                    ok = divider > 0 and oob == 0
                elif scenario == "cross_divider":
                    ok = oob == 0 and divider == 0 and crossed
                if ok:
                    out.append((src, moved, region_key))
    return out


def _orange_points(
    bbox: tuple[int, int, int, int],
    count: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    r0, r1, c0, c1 = bbox
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) not in blocked]
    if len(cells) < count:
        return None
    for radius in (1, 0):
        for _ in range(200):
            picks = random.sample(cells, count)
            good = True
            for i, (r1_, c1_) in enumerate(picks):
                for r2_, c2_ in picks[i + 1 :]:
                    if max(abs(r1_ - r2_), abs(c1_ - c2_)) <= radius:
                        good = False
                        break
                if not good:
                    break
            if good:
                return picks
    return None


def _halo(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _counts_assignment() -> dict[str, int]:
    if random.random() < 0.5:
        values = [1, 2, 3, 4]
    else:
        missing = random.choice([1, 2, 3, 4])
        values = [v for v in [1, 2, 3, 4, 5] if v != missing]
    random.shuffle(values)
    return {key: values[i] for i, key in enumerate(REGION_KEYS)}


def _blue_uniqueness_ok(info: dict) -> bool:
    actual = set(
        _translate_blue(
            info["blue_cells"],
            info["direction"],
            info["distance"],
            info["shape"],
            info["sep_r"],
            info["sep_c"],
        )[0]
    )
    if len(actual) < 2:
        return False
    for alt_dir in DIRS:
        for alt_d in info["counts"].values():
            if alt_dir == info["direction"] and alt_d == info["distance"]:
                continue
            alt = set(
                _translate_blue(
                    info["blue_cells"],
                    alt_dir,
                    alt_d,
                    info["shape"],
                    info["sep_r"],
                    info["sep_c"],
                )[0]
            )
            if alt == actual:
                return False
    return True


def _instance_checks(inp: np.ndarray) -> bool:
    try:
        info = _analyze_grid(inp)
        out = main(inp)
    except Exception:
        return False

    if not np.array_equal(out[info["sep_r"], :], inp[info["sep_r"], :]):
        return False
    if not np.array_equal(out[:, info["sep_c"]], inp[:, info["sep_c"]]):
        return False
    if out[info["marker"]] != RED:
        return False
    if not _blue_uniqueness_ok(info):
        return False

    actual_winner_orange = {(int(r), int(c)) for r, c in np.argwhere(out == ORANGE)}
    if actual_winner_orange != info["orange_cells"][info["winner"]]:
        return False

    for wrong in REGION_KEYS:
        if wrong == info["winner"]:
            continue
        if info["orange_cells"][wrong] == actual_winner_orange:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h, w, top_h, _bottom_h, left_w, _right_w = _choose_dimensions()
    sep_r = top_h
    sep_c = left_w
    direction = random.choice(tuple(DIRS))
    scenario = random.choice(SCENARIOS)
    counts = _counts_assignment()
    distance = max(counts.values())
    marker = _place_marker(h, w, sep_r, sep_c, direction)
    shape_cells = _random_shape()

    candidates = _blue_candidates(shape_cells, h, w, sep_r, sep_c, marker, direction, distance, scenario)
    if not candidates:
        return None
    blue_cells, moved_cells, _blue_region = random.choice(candidates)
    winner = max(counts, key=counts.get)
    bounds = _region_bounds(h, w, sep_r, sep_c)

    grid = np.full((h, w), BG, dtype=int)
    grid[sep_r, :] = GREY
    grid[:, sep_c] = GREY
    grid[marker] = RED
    for r, c in blue_cells:
        grid[r, c] = BLUE

    for key in REGION_KEYS:
        blocked = set(blue_cells)
        if key == winner:
            blocked |= set(moved_cells)
        blocked = _halo(blocked, h, w)
        points = _orange_points(bounds[key], counts[key], blocked)
        if points is None:
            return None
        for r, c in points:
            if grid[r, c] != BG:
                return None
            grid[r, c] = ORANGE

    info = _analyze_grid(grid)
    if info["distance"] != distance or info["direction"] != direction or info["winner"] != winner:
        return None
    if set(info["blue_cells"]) != set(blue_cells):
        return None
    pred = main(grid)

    expected = grid.copy()
    expected[expected == ORANGE] = BG
    expected[expected == BLUE] = BG
    for r, c in info["orange_cells"][winner]:
        expected[r, c] = ORANGE
    for r, c in moved_cells:
        expected[r, c] = BLUE
    if not np.array_equal(pred, expected):
        return None
    if not _instance_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
