import random
from collections import deque

import numpy as np


BG = 0
GEN_WALL = 5
PALETTE = tuple(c for c in range(10) if c not in (BG, GEN_WALL))
SIDES = ("top", "bottom", "left", "right")

GRID_H_RANGE = (14, 30)
GRID_W_RANGE = (14, 30)
MAX_REGIONS = 5
MIN_REGION_PAD = 1
MAX_GEN_ATTEMPTS = 300
MAX_REGION_PLACEMENT_ATTEMPTS = 80


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _most_common_non_bg(grid: np.ndarray, bg: int = BG) -> int:
    vals = grid[grid != bg]
    if vals.size == 0:
        return bg
    colors, counts = np.unique(vals, return_counts=True)
    return int(colors[np.argmax(counts)])


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        if r1 != r0:
            pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        if c1 != c0:
            pts.append((r, c1))
    return pts


def _stamp_patch(
    out: np.ndarray,
    patch: np.ndarray,
    color: int,
    top: int,
    left: int,
    bounds: tuple[int, int, int, int],
) -> None:
    ir0, ic0, ir1, ic1 = bounds
    rows, cols = np.where(patch)
    for pr, pc in zip(rows.tolist(), cols.tolist()):
        rr = top + pr
        cc = left + pc
        if ir0 <= rr <= ir1 and ic0 <= cc <= ic1:
            out[rr, cc] = color


def _apply_region_tiling(
    out: np.ndarray,
    side: str,
    color: int,
    patch: np.ndarray,
    seed_top: int,
    seed_left: int,
    bounds: tuple[int, int, int, int],
) -> None:
    ir0, ic0, ir1, ic1 = bounds
    out[ir0 : ir1 + 1, ic0 : ic1 + 1] = BG
    ph, pw = patch.shape

    if side == "top":
        top = seed_top
        while top <= ir1:
            _stamp_patch(out, patch, color, top, seed_left, bounds)
            top += ph
    elif side == "bottom":
        top = seed_top
        while top + ph - 1 >= ir0:
            _stamp_patch(out, patch, color, top, seed_left, bounds)
            top -= ph
    elif side == "left":
        left = seed_left
        while left <= ic1:
            _stamp_patch(out, patch, color, seed_top, left, bounds)
            left += pw
    else:
        left = seed_left
        while left + pw - 1 >= ic0:
            _stamp_patch(out, patch, color, seed_top, left, bounds)
            left -= pw


def _extract_region(grid: np.ndarray, comp: list[tuple[int, int]], wall: int):
    r0, c0, r1, c1 = _bbox(comp)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    wall_set = set(comp)
    if not all(r in (r0, r1) or c in (c0, c1) for r, c in wall_set):
        return None

    perimeter = _perimeter_coords(r0, c0, r1, c1)
    gaps = [(r, c) for r, c in perimeter if (r, c) not in wall_set]
    if len(gaps) != 1:
        return None

    gap_r, gap_c = gaps[0]
    if (gap_r in (r0, r1)) and (gap_c in (c0, c1)):
        return None
    if gap_r == r0:
        side = "top"
    elif gap_r == r1:
        side = "bottom"
    elif gap_c == c0:
        side = "left"
    else:
        side = "right"

    ir0, ic0, ir1, ic1 = r0 + 1, c0 + 1, r1 - 1, c1 - 1
    if ir0 > ir1 or ic0 > ic1:
        return None

    interior = grid[ir0 : ir1 + 1, ic0 : ic1 + 1]
    seed_mask = (interior != BG) & (interior != wall)
    if not np.any(seed_mask):
        return None
    colors = np.unique(interior[seed_mask])
    if colors.size != 1:
        return None
    color = int(colors[0])

    local_coords = np.argwhere(seed_mask)
    sr0, sc0 = local_coords.min(axis=0).tolist()
    sr1, sc1 = local_coords.max(axis=0).tolist()
    patch = seed_mask[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
    seed_top = ir0 + int(sr0)
    seed_left = ic0 + int(sc0)

    return {
        "side": side,
        "color": color,
        "patch": patch,
        "seed_top": seed_top,
        "seed_left": seed_left,
        "bounds": (ir0, ic0, ir1, ic1),
    }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    wall = _most_common_non_bg(g, BG)
    if wall == BG:
        return g.copy()

    out = g.copy()
    for comp in _components4(g == wall):
        region = _extract_region(g, comp, wall)
        if region is None:
            continue
        _apply_region_tiling(
            out,
            region["side"],
            region["color"],
            region["patch"],
            region["seed_top"],
            region["seed_left"],
            region["bounds"],
        )
    return out


def _random_segment(length: int, rng: random.Random) -> tuple[int, int]:
    seg_len = rng.randint(max(1, length // 3), length)
    start = rng.randint(0, length - seg_len)
    return start, start + seg_len


def _vertical_patch(height: int, width: int, side: str, rng: random.Random) -> np.ndarray:
    patch = np.zeros((height, width), dtype=bool)
    if height == 1:
        patch[0, :] = True
        return patch
    if height == 2:
        backbone = 1 if side == "top" else 0
        patch[backbone, :] = True
        row = 0 if side == "top" else 1
        start, end = _random_segment(width, rng)
        patch[row, start:end] = True
        return patch

    patch[1, :] = True
    start, end = _random_segment(width, rng)
    patch[0, start:end] = True
    start, end = _random_segment(width, rng)
    patch[2, start:end] = True
    near = 0 if side == "top" else 2
    if not np.any(patch[near]):
        start, end = _random_segment(width, rng)
        patch[near, start:end] = True
    return patch


def _horizontal_patch(height: int, width: int, side: str, rng: random.Random) -> np.ndarray:
    patch = np.zeros((height, width), dtype=bool)
    if width == 1:
        patch[:, 0] = True
        return patch
    if width == 2:
        backbone = 1 if side == "left" else 0
        patch[:, backbone] = True
        col = 0 if side == "left" else 1
        start, end = _random_segment(height, rng)
        patch[start:end, col] = True
        return patch

    patch[:, 1] = True
    start, end = _random_segment(height, rng)
    patch[start:end, 0] = True
    start, end = _random_segment(height, rng)
    patch[start:end, 2] = True
    near = 0 if side == "left" else 2
    if not np.any(patch[:, near]):
        start, end = _random_segment(height, rng)
        patch[start:end, near] = True
    return patch


def _sample_region(color: int, rng: random.Random) -> dict[str, object]:
    side = rng.choice(SIDES)
    if side in ("top", "bottom"):
        interior_h = rng.randint(4, 9)
        interior_w = rng.randint(3, 9)
        seed_h = rng.randint(1, min(3, interior_h - 1))
        patch = _vertical_patch(seed_h, interior_w, side, rng)
        gap_axis = [c for c in range(interior_w) if patch[0 if side == "top" else -1, c]]
        seed_row = 0 if side == "top" else interior_h - seed_h
        seed_col = 0
    else:
        interior_h = rng.randint(3, 9)
        interior_w = rng.randint(4, 9)
        seed_w = rng.randint(1, min(3, interior_w - 1))
        patch = _horizontal_patch(interior_h, seed_w, side, rng)
        gap_axis = [r for r in range(interior_h) if patch[r, 0 if side == "left" else -1]]
        seed_row = 0
        seed_col = 0 if side == "left" else interior_w - seed_w

    return {
        "side": side,
        "color": color,
        "interior_h": interior_h,
        "interior_w": interior_w,
        "patch": patch,
        "gap_offset": int(rng.choice(gap_axis)),
        "seed_row": seed_row,
        "seed_col": seed_col,
    }


def _candidate_positions(
    height: int,
    width: int,
    outer_h: int,
    outer_w: int,
    occupied: np.ndarray,
) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for r0 in range(MIN_REGION_PAD, height - outer_h - MIN_REGION_PAD + 1):
        for c0 in range(MIN_REGION_PAD, width - outer_w - MIN_REGION_PAD + 1):
            rr0 = max(0, r0 - MIN_REGION_PAD)
            rr1 = min(height, r0 + outer_h + MIN_REGION_PAD)
            cc0 = max(0, c0 - MIN_REGION_PAD)
            cc1 = min(width, c0 + outer_w + MIN_REGION_PAD)
            if not occupied[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    return positions


def _draw_input_region(grid: np.ndarray, spec: dict[str, object]) -> None:
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    interior_h = int(spec["interior_h"])
    interior_w = int(spec["interior_w"])
    color = int(spec["color"])
    patch = np.asarray(spec["patch"], dtype=bool)
    side = str(spec["side"])
    gap_offset = int(spec["gap_offset"])
    seed_row = int(spec["seed_row"])
    seed_col = int(spec["seed_col"])

    r1 = r0 + interior_h + 1
    c1 = c0 + interior_w + 1
    grid[r0, c0 : c1 + 1] = GEN_WALL
    grid[r1, c0 : c1 + 1] = GEN_WALL
    grid[r0 : r1 + 1, c0] = GEN_WALL
    grid[r0 : r1 + 1, c1] = GEN_WALL

    if side == "top":
        grid[r0, c0 + 1 + gap_offset] = BG
    elif side == "bottom":
        grid[r1, c0 + 1 + gap_offset] = BG
    elif side == "left":
        grid[r0 + 1 + gap_offset, c0] = BG
    else:
        grid[r0 + 1 + gap_offset, c1] = BG

    seed_top = r0 + 1 + seed_row
    seed_left = c0 + 1 + seed_col
    grid[seed_top : seed_top + patch.shape[0], seed_left : seed_left + patch.shape[1]][patch] = color


def _draw_output_region(grid: np.ndarray, spec: dict[str, object]) -> None:
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    interior_h = int(spec["interior_h"])
    interior_w = int(spec["interior_w"])
    color = int(spec["color"])
    patch = np.asarray(spec["patch"], dtype=bool)
    side = str(spec["side"])
    seed_row = int(spec["seed_row"])
    seed_col = int(spec["seed_col"])

    r1 = r0 + interior_h + 1
    c1 = c0 + interior_w + 1
    grid[r0, c0 : c1 + 1] = GEN_WALL
    grid[r1, c0 : c1 + 1] = GEN_WALL
    grid[r0 : r1 + 1, c0] = GEN_WALL
    grid[r0 : r1 + 1, c1] = GEN_WALL

    gap_offset = int(spec["gap_offset"])
    if side == "top":
        grid[r0, c0 + 1 + gap_offset] = BG
    elif side == "bottom":
        grid[r1, c0 + 1 + gap_offset] = BG
    elif side == "left":
        grid[r0 + 1 + gap_offset, c0] = BG
    else:
        grid[r0 + 1 + gap_offset, c1] = BG

    seed_top = r0 + 1 + seed_row
    seed_left = c0 + 1 + seed_col
    _apply_region_tiling(
        grid,
        side,
        color,
        patch,
        seed_top,
        seed_left,
        (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
    )


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_H_RANGE)
    width = rng.randint(*GRID_W_RANGE)
    grid = np.full((height, width), BG, dtype=int)
    expected = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    max_regions = min(MAX_REGIONS, max(1, (height * width) // 80))
    target_regions = rng.randint(1, max_regions)
    colors = rng.sample(PALETTE, target_regions)

    regions: list[dict[str, object]] = []
    for color in colors:
        placed = False
        for _ in range(MAX_REGION_PLACEMENT_ATTEMPTS):
            spec = _sample_region(color, rng)
            outer_h = int(spec["interior_h"]) + 2
            outer_w = int(spec["interior_w"]) + 2
            positions = _candidate_positions(height, width, outer_h, outer_w, occupied)
            if not positions:
                return None
            r0, c0 = rng.choice(positions)
            spec["r0"] = r0
            spec["c0"] = c0
            rr0 = max(0, r0 - MIN_REGION_PAD)
            rr1 = min(height, r0 + outer_h + MIN_REGION_PAD)
            cc0 = max(0, c0 - MIN_REGION_PAD)
            cc1 = min(width, c0 + outer_w + MIN_REGION_PAD)
            occupied[rr0:rr1, cc0:cc1] = True
            regions.append(spec)
            placed = True
            break
        if not placed:
            return None

    if not regions:
        return None

    for spec in regions:
        _draw_input_region(grid, spec)
        _draw_output_region(expected, spec)

    if np.array_equal(grid, expected):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
