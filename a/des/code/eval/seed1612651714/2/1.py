from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_ATTEMPTS = 400
LEGEND_ATTEMPTS = 40
REGION_ATTEMPTS = 60
PLACEMENT_ATTEMPTS = 120
MAX_GRID_SIZE = 30


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(pts)
    return out


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_edge(points: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in points)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _parse_legend(legend: np.ndarray) -> dict[int, np.ndarray] | None:
    comps = _components(legend != BLACK)
    if len(comps) < 2:
        return None

    mapping: dict[int, np.ndarray] = {}
    for pts in comps:
        r0, c0, r1, c1 = _bbox(pts)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != w or h not in (2, 3):
            return None
        if len(pts) != h * w:
            return None
        sample = legend[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(sample == BLACK):
            return None
        if np.unique(sample).size < 2:
            return None
        key = int(sample[0, 0])
        if key in mapping:
            return None
        mapping[key] = sample.copy()

    return mapping


def _find_divider_and_legend(grid: np.ndarray) -> tuple[int, int, dict[int, np.ndarray]]:
    h = grid.shape[0]
    for r in range(h - 2, 0, -1):
        if np.all(grid[r, :] == grid[r, 0]) and int(grid[r, 0]) != BLACK:
            grey = int(grid[r, 0])
            mapping = _parse_legend(grid[r + 1 :, :])
            if mapping is not None:
                return r, grey, mapping
    raise AssertionError("expected one full-width divider row above a parseable legend")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region by tiling the legend pattern keyed by its seed color.

    Input structure:
    - Background is black.
    - The upper area contains grey-walled enclosed regions; each region interior
      has exactly one colored seed pixel.
    - A full grey divider row separates the upper area from the bottom legend.
    - The legend contains several 2x2 or 3x3 non-black sample patterns. The
      top-left color of each sample is the lookup key for that pattern.

    Rule:
    - For each enclosed region, read its seed color.
    - Find the legend sample whose top-left pixel has that color.
    - Tile that sample over the region interior, anchored at the region
      interior bbox's top-left corner. The seed pixel is overwritten too.
    - Keep grey walls and the legend unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    divider, grey, legend = _find_divider_and_legend(g)
    top = g[:divider, :]
    out = g.copy()

    found_region = False
    for pts in _components(top != grey):
        if _touches_edge(pts, *top.shape):
            continue
        seed_cells = [(r, c) for r, c in pts if int(top[r, c]) != BLACK]
        assert len(seed_cells) == 1, "each enclosed region must contain exactly one seed pixel"

        found_region = True
        seed_r, seed_c = seed_cells[0]
        seed_color = int(top[seed_r, seed_c])
        assert seed_color in legend, "seed color must map to a legend pattern"

        pattern = legend[seed_color]
        ph, pw = pattern.shape
        r0, c0, _, _ = _bbox(pts)
        for r, c in pts:
            out[r, c] = int(pattern[(r - r0) % ph, (c - c0) % pw])

    assert found_region, "expected at least one enclosed region"
    return out


def _make_pattern(size: int, key: int, accent_pool: list[int], rng: random.Random) -> np.ndarray:
    arr = np.empty((size, size), dtype=int)
    color_count = rng.randint(2, min(4, 1 + len(accent_pool)))
    palette = [key] + rng.sample(accent_pool, color_count - 1)
    for r in range(size):
        for c in range(size):
            arr[r, c] = rng.choice(palette)
    arr[0, 0] = key
    if np.unique(arr).size < 2:
        rr, cc = size - 1, size - 1
        arr[rr, cc] = palette[1]
    return arr


def _place_row(
    legend: np.ndarray,
    patterns: list[np.ndarray],
    y0: int,
    band_h: int,
    rng: random.Random,
) -> bool:
    widths = [int(p.shape[1]) for p in patterns]
    gaps = [rng.randint(1, 3) for _ in range(len(patterns) - 1)]
    used = sum(widths) + sum(gaps)
    if used > legend.shape[1] - 2:
        return False

    max_left = legend.shape[1] - used - 1
    x = rng.randint(1, max_left)
    for i, pattern in enumerate(patterns):
        h, w = pattern.shape
        dy = 0 if band_h == h else rng.randint(0, band_h - h)
        legend[y0 + dy : y0 + dy + h, x : x + w] = pattern
        x += w
        if i < len(gaps):
            x += gaps[i]
    return True


def _render_legend(patterns: list[np.ndarray], width: int, rng: random.Random) -> np.ndarray | None:
    modes = [1, 2]
    rng.shuffle(modes)
    for mode in modes:
        for _ in range(LEGEND_ATTEMPTS):
            order = patterns[:]
            rng.shuffle(order)

            if mode == 1:
                band_h = max(int(p.shape[0]) for p in order)
                top_pad = rng.randint(0, 1)
                bottom_pad = rng.randint(0, 1)
                legend = np.zeros((top_pad + band_h + bottom_pad, width), dtype=int)
                if not _place_row(legend, order, top_pad, band_h, rng):
                    continue
            else:
                split = rng.randint(1, len(order) - 1)
                row1 = order[:split]
                row2 = order[split:]
                band1 = max(int(p.shape[0]) for p in row1)
                band2 = max(int(p.shape[0]) for p in row2)
                row_gap = rng.randint(1, 2)
                top_pad = rng.randint(0, 1)
                bottom_pad = rng.randint(0, 1)
                legend = np.zeros((top_pad + band1 + row_gap + band2 + bottom_pad, width), dtype=int)
                if not _place_row(legend, row1, top_pad, band1, rng):
                    continue
                if not _place_row(legend, row2, top_pad + band1 + row_gap, band2, rng):
                    continue

            if len(_components(legend != BLACK)) == len(patterns):
                return legend
    return None


def _rectangle_mask(min_side: int, rng: random.Random) -> np.ndarray:
    h = rng.randint(min_side, min(min_side + 3, 8))
    w = rng.randint(min_side, min(min_side + 3, 8))
    return np.ones((h, w), dtype=bool)


def _notch_mask(min_side: int, rng: random.Random) -> np.ndarray:
    h = rng.randint(max(5, min_side), min(max(5, min_side) + 3, 8))
    w = rng.randint(max(5, min_side), min(max(5, min_side) + 3, 8))
    mask = np.ones((h, w), dtype=bool)
    notch_h = rng.randint(1, h - 2)
    notch_w = rng.randint(1, w - 2)
    corner = rng.choice(("tl", "tr", "bl", "br"))
    if corner == "tl":
        mask[:notch_h, :notch_w] = False
    elif corner == "tr":
        mask[:notch_h, w - notch_w :] = False
    elif corner == "bl":
        mask[h - notch_h :, :notch_w] = False
    else:
        mask[h - notch_h :, w - notch_w :] = False
    return mask


def _cross_mask(min_side: int, rng: random.Random) -> np.ndarray:
    h = rng.randint(max(5, min_side), min(max(5, min_side) + 4, 9))
    w = rng.randint(max(5, min_side), min(max(5, min_side) + 4, 9))
    mask = np.zeros((h, w), dtype=bool)
    vert_w = rng.randint(1, max(1, w // 2))
    horiz_h = rng.randint(1, max(1, h // 2))
    vc = rng.randint(0, w - vert_w)
    hr = rng.randint(0, h - horiz_h)
    mask[:, vc : vc + vert_w] = True
    mask[hr : hr + horiz_h, :] = True
    return mask


def _make_region_mask(pattern_size: int, rng: random.Random) -> np.ndarray | None:
    min_side = pattern_size + 1
    for _ in range(REGION_ATTEMPTS):
        kind = rng.choice(("rect", "notch", "cross"))
        if kind == "rect":
            mask = _rectangle_mask(min_side, rng)
        elif kind == "notch":
            mask = _notch_mask(min_side, rng)
        else:
            mask = _cross_mask(min_side, rng)

        if int(mask.sum()) < 6:
            continue
        if mask.shape[0] <= pattern_size and mask.shape[1] <= pattern_size:
            continue
        return mask
    return None


def _place_regions(
    top_h: int,
    width: int,
    assignments: list[int],
    patterns: dict[int, np.ndarray],
    rng: random.Random,
) -> tuple[np.ndarray, list[dict]] | None:
    reserve_boxes: list[tuple[int, int, int, int]] = []
    records: list[dict] = []

    for key in assignments:
        pattern = patterns[key]
        for _ in range(PLACEMENT_ATTEMPTS):
            mask = _make_region_mask(int(pattern.shape[0]), rng)
            if mask is None:
                return None

            h, w = mask.shape
            max_r = top_h - h - 2
            max_c = width - w - 2
            if max_r < 2 or max_c < 2:
                return None

            base_r = rng.randint(2, max_r)
            base_c = rng.randint(2, max_c)
            reserve = (base_r - 2, base_c - 2, base_r + h + 1, base_c + w + 1)
            if any(_boxes_overlap(reserve, other) for other in reserve_boxes):
                continue

            local_pts = [tuple(map(int, pt)) for pt in np.argwhere(mask)]
            local_set = set(local_pts)
            cells = [(base_r + r, base_c + c) for r, c in local_pts]
            boundary: set[tuple[int, int]] = set()
            valid = True
            for r, c in local_pts:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in local_set:
                        continue
                    gr, gc = base_r + nr, base_c + nc
                    if not (0 <= gr < top_h and 0 <= gc < width):
                        valid = False
                        break
                    boundary.add((gr, gc))
                if not valid:
                    break
            if not valid:
                continue

            seed = rng.choice(cells)
            records.append(
                {
                    "key": key,
                    "pattern": pattern,
                    "cells": cells,
                    "boundary": boundary,
                    "seed": seed,
                }
            )
            reserve_boxes.append(reserve)
            break
        else:
            return None

    top = np.zeros((top_h, width), dtype=int)
    for record in records:
        for r, c in record["boundary"]:
            top[r, c] = GREY
    for record in records:
        sr, sc = record["seed"]
        top[sr, sc] = int(record["key"])
    return top, records


def _expected_output(inp: np.ndarray, records: list[dict]) -> np.ndarray:
    out = inp.copy()
    for record in records:
        cells = record["cells"]
        pattern = record["pattern"]
        ph, pw = pattern.shape
        r0, c0, _, _ = _bbox(cells)
        for r, c in cells:
            out[r, c] = int(pattern[(r - r0) % ph, (c - c0) % pw])
    return out


def _build_candidate(rng: random.Random) -> np.ndarray | None:
    sample_count = rng.randint(2, 4)
    keys = rng.sample(COLOR_POOL, sample_count)
    sizes = [2, 3]
    while len(sizes) < sample_count:
        sizes.append(rng.choice((2, 3)))
    rng.shuffle(sizes)

    accent_pool = [color for color in COLOR_POOL if color not in keys]
    patterns = {
        key: _make_pattern(size, key, accent_pool, rng)
        for key, size in zip(keys, sizes)
    }

    region_count = sample_count + (1 if sample_count < 4 and rng.random() < 0.5 else 0)
    assignments = list(keys)
    while len(assignments) < region_count:
        assignments.append(rng.choice(keys))
    rng.shuffle(assignments)

    width_min = max(18, 5 * region_count + 4)
    width = rng.randint(width_min, MAX_GRID_SIZE)
    legend = _render_legend([patterns[key] for key in keys], width, rng)
    if legend is None:
        return None

    max_top_h = MAX_GRID_SIZE - 1 - legend.shape[0]
    min_top_h = max(12, 4 * region_count + 2)
    max_top_h = min(max_top_h, 21)
    if min_top_h > max_top_h:
        return None
    top_h = rng.randint(min_top_h, max_top_h)

    placed = _place_regions(top_h, width, assignments, patterns, rng)
    if placed is None:
        return None
    top, records = placed

    grid = np.zeros((top_h + 1 + legend.shape[0], width), dtype=int)
    grid[:top_h, :] = top
    grid[top_h, :] = GREY
    grid[top_h + 1 :, :] = legend

    expected = _expected_output(grid, records)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        candidate = _build_candidate(rng)
        if candidate is not None:
            return candidate
    raise AssertionError("failed to generate a valid ARC task instance")
