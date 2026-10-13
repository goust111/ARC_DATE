from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIVIDER_WIDTH = 1
BACKGROUND_COLOR = 0
DIVIDER_COLOR = 5
PAIR_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GRID_OPTIONS = [(2, 2), (2, 3), (2, 4), (3, 2)]
REGION_SIZE_RANGE = (5, 8)
GENERATION_RETRIES = 400


def _split_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(mask):
        if not bool(mask[i]):
            i += 1
            continue
        j = i
        while j + 1 < len(mask) and bool(mask[j + 1]):
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def _partition_info(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(set(map(int, g.flat)))

    for divider in colors:
        row_non_divider = np.any(g != divider, axis=1)
        col_non_divider = np.any(g != divider, axis=0)
        row_runs = _split_runs(row_non_divider)
        col_runs = _split_runs(col_non_divider)

        if not row_runs or not col_runs:
            continue
        if len(row_runs) * len(col_runs) < 2 or (len(row_runs) * len(col_runs)) % 2 != 0:
            continue

        row_sizes = {r1 - r0 + 1 for r0, r1 in row_runs}
        col_sizes = {c1 - c0 + 1 for c0, c1 in col_runs}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue

        ok = True
        for a, b in zip(row_runs, row_runs[1:]):
            if b[0] - a[1] - 1 != DIVIDER_WIDTH:
                ok = False
                break
        for a, b in zip(col_runs, col_runs[1:]):
            if b[0] - a[1] - 1 != DIVIDER_WIDTH:
                ok = False
                break
        if not ok:
            continue

        for r0, r1 in row_runs:
            for c0, c1 in col_runs:
                region = g[r0 : r1 + 1, c0 : c1 + 1]
                if np.any(region == divider):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return int(divider), row_runs, col_runs

    raise AssertionError("failed to detect divider partition")


def _region_background(grid: np.ndarray, divider: int) -> int:
    interior = grid[grid != divider]
    return int(Counter(map(int, interior.flat)).most_common(1)[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each same-colored region pair by merging one fragment with the 180-degree rotated coordinates of its partner.

    Input structure:
    - A single-color divider grid splits the canvas into equally sized rectangular regions.
    - Every non-divider foreground color appears in exactly two regions, and each such region contains a fragment of one shared pattern on a common background.
    - The two partner fragments live in opposite local coordinate systems: one region stores its piece directly, the other stores it after a 180-degree coordinate rotation.

    Rule:
    - Detect the divider partition and group regions by their unique foreground color.
    - For each pair, read boolean masks A and B in local region coordinates and compute M = A union rot180(B).
    - Rewrite the first region with M, rewrite the second with rot180(M), keep the background elsewhere, and preserve the divider lines.

    Key insight:
    - Missing cells are supplied by the same-colored partner region, but only after undoing that partner's hidden 180-degree coordinate rotation.
    """
    g = np.asarray(grid, dtype=int)
    divider, row_runs, col_runs = _partition_info(g)
    bg = _region_background(g, divider)

    regions_by_color: dict[int, list[tuple[tuple[int, int], tuple[int, int], np.ndarray]]] = {}
    for rr in row_runs:
        for cc in col_runs:
            r0, r1 = rr
            c0, c1 = cc
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            present = sorted(set(map(int, region.flat)) - {bg})
            assert divider not in present
            assert len(present) == 1, "each region must have exactly one foreground color"
            color = int(present[0])
            regions_by_color.setdefault(color, []).append((rr, cc, region))

    out = np.full_like(g, bg)
    out[g == divider] = divider

    for color, items in regions_by_color.items():
        assert len(items) == 2, "each foreground color must appear in exactly two regions"
        (_, _, region_a), (_, _, region_b) = items
        mask_a = region_a == color
        mask_b = region_b == color
        merged = mask_a | np.rot90(mask_b, 2)
        merged_b = np.rot90(merged, 2)

        for (rr, cc, _region), mask in ((items[0], merged), (items[1], merged_b)):
            r0, r1 = rr
            c0, c1 = cc
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            patch[mask] = color
            out[r0 : r1 + 1, c0 : c1 + 1] = patch

    return out


def _neighbors(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _grow_connected_mask(h: int, w: int, total: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (rng.randrange(h), rng.randrange(w))
    mask[start] = True
    frontier = set(_neighbors(h, w, *start))

    while int(mask.sum()) < total and frontier:
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        for nxt in _neighbors(h, w, *cell):
            if not mask[nxt]:
                frontier.add(nxt)
    return mask


def _grow_subset(base: np.ndarray, seeds: np.ndarray, target: int, rng: random.Random) -> np.ndarray | None:
    h, w = base.shape
    subset = seeds.copy()
    frontier: set[tuple[int, int]] = set()
    seed_coords = np.argwhere(subset)
    for r, c in seed_coords:
        for nr, nc in _neighbors(h, w, int(r), int(c)):
            if base[nr, nc] and not subset[nr, nc]:
                frontier.add((nr, nc))

    while int(subset.sum()) < target:
        if not frontier:
            return None
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        subset[cell] = True
        for nxt in _neighbors(h, w, *cell):
            if base[nxt] and not subset[nxt]:
                frontier.add(nxt)
    return subset


def _covers_all_rows_cols(mask: np.ndarray) -> bool:
    return bool(mask.any(axis=0).all() and mask.any(axis=1).all())


def _make_fragments(h: int, w: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    area = h * w
    min_cells = max(h + w - 1, area // 3)
    max_cells = min(area - 2, (2 * area) // 3 + 1)
    if min_cells > max_cells:
        return None

    for _ in range(200):
        total = rng.randint(min_cells, max_cells)
        full = _grow_connected_mask(h, w, total, rng)
        if not _covers_all_rows_cols(full):
            continue

        coords = [tuple(map(int, rc)) for rc in np.argwhere(full)]
        if len(coords) < 6:
            continue

        overlap_size = rng.randint(1, min(3, len(coords) - 4))
        overlap = np.zeros((h, w), dtype=bool)
        overlap_seed = rng.choice(coords)
        overlap[overlap_seed] = True
        overlap = _grow_subset(full, overlap, overlap_size, rng)
        if overlap is None:
            continue

        low_target = max(int(overlap.sum()) + 2, len(coords) // 2)
        high_target = len(coords) - 1
        if low_target > high_target:
            continue
        a_target = rng.randint(low_target, high_target)
        b_target = rng.randint(low_target, high_target)

        mask_a = _grow_subset(full, overlap, a_target, rng)
        mask_v = _grow_subset(full, overlap, b_target, rng)
        if mask_a is None or mask_v is None:
            continue

        while True:
            uncovered = full & ~(mask_a | mask_v)
            if not np.any(uncovered):
                break
            candidates = []
            for r, c in np.argwhere(uncovered):
                options = []
                rr = int(r)
                cc = int(c)
                if any(mask_a[nr, nc] for nr, nc in _neighbors(h, w, rr, cc)):
                    options.append("a")
                if any(mask_v[nr, nc] for nr, nc in _neighbors(h, w, rr, cc)):
                    options.append("v")
                if options:
                    candidates.append((rr, cc, options))
            if not candidates:
                break
            rr, cc, options = rng.choice(candidates)
            if rng.choice(options) == "a":
                mask_a[rr, cc] = True
            else:
                mask_v[rr, cc] = True

        if not np.array_equal(mask_a | mask_v, full):
            continue
        if not np.any(mask_a & mask_v):
            continue
        if int(mask_a.sum()) >= int(full.sum()) or int(mask_v.sum()) >= int(full.sum()):
            continue
        if not (_covers_all_rows_cols(mask_a) and _covers_all_rows_cols(mask_v)):
            continue

        stored_b = np.rot90(mask_v, 2)
        if np.array_equal(mask_a | stored_b, full):
            continue

        return full, mask_a, stored_b

    return None


def _pair_regions(rows: int, cols: int, colors: list[int], rng: random.Random) -> dict[tuple[int, int], int]:
    positions = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(positions)
    mapping: dict[tuple[int, int], int] = {}
    for color, (p0, p1) in zip(colors, zip(positions[::2], positions[1::2])):
        mapping[p0] = color
        mapping[p1] = color
    return mapping


def _build_case(rng: random.Random) -> np.ndarray | None:
    region_rows, region_cols = rng.choice(GRID_OPTIONS)
    pair_count = (region_rows * region_cols) // 2
    colors = rng.sample(PAIR_COLORS, pair_count)

    max_region_h = min(REGION_SIZE_RANGE[1], (30 - (region_rows - 1) * DIVIDER_WIDTH) // region_rows)
    max_region_w = min(REGION_SIZE_RANGE[1], (30 - (region_cols - 1) * DIVIDER_WIDTH) // region_cols)
    if max_region_h < REGION_SIZE_RANGE[0] or max_region_w < REGION_SIZE_RANGE[0]:
        return None

    region_h = rng.randint(REGION_SIZE_RANGE[0], max_region_h)
    region_w = rng.randint(REGION_SIZE_RANGE[0], max_region_w)
    height = region_rows * region_h + (region_rows - 1) * DIVIDER_WIDTH
    width = region_cols * region_w + (region_cols - 1) * DIVIDER_WIDTH
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    expected = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    for r in range(1, region_rows):
        rr = r * region_h + (r - 1) * DIVIDER_WIDTH
        grid[rr : rr + DIVIDER_WIDTH, :] = DIVIDER_COLOR
        expected[rr : rr + DIVIDER_WIDTH, :] = DIVIDER_COLOR
    for c in range(1, region_cols):
        cc = c * region_w + (c - 1) * DIVIDER_WIDTH
        grid[:, cc : cc + DIVIDER_WIDTH] = DIVIDER_COLOR
        expected[:, cc : cc + DIVIDER_WIDTH] = DIVIDER_COLOR

    placement = _pair_regions(region_rows, region_cols, colors, rng)
    regions_for_color: dict[int, list[tuple[int, int]]] = {}
    for pos, color in placement.items():
        regions_for_color.setdefault(color, []).append(pos)

    for color, region_positions in regions_for_color.items():
        built = _make_fragments(region_h, region_w, rng)
        if built is None:
            return None
        full, mask_a, stored_b = built

        assignments = [(mask_a, full), (stored_b, np.rot90(full, 2))]
        if rng.choice([True, False]):
            assignments.reverse()

        for (rr_idx, cc_idx), (fragment, completed) in zip(region_positions, assignments):
            r0 = rr_idx * (region_h + DIVIDER_WIDTH)
            c0 = cc_idx * (region_w + DIVIDER_WIDTH)
            patch = np.full((region_h, region_w), BACKGROUND_COLOR, dtype=int)
            patch[fragment] = color
            grid[r0 : r0 + region_h, c0 : c0 + region_w] = patch

            solved = np.full((region_h, region_w), BACKGROUND_COLOR, dtype=int)
            solved[completed] = color
            expected[r0 : r0 + region_h, c0 : c0 + region_w] = solved

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if not np.array_equal(out, expected):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        candidate = _build_case(rng)
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid fragmented-pair grid")
